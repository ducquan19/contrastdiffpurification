"""Denoisers of diffusion denoised smoothing (DDS) and WA-DDS (docs/wadds.md).

The base classifier of randomized smoothing is ``h(x~) = argmax f(D(x~))`` for ONE noisy input
``x~ = x + sigma * xi`` (pixel scale [0, 1], not clipped). Following Carlini et al. (ICLR 2023),
``x~`` is mapped to the VP-SDE time ``t_sigma`` with

    alpha_bar(t_sigma) = 1 / (1 + sigma_d^2),   sigma_d = 2 sigma,   x_t = sqrt(alpha_bar) (2 x~ - 1),

which is exactly a sample of the forward process at ``t_sigma`` started from ``2 x - 1``.

Methods (``DDSDenoiser(method=...)``):

``onestep``    Carlini et al.: the Tweedie posterior mean ``m = (x_t - sqrt(1-ab) eps(x_t)) / sqrt(ab)``.
``multistep``  ``steps`` DDIM steps (stochasticity ``eta``) from ``t_sigma`` to ~0, no anchor
               (the single-run denoiser of DensePure).
``wadds``      the same DDIM steps, but at every step inside the anchor window the Tweedie
               estimate is replaced by its Wiener-anchored version

                   x0~ = F^-1[ gamma W * F(m) + (1 - gamma W) * F(x0_hat) ],   W = P / (P + sigma_d^2)

               and the noise prediction is made consistent with it, ``eps~ = (x_t - sqrt(ab) x0~) /
               sqrt(1-ab)`` (the noise-correction mechanism of src/guidance/spectral.py, with the
               posterior mean ``m`` as the reference instead of the purifier input).

Validity of the certificate: ``m``, the gain ``W`` (a fixed dataset statistic) and the DDIM noise
depend on ``x~`` and on randomness independent of ``x`` only, so Cohen et al.'s theorem applies to
the smoothed classifier. Never anchor to the clean / un-noised input, and never give several
particles different smoothing noise (the radius would shrink by about sqrt(K)).

The DDIM step count is the number of score-network evaluations (NFE): the first step reuses the
evaluation at ``t_sigma`` that produced ``m``.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Sequence, Tuple

import torch

from src.models.diffusion.sampler import BETA_MAX, BETA_MIN, alpha_bar_cont, make_time_grid

METHODS = ("onestep", "multistep", "wadds")
DIAG_VARIANTS = ("onestep", "multistep", "posthoc", "posthoc_rev", "wadds")


def sigma_to_time(sigma: float, beta_min: float = BETA_MIN, beta_max: float = BETA_MAX) -> float:
    """Continuous VP time with ``alpha_bar(t) = 1 / (1 + (2 sigma)^2)`` (closed form)."""
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    c = math.log1p((2.0 * sigma) ** 2)               # -log alpha_bar
    a, b = 0.5 * (beta_max - beta_min), beta_min       # -log alpha_bar(t) = a t^2 + b t
    t = (-b + math.sqrt(b * b + 4 * a * c)) / (2 * a)
    if t > 1.0:
        raise ValueError(f"sigma={sigma} needs t={t:.3f} > 1, beyond the diffusion horizon")
    return t


def spectral_blend(base: torch.Tensor, ref: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
    """``F^-1[ gain * F(ref) + (1 - gain) * F(base) ]`` (real part), images in [-1, 1].

    ``gain`` broadcasts against (N, C, H, W): (C, H, W) or (1, H, W) in fft2 order.
    """
    spec = torch.fft.fft2(ref - base, norm="ortho")
    return base + torch.fft.ifft2(gain * spec, norm="ortho").real


class DDSDenoiser:
    """Denoiser of noisy images ``x~`` (B, C, H, W) on the [0, 1] scale; returns [0, 1] images.

    Args:
        eps_fn: ``eps_fn(x, t_vec) -> eps`` of the score network (x in [-1, 1], continuous time).
        sigma: smoothing noise level on the [0, 1] scale.
        method: one of ``METHODS``.
        steps: DDIM steps (``multistep`` / ``wadds``).
        eta: DDIM stochasticity (1 ~ ancestral sampling, 0 = deterministic probability flow).
        gain: Wiener gain W (``spectrum.wiener_gain``); required by ``wadds`` and the diagnostics.
        gamma: anchor strength in [0, 1] (``gamma W`` is the per-frequency weight of ``m``).
        anchor_window: anchor only at steps with ``t`` in ``[lo, hi] * t_sigma``.
        t_end: final time of the DDIM grid.
    """

    def __init__(self, eps_fn: Callable, *, sigma: float, method: str = "wadds", steps: int = 10,
                 eta: float = 1.0, gain: Optional[torch.Tensor] = None, gamma: float = 1.0,
                 anchor_window: Tuple[float, float] = (0.0, 1.0), t_end: float = 1e-5,
                 beta_min: float = BETA_MIN, beta_max: float = BETA_MAX):
        if method not in METHODS:
            raise ValueError(f"unknown method '{method}'; available: {', '.join(METHODS)}")
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        if eta < 0:
            raise ValueError(f"eta must be >= 0, got {eta}")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")
        if method == "wadds" and gain is None:
            raise ValueError("method 'wadds' needs the Wiener gain (gain=...)")
        self.eps_fn = eps_fn
        self.sigma = float(sigma)
        self.method = method
        self.steps = int(steps)
        self.eta = float(eta)
        self.gain = gain
        self.gamma = float(gamma)
        self.anchor_window = tuple(float(v) for v in anchor_window)
        self.t_end = float(t_end)
        self.beta_min, self.beta_max = beta_min, beta_max
        self.t_sigma = sigma_to_time(self.sigma, beta_min, beta_max)
        if not self.t_sigma > self.t_end:
            raise ValueError(f"t_sigma={self.t_sigma} must exceed t_end={self.t_end}")

    # ------------------------------------------------------------------ helpers
    @property
    def nfe(self) -> int:
        """Score-network evaluations per noisy input."""
        return 1 if self.method == "onestep" else self.steps

    def _ab(self, t: float) -> float:
        return float(alpha_bar_cont(t, self.beta_min, self.beta_max))

    def _eps(self, x: torch.Tensor, t: float) -> torch.Tensor:
        return self.eps_fn(x, torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype))

    def _weight(self, ref: torch.Tensor) -> torch.Tensor:
        return (self.gamma * self.gain).to(device=ref.device, dtype=ref.dtype)

    def _in_window(self, t: float) -> bool:
        lo, hi = self.anchor_window
        return lo * self.t_sigma - 1e-12 <= t <= hi * self.t_sigma + 1e-12

    def start(self, x_noisy: torch.Tensor):
        """``x_t`` at ``t_sigma``, the score evaluation there and the posterior mean ``m``."""
        ab = self._ab(self.t_sigma)
        x_t = math.sqrt(ab) * (2.0 * x_noisy - 1.0)
        eps0 = self._eps(x_t, self.t_sigma)
        m = (x_t - math.sqrt(1.0 - ab) * eps0) / math.sqrt(ab)
        return x_t, eps0, m

    def draw_step_noise(self, shape, generator: Optional[torch.Generator], device, dtype):
        return torch.randn((self.steps, *shape), generator=generator, device=device, dtype=dtype)

    def trajectory(self, x_t: torch.Tensor, eps0: torch.Tensor, m: torch.Tensor, zs: torch.Tensor,
                   anchored: bool) -> torch.Tensor:
        """DDIM from ``t_sigma`` towards ``t_end``; returns the clean estimate of the last step.

        The last step's (anchored) Tweedie estimate ``x0`` is returned instead of the state at
        ``t_end``, which still carries noise of standard deviation ``sqrt(1 - ab(t_end))``
        (the usual "denoise at the final step"). In [-1, 1], unclipped.
        """
        ts = make_time_grid(self.t_sigma, self.steps, self.t_end).tolist()
        x = x_t
        x0 = None
        for i in range(self.steps):
            t, s = ts[i], ts[i + 1]
            ab_t, ab_s = self._ab(t), self._ab(s)
            eps = eps0 if i == 0 else self._eps(x, t)
            x0 = (x - math.sqrt(1.0 - ab_t) * eps) / math.sqrt(ab_t)
            if anchored and self._in_window(t):
                x0 = spectral_blend(x0, m, self._weight(m))
                eps = (x - math.sqrt(ab_t) * x0) / math.sqrt(1.0 - ab_t)
            sig = self.eta * math.sqrt(max((1 - ab_s) / (1 - ab_t), 0.0)) * math.sqrt(max(1 - ab_t / ab_s, 0.0))
            c_eps = math.sqrt(max(1.0 - ab_s - sig ** 2, 0.0))
            x = math.sqrt(ab_s) * x0 + c_eps * eps + sig * zs[i]
        return x0

    # ------------------------------------------------------------------ main entries
    @torch.no_grad()
    def __call__(self, x_noisy: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        x_t, eps0, m = self.start(x_noisy)
        if self.method == "onestep":
            out = m
        else:
            zs = self.draw_step_noise(x_noisy.shape, generator, x_noisy.device, x_noisy.dtype)
            out = self.trajectory(x_t, eps0, m, zs, anchored=(self.method == "wadds"))
        return ((out.clamp(-1, 1) + 1) * 0.5)

    @torch.no_grad()
    def diagnose(self, x_noisy: torch.Tensor, generator: Optional[torch.Generator] = None,
                 variants: Sequence[str] = DIAG_VARIANTS) -> Dict[str, torch.Tensor]:
        """All diagnostic outputs of gate G0 from the SAME noisy input and DDIM noise.

        ``onestep`` = m; ``multistep`` = unanchored sample s; ``posthoc`` = low band of m + high
        band of s (``W``-blend of the two final images); ``posthoc_rev`` = low band of s + high
        band of m; ``wadds`` = the anchored trajectory. Common random numbers make the paired
        differences between variants much less noisy than independent runs.
        """
        if self.gain is None:
            raise ValueError("diagnose() needs the Wiener gain")
        unknown = set(variants) - set(DIAG_VARIANTS)
        if unknown:
            raise ValueError(f"unknown variants {sorted(unknown)}; available: {DIAG_VARIANTS}")
        x_t, eps0, m = self.start(x_noisy)
        zs = self.draw_step_noise(x_noisy.shape, generator, x_noisy.device, x_noisy.dtype)
        out = {"onestep": m}
        if {"multistep", "posthoc", "posthoc_rev"} & set(variants):
            s = self.trajectory(x_t, eps0, m, zs, anchored=False)
            w = self._weight(m)
            out["multistep"] = s
            out["posthoc"] = spectral_blend(s, m, w)
            out["posthoc_rev"] = spectral_blend(m, s, w)
        if "wadds" in variants:
            out["wadds"] = self.trajectory(x_t, eps0, m, zs, anchored=True)
        return {k: ((v.clamp(-1, 1) + 1) * 0.5) for k, v in out.items() if k in variants}


def make_base_classifier(denoiser: DDSDenoiser, classifier: Callable[[torch.Tensor], torch.Tensor],
                         generator_ref: list) -> Callable[[torch.Tensor], torch.Tensor]:
    """``h(x~) = argmax classifier(denoiser(x~))``; ``generator_ref[0]`` is the current generator.

    The generator is read at call time so that ``certify`` can use one generator per image for
    both the smoothing noise and the DDIM noise.
    """
    @torch.no_grad()
    def base(x_noisy: torch.Tensor) -> torch.Tensor:
        return classifier(denoiser(x_noisy, generator_ref[0])).argmax(1)
    return base
