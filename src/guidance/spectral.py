"""Frequency-domain guidance for the particle sampler: SRSG and a FreqPure re-implementation.

Both act on the Tweedie estimate ``x0_hat`` at every reverse step and use the purifier's *input*
``x_ref`` (the possibly adversarial image) as the reference (docs/srsg.md):

* ``SpectralGuidance`` (SRSG, ``--guidance srsg``): partial, soft replacement of a *random* low
  band, ``x~0 = x0_hat - gamma * F^-1( M_r * F(x0_hat - x_ref) )`` with the soft radial mask
  ``M_r = sigmoid((r - |f|) / tau)``. ``r`` and ``gamma`` are drawn once per particle and per
  purification from a generator seeded by the image's seed, so the purification of an image does
  not depend on the batch and the anchored band changes from one inference to the next.
  ``mode='amp'`` pulls only the amplitude spectrum and keeps the phase of ``x0_hat``.
* ``FreqPureGuidance`` (``--guidance freqpure``; Pei et al., ICML 2025): hard exchange of the
  amplitude inside radius ``D_A`` and projection of the phase inside radius ``D_P`` into
  ``[phase_ref - delta, phase_ref + delta]``.

The corrected estimate is returned to the sampler as a noise correction
``d = (x_t - sqrt(ab) x~0) / sqrt(1 - ab) - eps``, so the DDIM step uses ``x~0`` and the matching
noise. Everything is differentiable (FFT, sigmoid, abs/angle, clamp), and ``x_ref`` keeps its
graph: a full-gradient attack sees how its perturbation is carried through the anchored band.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional

import torch

from src.models.diffusion.sampler import alpha_bar_cont

SRSG_MODES = ("complex", "amp")
_AMP_EPS = 1e-8


# --------------------------------------------------------------------------------------------
# spectral helpers
# --------------------------------------------------------------------------------------------

def radial_frequency(h: int, w: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """(h, w) magnitude of the 2-D DFT frequency in cycles per image, in ``fft2`` order.

    Equal to the distance to the centre of the fftshift-ed spectrum (FreqPure's ``D(u, v)``).
    """
    fy = torch.fft.fftfreq(h, d=1.0 / h, device=device, dtype=dtype)
    fx = torch.fft.fftfreq(w, d=1.0 / w, device=device, dtype=dtype)
    return torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)


def soft_lowpass(radius: torch.Tensor, r: torch.Tensor, tau: float) -> torch.Tensor:
    """Soft radial low-pass mask (N, 1, h, w) for per-row radii ``r`` (N,)."""
    return torch.sigmoid((r.view(-1, 1, 1, 1) - radius) / tau)


def hard_lowpass(radius: torch.Tensor, r: float) -> torch.Tensor:
    """Ideal low-pass mask (h, w): 1 where ``|f| < r``."""
    return (radius < r).to(radius.dtype)


def gaussian_lowpass(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Filter ``x`` (N, C, h, w) with the Gaussian transfer function ``exp(-|f|^2 / (2 sigma^2))``.

    The kernel is (up to truncation at the Nyquist frequency) non-negative and sums to 1, so the
    output stays within the input's l_inf bound; callers clamp anyway. Used by the low-frequency
    PGD attack (``--pgd_lowpass_sigma``).
    """
    radius = radial_frequency(x.shape[-2], x.shape[-1], device=x.device, dtype=x.dtype)
    g = torch.exp(-radius ** 2 / (2.0 * sigma ** 2))
    return torch.fft.ifft2(torch.fft.fft2(x, norm="ortho") * g, norm="ortho").real


def _to_noise_correction(x0_new, *, x_t, eps, t: float):
    ab = float(alpha_bar_cont(t))
    eps_new = (x_t - math.sqrt(ab) * x0_new) / math.sqrt(1.0 - ab)
    return eps_new - eps


def _per_image_uniform(seeds: torch.Tensor, n_particles: int, n: int, salt: int) -> torch.Tensor:
    """(B * K, n) uniforms in [0, 1); row ``b * K + k`` depends only on ``seeds[b]`` and ``k``."""
    rows = []
    for s in torch.as_tensor(seeds, dtype=torch.int64).cpu().tolist():
        g = torch.Generator(device="cpu")
        g.manual_seed((int(s) ^ salt) & 0x7FFFFFFFFFFFFFFF)
        rows.append(torch.rand((n_particles, n), generator=g))
    return torch.cat(rows, dim=0)


class _ReferenceMixin:
    """Holds the purifier input set by ``ParticleSampler`` before the reverse loop."""

    def set_reference(self, *, x_ref: torch.Tensor, seeds: torch.Tensor, n_particles: int):
        self.x_ref = x_ref                      # (B * K, C, H, W) in [-1, 1], keeps its graph
        self.n_particles = int(n_particles)
        self._draw(seeds, n_particles)

    def _draw(self, seeds, n_particles):       # overridden by randomized guidances
        pass

    def reset(self):
        self.x_ref = None
        self.last_stats = {}

    @staticmethod
    def _active(window, t: float, t_star: float) -> bool:
        lo, hi = window
        return lo * t_star - 1e-12 <= t <= hi * t_star + 1e-12


# --------------------------------------------------------------------------------------------
# SRSG
# --------------------------------------------------------------------------------------------

@dataclass
class SRSGConfig:
    r_min: float = 2.0
    r_max: float = 6.0
    tau: float = 1.0
    gamma_min: float = 0.3
    gamma_max: float = 0.9
    mode: str = "complex"
    t_min: float = 0.0
    t_max: float = 1.0

    def __post_init__(self):
        if not 0 < self.r_min <= self.r_max:
            raise ValueError(f"need 0 < r_min <= r_max, got {self.r_min}, {self.r_max}")
        if not 0 <= self.gamma_min <= self.gamma_max <= 1:
            raise ValueError(f"need 0 <= gamma_min <= gamma_max <= 1, got {self.gamma_min}, {self.gamma_max}")
        if self.tau <= 0:
            raise ValueError(f"tau must be > 0, got {self.tau}")
        if self.mode not in SRSG_MODES:
            raise ValueError(f"unknown mode '{self.mode}'; available: {', '.join(SRSG_MODES)}")

    def as_dict(self):
        return asdict(self)


class SpectralGuidance(_ReferenceMixin):
    """Soft Randomized Spectral Guidance (docs/srsg.md, section 2)."""

    SALT = 0x5DEECE66D

    def __init__(self, config: Optional[SRSGConfig] = None):
        self.config = config or SRSGConfig()
        self.x_ref = None
        self.r = self.gamma = None
        self.last_stats = {}

    def _draw(self, seeds, n_particles):
        c = self.config
        u = _per_image_uniform(seeds, n_particles, 2, self.SALT)
        self.r = c.r_min + (c.r_max - c.r_min) * u[:, 0]
        self.gamma = c.gamma_min + (c.gamma_max - c.gamma_min) * u[:, 1]

    def corrected_x0(self, x0_hat: torch.Tensor) -> torch.Tensor:
        if self.x_ref is None:
            raise RuntimeError("SpectralGuidance needs set_reference() before the reverse loop")
        c = self.config
        radius = radial_frequency(x0_hat.shape[-2], x0_hat.shape[-1], device=x0_hat.device,
                                  dtype=x0_hat.dtype)
        mask = soft_lowpass(radius, self.r.to(x0_hat), c.tau)            # (N, 1, H, W)
        gamma = self.gamma.to(x0_hat).view(-1, 1, 1, 1)
        if c.mode == "complex":
            spec = torch.fft.fft2(x0_hat - self.x_ref, norm="ortho")
            return x0_hat - gamma * torch.fft.ifft2(mask * spec, norm="ortho").real
        X = torch.fft.fft2(x0_hat, norm="ortho")
        a_h = X.abs()
        a_r = torch.fft.fft2(self.x_ref, norm="ortho").abs()
        a_new = a_h - gamma * mask * (a_h - a_r)
        return torch.fft.ifft2(X * (a_new / (a_h + _AMP_EPS)), norm="ortho").real

    def __call__(self, *, x_t, x0_hat, eps, t, t_star, n_particles):
        if not self._active((self.config.t_min, self.config.t_max), t, t_star):
            return None
        x0_new = self.corrected_x0(x0_hat)
        return _to_noise_correction(x0_new, x_t=x_t, eps=eps, t=t)


# --------------------------------------------------------------------------------------------
# FreqPure (baseline)
# --------------------------------------------------------------------------------------------

@dataclass
class FreqPureConfig:
    d_a: float = 3.0
    d_p: float = 2.0
    delta: float = 0.2

    def as_dict(self):
        return asdict(self)


def _wrap(phase: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(phase), torch.cos(phase))


class FreqPureGuidance(_ReferenceMixin):
    """Amplitude exchange + phase projection of FreqPure, applied at every reverse step."""

    def __init__(self, config: Optional[FreqPureConfig] = None):
        self.config = config or FreqPureConfig()
        self.x_ref = None
        self.last_stats = {}

    def corrected_x0(self, x0_hat: torch.Tensor) -> torch.Tensor:
        if self.x_ref is None:
            raise RuntimeError("FreqPureGuidance needs set_reference() before the reverse loop")
        c = self.config
        radius = radial_frequency(x0_hat.shape[-2], x0_hat.shape[-1], device=x0_hat.device,
                                  dtype=x0_hat.dtype)
        h_a, h_p = hard_lowpass(radius, c.d_a), hard_lowpass(radius, c.d_p)
        X = torch.fft.fft2(x0_hat, norm="ortho") + _AMP_EPS
        Xr = torch.fft.fft2(self.x_ref, norm="ortho") + _AMP_EPS
        a_h, p_h, a_r, p_r = X.abs(), X.angle(), Xr.abs(), Xr.angle()
        a_new = a_h * (1 - h_a) + a_r * h_a
        p_proj = p_r + torch.clamp(_wrap(p_h - p_r), -c.delta, c.delta)
        p_new = p_proj * h_p + p_h * (1 - h_p)
        return torch.fft.ifft2(torch.polar(a_new, p_new), norm="ortho").real

    def __call__(self, *, x_t, x0_hat, eps, t, t_star, n_particles):
        x0_new = self.corrected_x0(x0_hat)
        return _to_noise_correction(x0_new, x_t=x_t, eps=eps, t=t)


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------

def add_spectral_args(parser):
    g = parser.add_argument_group("spectral guidance (--guidance srsg / freqpure; docs/srsg.md)")
    d = SRSGConfig()
    g.add_argument("--srsg_r_min", type=float, default=d.r_min, help="smallest anchored radius (cycles/image)")
    g.add_argument("--srsg_r_max", type=float, default=d.r_max, help="largest anchored radius")
    g.add_argument("--srsg_tau", type=float, default=d.tau, help="softness of the radial mask")
    g.add_argument("--srsg_gamma_min", type=float, default=d.gamma_min, help="smallest pull fraction")
    g.add_argument("--srsg_gamma_max", type=float, default=d.gamma_max, help="largest pull fraction")
    g.add_argument("--srsg_mode", type=str, default=d.mode, choices=SRSG_MODES)
    g.add_argument("--srsg_t_min", type=float, default=d.t_min, help="guidance window start (fraction of t*)")
    g.add_argument("--srsg_t_max", type=float, default=d.t_max, help="guidance window end (fraction of t*)")
    f = FreqPureConfig()
    g.add_argument("--freqpure_da", type=float, default=f.d_a, help="FreqPure amplitude radius D_A")
    g.add_argument("--freqpure_dp", type=float, default=f.d_p, help="FreqPure phase radius D_P")
    g.add_argument("--freqpure_delta", type=float, default=f.delta, help="FreqPure phase range delta")


def build_spectral_guidance(args):
    if args.guidance == "srsg":
        return SpectralGuidance(SRSGConfig(
            r_min=args.srsg_r_min, r_max=args.srsg_r_max, tau=args.srsg_tau,
            gamma_min=args.srsg_gamma_min, gamma_max=args.srsg_gamma_max, mode=args.srsg_mode,
            t_min=args.srsg_t_min, t_max=args.srsg_t_max))
    if args.guidance == "freqpure":
        return FreqPureGuidance(FreqPureConfig(d_a=args.freqpure_da, d_p=args.freqpure_dp,
                                               delta=args.freqpure_delta))
    raise ValueError(f"not a spectral guidance: {args.guidance}")
