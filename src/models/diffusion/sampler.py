"""Differentiable reverse diffusion for (multi-particle) purification.

This is the solver behind ``--diffusion_type particle``. It differs from ``sde.py`` / ``ode.py``
(``torchsde`` / ``torchdiffeq`` adjoint solvers) in four ways, each needed by the PCP method and by
a full-gradient evaluation (Lee & Kim, ICCV 2023):

* an explicit loop over ``steps`` reverse steps, so that an attack back-propagates *directly*
  through every step (no adjoint re-evaluation, which would also re-run any stateful guidance);
* ``n_particles`` (K) independent purifications of every input image, laid out image-major
  (row ``b * K + k`` is particle ``k`` of image ``b``, the layout of ``repeat_interleave``);
* every random draw (forward noise and every reverse step) comes from a generator seeded *per
  image*, so the purification of one image never depends on the other images of the batch;
* an optional ``guidance`` callable that returns a correction of the predicted noise.

Nothing here imports ``score_sde``: the score network enters only through ``eps_fn``, which makes
the module testable on CPU with toy networks (tests/test_sampler.py).

Conventions
-----------
Images inside the sampler are in [-1, 1]; ``ParticleSampler.forward`` takes and returns [0, 1].
Time is the continuous VP-SDE time ``t`` in [0, 1] with ``alpha_bar(t) = exp(-0.5 (b1-b0) t^2 - b0 t)``.
``t_star`` is given on the discrete 1000-step scale, like ``--t`` (t* = 100 -> t = 0.1); the
forward noising uses the *discrete* ``alpha_bar[t_star - 1]``, exactly as
``DiffusionPurifier.image_editing_sample``, and the reverse process starts at ``t_star / 1000``.

Guidance sign: a correction ``d`` is added to the predicted noise, ``eps~ = eps + d``. Since
``score = -eps / sqrt(1 - alpha_bar)``, ``d = +c * grad L`` (c > 0) lowers ``L`` along the reverse
process; this is the convention of classifier guidance and of ContrastDiffPure (``out += delta``).
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Protocol

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

BETA_MIN = 0.1
BETA_MAX = 20.0
N_DISCRETE = 1000
SOLVERS = ("ddim", "em")
CHECKPOINT_MODES = ("none", "eps")


# --------------------------------------------------------------------------------------------
# noise schedule
# --------------------------------------------------------------------------------------------

def alpha_bar_cont(t, beta_min: float = BETA_MIN, beta_max: float = BETA_MAX) -> torch.Tensor:
    """Continuous VP-SDE ``alpha_bar(t)``; matches ``alphas_cumprod_cont`` of the original code."""
    t = torch.as_tensor(t, dtype=torch.float64)
    return torch.exp(-0.5 * (beta_max - beta_min) * t ** 2 - beta_min * t)


def beta_cont(t, beta_min: float = BETA_MIN, beta_max: float = BETA_MAX) -> torch.Tensor:
    t = torch.as_tensor(t, dtype=torch.float64)
    return beta_min + t * (beta_max - beta_min)


def discrete_alpha_bar(n: int, beta_min: float = BETA_MIN, beta_max: float = BETA_MAX,
                       N: int = N_DISCRETE) -> float:
    """``alpha_bar`` of the discrete forward noising at level ``n`` (1-based), as in base.py."""
    if not 1 <= n <= N:
        raise ValueError(f"noise level must be in [1, {N}], got {n}")
    betas = torch.linspace(beta_min / N, beta_max / N, N, dtype=torch.float64)
    return float((1 - betas).cumprod(0)[n - 1])


def make_time_grid(t_start: float, steps: int, t_end: float = 1e-5) -> torch.Tensor:
    """``steps + 1`` decreasing times from ``t_start`` to ``t_end`` (uniform spacing)."""
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if not t_start > t_end:
        raise ValueError(f"t_start ({t_start}) must be larger than t_end ({t_end})")
    return torch.linspace(t_start, t_end, steps + 1, dtype=torch.float64)


# --------------------------------------------------------------------------------------------
# per-image randomness
# --------------------------------------------------------------------------------------------

def fresh_seeds(batch_size: int, device=None) -> torch.Tensor:
    """One seed per image, drawn from the global torch RNG (reproducible under ``--seed``)."""
    return torch.randint(0, 2 ** 62, (batch_size,), dtype=torch.int64, device="cpu")


def per_sample_noise(seeds: torch.Tensor, n_particles: int, n_draws: int, img_shape,
                     device, dtype=torch.float32) -> torch.Tensor:
    """Standard normal noise of shape ``(n_draws, B * K, *img_shape)``.

    Image ``b`` uses only ``seeds[b]``, so its noise does not depend on its position in the batch
    or on the other images. The generator lives on ``device`` (a CUDA generator on GPU).
    """
    seeds = torch.as_tensor(seeds, dtype=torch.int64).cpu()
    gen_device = torch.device(device)
    chunks = []
    for s in seeds.tolist():
        g = torch.Generator(device=gen_device)
        g.manual_seed(int(s))
        chunks.append(torch.randn((n_draws, n_particles, *img_shape), generator=g,
                                  device=gen_device, dtype=dtype))
    noise = torch.stack(chunks, dim=1)                      # (n_draws, B, K, C, H, W)
    return noise.reshape(n_draws, len(seeds) * n_particles, *img_shape)


# --------------------------------------------------------------------------------------------
# guidance protocol
# --------------------------------------------------------------------------------------------

class Guidance(Protocol):
    """A correction of the predicted noise at one reverse step.

    Called with keyword arguments only; returns a tensor shaped like ``eps`` (or ``None`` for no
    correction at this step). ``t`` is the continuous time of the step, ``x0_hat`` the Tweedie
    estimate of the clean image (in [-1, 1]) and ``t_star`` the continuous start time of the run.
    An optional ``reset()`` is called once at the start of every purification.
    """

    def __call__(self, *, x_t: torch.Tensor, x0_hat: torch.Tensor, eps: torch.Tensor, t: float,
                 t_star: float, n_particles: int) -> Optional[torch.Tensor]:
        ...


# --------------------------------------------------------------------------------------------
# sampler
# --------------------------------------------------------------------------------------------

class ParticleSampler(nn.Module):
    """K-particle purification with an explicit, differentiable reverse loop.

    Args:
        eps_fn: ``eps_fn(x, t_vec) -> eps``; ``x`` in [-1, 1], ``t_vec`` the continuous time of
            every row. For the CIFAR-10 score_sde model see ``ParticlePurifier.eps_fn``.
        t_star: forward noise level on the 1000-step scale (``--t``).
        steps: number of reverse steps from ``t_star / 1000`` to ``t_end``.
        solver: ``'ddim'`` (DDIM with stochasticity ``eta``; eta = 1 ~ DDPM ancestral) or
            ``'em'`` (Euler-Maruyama of the reverse VP-SDE, the discretisation DiffPure uses).
        eta: DDIM stochasticity (ignored by ``'em'``).
        n_particles: K.
        guidance: optional ``Guidance``.
        checkpoint: ``'eps'`` recomputes the score network in the backward pass (saves most of
            the memory of a full-gradient attack); ``'none'`` stores everything.
        t_end: final time of the reverse process.
    """

    def __init__(self, eps_fn: Callable, *, t_star: int = 100, steps: int = 20,
                 solver: str = "ddim", eta: float = 1.0, n_particles: int = 1,
                 guidance: Optional[Guidance] = None, checkpoint: str = "eps",
                 t_end: float = 1e-5, beta_min: float = BETA_MIN, beta_max: float = BETA_MAX):
        super().__init__()
        if solver not in SOLVERS:
            raise ValueError(f"unknown solver '{solver}'; available: {', '.join(SOLVERS)}")
        if checkpoint not in CHECKPOINT_MODES:
            raise ValueError(f"unknown checkpoint mode '{checkpoint}'; available: {', '.join(CHECKPOINT_MODES)}")
        if n_particles < 1:
            raise ValueError(f"n_particles must be >= 1, got {n_particles}")
        if eta < 0:
            raise ValueError(f"eta must be >= 0, got {eta}")
        self.eps_fn = eps_fn
        self.t_star = int(t_star)
        self.steps = int(steps)
        self.solver = solver
        self.eta = float(eta)
        self.n_particles = int(n_particles)
        self.guidance = guidance
        self.checkpoint_mode = checkpoint
        self.t_end = float(t_end)
        self.beta_min, self.beta_max = beta_min, beta_max
        self.last_stats: dict = {}

    # ------------------------------------------------------------------ helpers
    def _ab(self, t: float) -> float:
        return float(alpha_bar_cont(t, self.beta_min, self.beta_max))

    def _eps(self, x: torch.Tensor, t: float) -> torch.Tensor:
        t_vec = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
        if self.checkpoint_mode == "eps" and torch.is_grad_enabled() and x.requires_grad:
            return torch_checkpoint(self.eps_fn, x, t_vec, use_reentrant=False)
        return self.eps_fn(x, t_vec)

    def _guided_eps(self, x: torch.Tensor, t: float, t_star_c: float):
        eps = self._eps(x, t)
        if self.guidance is None:
            return eps, None
        ab = self._ab(t)
        x0_hat = (x - math.sqrt(1.0 - ab) * eps) / math.sqrt(ab)
        d = self.guidance(x_t=x, x0_hat=x0_hat, eps=eps, t=t, t_star=t_star_c,
                          n_particles=self.n_particles)
        if d is None:
            return eps, None
        return eps + d, d

    def _step(self, x, eps_t, t: float, s: float, z):
        """One reverse step from ``t`` to ``s < t`` with the (guided) noise prediction ``eps_t``."""
        ab_t, ab_s = self._ab(t), self._ab(s)
        if self.solver == "ddim":
            x0 = (x - math.sqrt(1.0 - ab_t) * eps_t) / math.sqrt(ab_t)
            sigma = self.eta * math.sqrt(max((1 - ab_s) / (1 - ab_t), 0.0)) * math.sqrt(max(1 - ab_t / ab_s, 0.0))
            c_eps = math.sqrt(max(1.0 - ab_s - sigma ** 2, 0.0))
            return math.sqrt(ab_s) * x0 + c_eps * eps_t + sigma * z
        # Euler-Maruyama of dx = [-0.5 b x - b score] dt + sqrt(b) dw, integrated backwards
        h = t - s
        b = float(beta_cont(t, self.beta_min, self.beta_max))
        score = -eps_t / math.sqrt(1.0 - ab_t)
        return x + h * (0.5 * b * x + b * score) + math.sqrt(b * h) * z

    # ------------------------------------------------------------------ main entry
    def purify_centered(self, x: torch.Tensor, seeds: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Purify ``x`` in [-1, 1], shape (B, C, H, W) -> (B * K, C, H, W) in [-1, 1]."""
        if x.ndim != 4:
            raise ValueError(f"expected a (B, C, H, W) batch, got shape {tuple(x.shape)}")
        B, K = x.shape[0], self.n_particles
        if seeds is None:
            seeds = fresh_seeds(B)
        if len(seeds) != B:
            raise ValueError(f"need one seed per image: {len(seeds)} seeds for {B} images")

        if self.guidance is not None and hasattr(self.guidance, "reset"):
            self.guidance.reset()

        noise = per_sample_noise(seeds, K, self.steps + 1, x.shape[1:], x.device, x.dtype)
        a = discrete_alpha_bar(self.t_star, self.beta_min, self.beta_max)
        xk = x.repeat_interleave(K, dim=0)
        x_t = math.sqrt(a) * xk + math.sqrt(1.0 - a) * noise[0]

        t_star_c = self.t_star / N_DISCRETE
        ts = make_time_grid(t_star_c, self.steps, self.t_end).tolist()
        d_norms = []
        for i in range(self.steps):
            t, s = ts[i], ts[i + 1]
            eps_t, d = self._guided_eps(x_t, t, t_star_c)
            if d is not None:
                d_norms.append(float(d.detach().flatten(1).norm(dim=1).mean()))  # SAFE: logging only
            x_t = self._step(x_t, eps_t, t, s, noise[i + 1])
        self.last_stats = {"guidance_norm_mean": (sum(d_norms) / len(d_norms)) if d_norms else 0.0,
                           "guided_steps": len(d_norms)}
        return x_t

    def forward(self, x01: torch.Tensor, seeds: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Purify ``x01`` in [0, 1]: (B, C, H, W) -> (B, K, C, H, W), in [0, 1] up to the solver."""
        out = self.purify_centered(x01 * 2 - 1, seeds)
        return ((out + 1) * 0.5).view(x01.shape[0], self.n_particles, *x01.shape[1:])
