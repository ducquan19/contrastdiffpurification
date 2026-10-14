"""Particle-Contrastive guidance (PCP): contrast the K purifications of the *same* image.

ContrastDiffPure draws its negatives from the other images of the minibatch, so the
purification of an image depends on the batch it is in. PCP couples only the K particles that
purify one image, and uses a fixed bank of clean prototypes instead of other images:

* ``L_sem`` (pull, semantic): SwAV-style swapped prediction. For particle k with normalised
  feature ``z_k``, the target is the soft prototype assignment of the leave-one-out consensus of
  the other particles, ``q_k = softmax(c_{-k} P^T / tau_t)``; the loss is the cross-entropy
  ``-sum_m q_km log softmax(z_k P^T / tau_s)_m``. The prototypes the consensus agrees on are the
  positives, all others the negatives -- no in-batch negatives and no false negatives.
* ``L_rep`` (push, detail): an RBF repulsion between particles in a low-resolution pixel space,
  ``sum_{k != j} exp(-||phi_k - phi_j||^2 / h)`` with the median bandwidth ``h``, as in particle
  guidance (Corso et al., ICLR 2024). It decorrelates the particles, which is what makes a
  majority vote over them robust.

Both losses are defined *per image* (no averaging over the batch), so neither the direction nor
the magnitude of the guidance of one image depends on the other images.

No gradient masking
-------------------
The guidance is a function ``g(x)`` of the attacked input. In a full-gradient evaluation the
attacker must differentiate through ``g`` itself, including the dependence of the target ``q``
and of the bandwidth ``h`` on ``x``. Therefore:

* the "stop-gradients" that *define* the guidance (``q`` and ``h`` are held fixed when taking
  the gradient of the loss) are implemented analytically -- ``dL/dz`` and ``dL/dphi`` are written
  out -- instead of with ``.detach()``, so ``q`` and ``h`` stay in the autograd graph;
* when the caller tracks gradients (``torch.is_grad_enabled()`` and ``x0_hat.requires_grad``),
  every ``autograd.grad`` uses ``create_graph=True``; otherwise the guidance is computed on a
  detached copy (plain inference, nothing to differentiate);
* tests/test_no_gradient_masking.py rejects any ``detach`` / ``no_grad`` on the data path of
  this file that is not marked ``# SAFE``.

The scale of the correction is set per particle relative to the predicted noise
(``scale_mode='normalized'``: ``||d|| = strength * ||eps||``), or as a classic guidance weight
(``'raw'``: ``d = strength * sqrt(1 - a) / sqrt(a) * grad_x0 L``).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.diffusion.sampler import alpha_bar_cont

SCALE_MODES = ("normalized", "raw")


# --------------------------------------------------------------------------------------------
# analytic gradients of the two losses (used by the guidance and checked against autograd)
# --------------------------------------------------------------------------------------------

def sem_target(z: torch.Tensor, prototypes: torch.Tensor, tau_t: float) -> torch.Tensor:
    """Soft prototype assignment ``q`` of the leave-one-out consensus; ``z`` is (B, K, D) unit.

    With K = 1 the particle's own feature is its target (entropy sharpening towards the nearest
    prototypes, since ``tau_t < tau_s``).
    """
    K = z.shape[1]
    c = (z.sum(dim=1, keepdim=True) - z) if K > 1 else z
    c = F.normalize(c, dim=-1)
    return torch.softmax(c @ prototypes.t() / tau_t, dim=-1)


def sem_loss(z: torch.Tensor, prototypes: torch.Tensor, q: torch.Tensor, tau_s: float) -> torch.Tensor:
    """Per-image ``L_sem`` (B,) for features ``z`` (B, K, D) and targets ``q`` (B, K, M)."""
    logp = torch.log_softmax(z @ prototypes.t() / tau_s, dim=-1)
    return -(q * logp).sum(dim=(-1, -2))


def sem_grad_z(z: torch.Tensor, prototypes: torch.Tensor, tau_s: float, tau_t: float) -> torch.Tensor:
    """``dL_sem/dz`` with ``q`` held fixed: ``(softmax(z P^T / tau_s) - q) P / tau_s``.

    ``q`` keeps its graph, so an attacker differentiating this vector field sees how the target
    moves with the input.
    """
    q = sem_target(z, prototypes, tau_t)
    p = torch.softmax(z @ prototypes.t() / tau_s, dim=-1)
    return ((p - q) @ prototypes) / tau_s


def rep_features(x0: torch.Tensor, n_particles: int, size: int) -> torch.Tensor:
    """Low-resolution pixel features ``phi`` (B, K, C*size*size) of (B*K, C, H, W) images."""
    phi = F.adaptive_avg_pool2d(x0, size).flatten(1)
    return phi.view(x0.shape[0] // n_particles, n_particles, -1)


def rep_bandwidth(d2: torch.Tensor) -> torch.Tensor:
    """Median heuristic ``h = median_{k != j} d2 / log(K + 1)`` per image; ``d2`` is (B, K, K)."""
    K = d2.shape[-1]
    off = ~torch.eye(K, dtype=torch.bool, device=d2.device)
    return (d2[:, off].median(dim=-1).values / math.log(K + 1)).clamp_min(1e-12)


def rep_loss(phi: torch.Tensor, h: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Per-image ``L_rep`` (B,) = mean over ordered pairs k != j of ``exp(-d2_kj / h)``."""
    B, K, _ = phi.shape
    d2 = (phi.unsqueeze(2) - phi.unsqueeze(1)).pow(2).sum(-1)
    if h is None:
        h = rep_bandwidth(d2)
    off = ~torch.eye(K, dtype=torch.bool, device=phi.device)
    kmat = torch.exp(-d2 / h[:, None, None]) * off
    return kmat.sum(dim=(1, 2)) / (K * (K - 1))


def rep_grad_phi(phi: torch.Tensor) -> torch.Tensor:
    """``dL_rep/dphi`` with ``h`` held fixed: ``-4 / (h K (K-1)) * sum_j k_kj (phi_k - phi_j)``."""
    B, K, _ = phi.shape
    diff = phi.unsqueeze(2) - phi.unsqueeze(1)                 # (B, K, K, d): phi_k - phi_j
    d2 = diff.pow(2).sum(-1)
    h = rep_bandwidth(d2)
    off = ~torch.eye(K, dtype=torch.bool, device=phi.device)
    kmat = torch.exp(-d2 / h[:, None, None]) * off
    coef = -4.0 / (h * K * (K - 1))
    return coef[:, None, None] * (kmat.unsqueeze(-1) * diff).sum(dim=2)


# --------------------------------------------------------------------------------------------
# the guidance
# --------------------------------------------------------------------------------------------

class ParticleContrastiveGuidance(nn.Module):
    """``Guidance`` for ``ParticleSampler``: semantic pull to the particles' consensus prototypes
    plus repulsion between the particles.

    Args:
        encoder: frozen feature extractor F, images in [0, 1] -> (N, D) features
            (see ``src/guidance/encoders.py``). Only needed when ``sem_strength > 0``.
        bank: ``PrototypeBank`` with unit prototypes in F's feature space; ``bank.at(t)``
            returns the (M, D) prototypes of the time bucket nearest to ``t``.
        sem_strength, rep_strength: weights of the two corrections (see ``scale_mode``).
        tau_s, tau_t: temperatures of the prediction and of the (sharper) target, tau_t < tau_s.
        sem_t_min, sem_t_max, rep_t_min, rep_t_max: each correction is applied while
            ``t_min * t_star <= t <= t_max * t_star`` (fractions of the start time).
        rep_size: side of the low-resolution space of the repulsion.
        scale_mode: ``'normalized'`` or ``'raw'`` (module docstring).
    """

    def __init__(self, encoder: Optional[nn.Module], bank, *, sem_strength: float = 0.1,
                 rep_strength: float = 0.1, tau_s: float = 0.1, tau_t: float = 0.05,
                 sem_t_min: float = 0.0, sem_t_max: float = 1.0, rep_t_min: float = 0.5,
                 rep_t_max: float = 1.0, rep_size: int = 16, scale_mode: str = "normalized"):
        super().__init__()
        if scale_mode not in SCALE_MODES:
            raise ValueError(f"unknown scale_mode '{scale_mode}'; available: {', '.join(SCALE_MODES)}")
        if sem_strength > 0 and (encoder is None or bank is None):
            raise ValueError("sem_strength > 0 needs an encoder and a prototype bank")
        if not 0 < tau_t <= tau_s:
            raise ValueError(f"need 0 < tau_t <= tau_s, got tau_t={tau_t}, tau_s={tau_s}")
        self.encoder = encoder
        self.bank = bank
        self.sem_strength, self.rep_strength = float(sem_strength), float(rep_strength)
        self.tau_s, self.tau_t = float(tau_s), float(tau_t)
        self.sem_window = (float(sem_t_min), float(sem_t_max))
        self.rep_window = (float(rep_t_min), float(rep_t_max))
        self.rep_size = int(rep_size)
        self.scale_mode = scale_mode
        self.last_stats: dict = {}

    def reset(self):
        """Stateless; kept for the ``Guidance`` protocol."""
        self.last_stats = {}

    @staticmethod
    def _active(window, t: float, t_star: float) -> bool:
        lo, hi = window
        return lo * t_star - 1e-12 <= t <= hi * t_star + 1e-12

    # ---------------------------------------------------------------- gradients w.r.t. x0_hat
    def sem_grad(self, x0: torch.Tensor, t: float, n_particles: int, create_graph: bool) -> torch.Tensor:
        B = x0.shape[0] // n_particles
        feats = self.encoder((x0 + 1) * 0.5)
        z = F.normalize(feats, dim=-1).view(B, n_particles, -1)
        prototypes = self.bank.at(t).to(device=z.device, dtype=z.dtype)
        g_z = sem_grad_z(z, prototypes, self.tau_s, self.tau_t)
        return torch.autograd.grad(z, x0, grad_outputs=g_z, create_graph=create_graph,
                                   retain_graph=create_graph)[0]

    def rep_grad(self, x0: torch.Tensor, n_particles: int, create_graph: bool) -> torch.Tensor:
        phi = rep_features(x0, n_particles, self.rep_size)
        g_phi = rep_grad_phi(phi)
        return torch.autograd.grad(phi, x0, grad_outputs=g_phi, create_graph=create_graph,
                                   retain_graph=create_graph)[0]

    # ---------------------------------------------------------------- scaling to a noise correction
    def _to_eps(self, g: torch.Tensor, eps: torch.Tensor, strength: float, t: float) -> torch.Tensor:
        if self.scale_mode == "normalized":
            g_norm = g.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, *([1] * (g.ndim - 1)))
            e_norm = eps.flatten(1).norm(dim=1).view(-1, *([1] * (g.ndim - 1)))
            return strength * e_norm * g / g_norm
        ab = float(alpha_bar_cont(t))
        return strength * math.sqrt(1.0 - ab) / math.sqrt(ab) * g

    def forward(self, *, x_t: torch.Tensor, x0_hat: torch.Tensor, eps: torch.Tensor, t: float,
                t_star: float, n_particles: int) -> Optional[torch.Tensor]:
        use_sem = self.sem_strength > 0 and self._active(self.sem_window, t, t_star)
        use_rep = self.rep_strength > 0 and n_particles > 1 and self._active(self.rep_window, t, t_star)
        if not (use_sem or use_rep):
            return None
        if x0_hat.shape[0] % n_particles:
            raise ValueError(f"batch of {x0_hat.shape[0]} rows is not a multiple of K={n_particles}")

        create_graph = torch.is_grad_enabled() and x0_hat.requires_grad
        with torch.enable_grad():
            # plain inference: work on a leaf copy; under an attack: keep the graph
            x0 = x0_hat if create_graph else x0_hat.detach().requires_grad_(True)  # SAFE: inference branch only
            d = torch.zeros_like(eps)
            stats = {}
            if use_sem:
                g = self.sem_grad(x0, t, n_particles, create_graph)
                d = d + self._to_eps(g, eps, self.sem_strength, t)
                stats["sem_grad_norm"] = float(g.detach().flatten(1).norm(dim=1).mean())  # SAFE: logging
            if use_rep:
                g = self.rep_grad(x0, n_particles, create_graph)
                d = d + self._to_eps(g, eps, self.rep_strength, t)
                stats["rep_grad_norm"] = float(g.detach().flatten(1).norm(dim=1).mean())  # SAFE: logging
        self.last_stats = stats
        return d
