"""Randomized smoothing (Cohen, Rosenfeld & Kolter, ICML 2019): CERTIFY and the l2 radius.

The smoothed classifier is ``g(x) = argmax_c P[h(x + sigma * xi) = c]`` with ``xi ~ N(0, I)`` in
the [0, 1] pixel scale. ``h`` may be randomized (the reverse diffusion draws its own noise) as long
as its internal randomness does not depend on ``x``: the certificate then holds for the smoothed
classifier whose probabilities are taken over ``xi`` *and* that internal randomness.

``certify`` follows Algorithm CERTIFY of Cohen et al.: ``n0`` samples select the class ``cA``,
``n`` fresh samples give the one-sided ``(1 - alpha)`` Clopper-Pearson lower bound ``pA`` of
``P[h = cA]``, and the image is certified at radius ``sigma * Phi^-1(pA)`` if ``pA > 1/2``,
otherwise the smoothed classifier abstains (``ABSTAIN``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from scipy.stats import beta as beta_dist
from scipy.stats import norm

ABSTAIN = -1
DEFAULT_RADII = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


def clopper_pearson_lower(k: int, n: int, alpha: float) -> float:
    """One-sided ``(1 - alpha)`` lower confidence bound of a binomial proportion ``k / n``.

    Equal to the lower end of the two-sided ``(1 - 2 alpha)`` Clopper-Pearson interval, which is
    what Cohen et al.'s reference code (``proportion_confint(..., alpha=2*alpha, method='beta')``)
    uses.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    if k == 0:
        return 0.0
    return float(beta_dist.ppf(alpha, k, n - k + 1))


def certified_radius(p_lower: float, sigma: float) -> float:
    """l2 radius ``sigma * Phi^-1(p_lower)``; 0 when ``p_lower <= 1/2`` (abstain)."""
    if p_lower <= 0.5:
        return 0.0
    return float(sigma * norm.ppf(p_lower))


@dataclass
class CertifyResult:
    predict: int          # cA, or ABSTAIN
    p_lower: float
    radius: float
    counts_select: np.ndarray
    counts_estimate: np.ndarray


def sample_counts(base: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, n: int,
                  sigma: float, n_classes: int, batch: int, generator: torch.Generator) -> np.ndarray:
    """Class counts of ``base`` on ``n`` noisy copies ``x + sigma * xi`` of one image ``x`` (C,H,W).

    ``base`` maps a (b, C, H, W) batch of noisy images (pixel scale, *not* clipped) to (b,) labels.
    The Gaussian noise comes from ``generator`` only, so a run is reproducible per image.
    """
    counts = np.zeros(n_classes, dtype=np.int64)
    remaining = n
    while remaining > 0:
        b = min(batch, remaining)
        xi = torch.randn((b, *x.shape), generator=generator, device=x.device, dtype=x.dtype)
        labels = base(x.unsqueeze(0) + sigma * xi)
        counts += np.bincount(labels.cpu().numpy(), minlength=n_classes)[:n_classes]
        remaining -= b
    return counts


def certify(base: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, *, sigma: float,
            n0: int, n: int, alpha: float, n_classes: int, batch: int,
            generator: torch.Generator) -> CertifyResult:
    """Cohen et al.'s CERTIFY for one image ``x`` (C, H, W) in [0, 1]."""
    c0 = sample_counts(base, x, n0, sigma, n_classes, batch, generator)
    c_a = int(c0.argmax())
    c1 = sample_counts(base, x, n, sigma, n_classes, batch, generator)
    p_lower = clopper_pearson_lower(int(c1[c_a]), n, alpha)
    if p_lower <= 0.5:
        return CertifyResult(ABSTAIN, p_lower, 0.0, c0, c1)
    return CertifyResult(c_a, p_lower, certified_radius(p_lower, sigma), c0, c1)


def certified_accuracy(labels: Sequence[int], predicts: Sequence[int], radii: Sequence[float],
                       at: Sequence[float] = DEFAULT_RADII) -> dict:
    """Fraction of images that are correct *and* certified at radius >= r, for every r in ``at``.

    At r = 0 this is the accuracy of the smoothed classifier with abstentions counted as errors.
    """
    labels, predicts, radii = map(np.asarray, (labels, predicts, radii))
    ok = predicts == labels
    return {float(r): (float(np.mean(ok & (radii >= r))) if len(labels) else math.nan) for r in at}


def average_certified_radius(labels: Sequence[int], predicts: Sequence[int],
                             radii: Sequence[float]) -> float:
    """ACR: mean over images of the radius if correct, else 0."""
    labels, predicts, radii = map(np.asarray, (labels, predicts, radii))
    if not len(labels):
        return math.nan
    return float(np.mean(np.where(predicts == labels, radii, 0.0)))


def plugin_radius(p_true: float, sigma: float, n: int) -> float:
    """Radius from the *empirical* frequency of the true class instead of a confidence bound.

    Only a diagnostic (gate G0): it is optimistic, ignores ``n0`` / ``alpha``, and clips at
    ``1 - 1/(2n)`` so that ``p_true = 1`` stays finite.
    """
    if p_true <= 0.5:
        return 0.0
    return float(sigma * norm.ppf(min(p_true, 1.0 - 0.5 / n)))


def image_generator(seed: int, index: int, device) -> torch.Generator:
    """Generator of the noise of one image: depends on ``(seed, index)`` only."""
    g = torch.Generator(device=torch.device(device))
    g.manual_seed((int(seed) * 1_000_003 + int(index)) & 0x7FFFFFFFFFFFFFFF)
    return g


def summarize(labels, predicts, radii, at: Optional[Sequence[float]] = None) -> dict:
    """Certified accuracy at the radii ``at``, ACR and abstention rate of a CERTIFY run."""
    at = list(at) if at is not None else list(DEFAULT_RADII)
    return {
        "n_images": int(len(labels)),
        "certified_accuracy": certified_accuracy(labels, predicts, radii, at),
        "acr": average_certified_radius(labels, predicts, radii),
        "abstain_rate": float(np.mean(np.asarray(predicts) == ABSTAIN)) if len(labels) else math.nan,
    }
