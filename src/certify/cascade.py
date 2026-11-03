"""Confidence-routed cascade of diffusion denoised smoothing (docs/cascade.md). Training-free.

Two ingredients, both deterministic functions of ONE noisy input ``x~`` (so Cohen et al.'s
certificate of the smoothed classifier stays valid):

* **early exit** -- denoise ``x~`` with the one-shot posterior mean ``m`` (1 NFE) and classify it;
  only if the head's confidence ``max_c p(c | m)`` is below ``tau`` continue the SAME DDIM
  trajectory (it reuses the score evaluation that produced ``m``) for ``S - 1`` more NFE and
  classify the multi-step output instead. ``tau <= 0`` gives the one-shot base classifier,
  ``tau > 1`` the multi-step one, exactly.
* **routing** -- the head is a set of off-the-shelf classifiers (standard, l2-adversarially
  trained, ViT, ...); ``heads.ConfidenceRouter`` answers with the most confident member.

``tau`` and the head are hyperparameters: choose them on the dev split, never on the test images.

The torch part (``make_cascade_base``) is the base classifier used by CERTIFY; the numpy part
evaluates every head / cascade offline from saved class probabilities (scripts/screen_cascade.py
and scripts/analyze_cascade.py), with common random numbers across all policies.
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.certify.smoothing import plugin_radius


# --------------------------------------------------------------------------------------------
# torch: base classifier of CERTIFY
# --------------------------------------------------------------------------------------------

def make_cascade_base(denoiser, head_first: Callable, head_second: Callable, tau: float,
                      generator_ref: list, stats: Optional[dict] = None) -> Callable:
    """``h(x~)``: one-shot + ``head_first``; rows with confidence < ``tau`` continue to multi-step.

    ``denoiser`` must be a ``DDSDenoiser`` with ``method='multistep'`` (its ``steps`` / ``eta``
    define the second stage). The DDIM noise is drawn for the whole batch, as the plain
    multi-step denoiser does, so ``tau > 1`` reproduces ``argmax head_second(denoiser(x~))``
    bit for bit. ``stats`` (optional) accumulates ``n`` and ``n_continued`` for the mean NFE
    ``1 + (S - 1) * n_continued / n``.
    """
    if denoiser.method != "multistep":
        raise ValueError(f"the cascade needs a 'multistep' denoiser, got '{denoiser.method}'")
    if denoiser.guide is not None:
        raise ValueError("the cascade does not support guidance")

    @torch.no_grad()
    def base(x_noisy: torch.Tensor) -> torch.Tensor:
        x_t, eps0, m = denoiser.start(x_noisy)
        probs = torch.softmax(head_first((m.clamp(-1, 1) + 1) * 0.5).float(), dim=1)
        conf, pred = probs.max(dim=1)
        zs = denoiser.draw_step_noise(x_noisy.shape, generator_ref[0], x_noisy.device, x_noisy.dtype)
        cont = conf < tau
        n_cont = int(cont.sum())
        if n_cont:
            idx = cont.nonzero(as_tuple=False).squeeze(1)
            s = denoiser.trajectory(x_t[idx], eps0[idx], m[idx], zs[:, idx], anchored=False)
            pred[idx] = head_second((s.clamp(-1, 1) + 1) * 0.5).argmax(1)
        if stats is not None:
            stats["n"] = stats.get("n", 0) + int(x_noisy.shape[0])
            stats["n_continued"] = stats.get("n_continued", 0) + n_cont
        return pred
    return base


def cascade_nfe(steps: int, frac_continued: float) -> float:
    """Mean score-network evaluations per noisy sample of the cascade."""
    return 1.0 + (steps - 1) * float(frac_continued)


# --------------------------------------------------------------------------------------------
# numpy: offline policies on saved probabilities
# --------------------------------------------------------------------------------------------
# P has shape (I, D, V, C, K): images x noisy draws x denoiser variants x classifiers x classes.

COMBINERS = ("single", "ens", "route")


def head_specs(n_classifiers: int) -> List[Tuple[str, Tuple[int, ...]]]:
    """Every single classifier, and the ensemble / router of every subset of size >= 2."""
    specs = [("single", (c,)) for c in range(n_classifiers)]
    for k in range(2, n_classifiers + 1):
        for sub in combinations(range(n_classifiers), k):
            specs += [("ens", sub), ("route", sub)]
    return specs


def head_name(spec: Tuple[str, Tuple[int, ...]], names: Sequence[str]) -> str:
    kind, members = spec
    if kind == "single":
        return names[members[0]]
    return f"{kind}(" + "+".join(names[c] for c in members) + ")"


def combine(P: np.ndarray, spec: Tuple[str, Tuple[int, ...]]) -> np.ndarray:
    """Class probabilities of a head: (..., C, K) -> (..., K).

    ``ens`` = mean probability of the members; ``route`` = the probabilities of the member with
    the largest max-probability (ties: the first member), as ``heads.ConfidenceRouter``.
    """
    kind, members = spec
    if kind not in COMBINERS:
        raise ValueError(f"unknown head kind '{kind}'")
    sub = P[..., list(members), :].astype(np.float64)
    if kind == "single":
        return sub[..., 0, :]
    if kind == "ens":
        return sub.mean(axis=-2)
    best = sub.max(axis=-1).argmax(axis=-1)                      # (...,)
    return np.take_along_axis(sub, best[..., None, None], axis=-2)[..., 0, :]


def cascade_predictions(p_first: np.ndarray, p_second: np.ndarray, tau: float):
    """Predictions (I, D) and continue-mask of the cascade from the two stages' probabilities."""
    conf = p_first.max(axis=-1)
    cont = conf < tau
    pred = np.where(cont, p_second.argmax(axis=-1), p_first.argmax(axis=-1))
    return pred, cont


def draw_metrics(pred: np.ndarray, labels: np.ndarray, sigma: float, n_classes: int) -> Dict[str, np.ndarray]:
    """Per-image p_true, smoothed correctness (majority class == label) and plug-in radius."""
    I, D = pred.shape
    p_true = (pred == labels[:, None]).mean(axis=1)
    counts = np.stack([np.bincount(pred[i], minlength=n_classes)[:n_classes] for i in range(I)])
    top_ok = counts.argmax(axis=1) == labels
    radius = np.array([plugin_radius(p, sigma, D) if ok else 0.0 for p, ok in zip(p_true, top_ok)])
    return {"p_true": p_true, "top_ok": top_ok, "radius": radius}


def summarize_metrics(m: Dict[str, np.ndarray], idx: Optional[np.ndarray] = None) -> Dict[str, float]:
    sel = slice(None) if idx is None else idx
    return {"acc_per_draw": float(m["p_true"][sel].mean()),
            "smoothed_acc": float(m["top_ok"][sel].mean()),
            "plugin_acr": float(m["radius"][sel].mean())}


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """Mean of ``a - b`` (per image) and its 95% percentile bootstrap interval."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    rng = np.random.RandomState(seed)
    bs = d[rng.randint(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return float(d.mean()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def tau_grid(conf: np.ndarray, n: int = 41) -> np.ndarray:
    """Thresholds: 0 (one-shot), quantiles of the observed confidences, and 1 + eps (multi-step)."""
    qs = np.quantile(conf.ravel(), np.linspace(0.0, 1.0, n))
    return np.unique(np.concatenate([[0.0], qs, [1.0 + 1e-9]]))


def choose_tau(p_first: np.ndarray, p_second: np.ndarray, labels: np.ndarray, fit_idx: np.ndarray,
               retain: float, taus: np.ndarray) -> float:
    """Smallest ``tau`` (fewest NFE) whose acc/draw on ``fit_idx`` is >= ``retain`` x multi-step's."""
    y = labels[fit_idx][:, None]
    target = retain * float((p_second[fit_idx].argmax(-1) == y).mean())
    for tau in np.sort(taus):
        pred, _ = cascade_predictions(p_first[fit_idx], p_second[fit_idx], tau)
        if float((pred == y).mean()) >= target - 1e-12:
            return float(tau)
    return float(np.max(taus))
