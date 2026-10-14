"""Turning the K particle predictions of an image into one prediction.

* ``soft_vote_logits``: ``log(mean_k softmax(logits_k))`` -- differentiable; this is what an
  attack optimises (the majority-vote setting of DiffBreak attacks the mean probability).
* ``hard_vote_scores``: vote counts plus a 1e-3 * mean-probability tie-break; ``argmax`` is the
  majority class. Not differentiable (counts), so used only to *score* a prediction.

Both return (B, C) tensors whose ``argmax`` is the prediction, so ``get_accuracy`` works on them.
"""

import torch
import torch.nn.functional as F


def soft_vote_logits(logits: torch.Tensor) -> torch.Tensor:
    """(B, K, C) particle logits -> (B, C) log mean probability."""
    if logits.ndim != 3:
        raise ValueError(f"expected (B, K, C) logits, got {tuple(logits.shape)}")
    return torch.logsumexp(F.log_softmax(logits, dim=-1), dim=1) - torch.log(
        torch.tensor(float(logits.shape[1]), device=logits.device, dtype=logits.dtype))


def hard_vote_scores(logits: torch.Tensor) -> torch.Tensor:
    """(B, K, C) particle logits -> (B, C) majority-vote scores (counts + small soft tie-break)."""
    if logits.ndim != 3:
        raise ValueError(f"expected (B, K, C) logits, got {tuple(logits.shape)}")
    C = logits.shape[-1]
    counts = F.one_hot(logits.argmax(dim=-1), C).sum(dim=1).to(logits.dtype)
    return counts + 1e-3 * F.softmax(logits, dim=-1).mean(dim=1)


def vote(logits: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "soft":
        return soft_vote_logits(logits)
    if mode == "hard":
        return hard_vote_scores(logits)
    raise ValueError(f"unknown vote '{mode}'; available: hard, soft")


def particle_agreement(pred: torch.Tensor, y: torch.Tensor) -> dict:
    """Diagnostics of (B, K) particle predictions against labels (B,).

    ``error_corr`` is the mean pairwise Pearson correlation, over images, of the error indicators
    of two particles (rho in the plan); ``k_eff = K / (1 + (K - 1) rho)``.
    """
    pred, y = pred.cpu(), y.cpu()
    B, K = pred.shape
    err = (pred != y[:, None]).float()                      # (B, K)
    out = {"particle_acc": float(1 - err.mean()), "K": K}
    if K > 1 and B > 1:
        e = err - err.mean(dim=0, keepdim=True)
        cov = (e.t() @ e) / (B - 1)
        std = cov.diag().clamp_min(1e-12).sqrt()
        corr = cov / (std[:, None] * std[None, :])
        off = ~torch.eye(K, dtype=torch.bool)
        valid = (cov.diag() > 1e-12)
        mask = off & valid[:, None] & valid[None, :]
        rho = float(corr[mask].mean()) if mask.any() else float("nan")
        out["error_corr"] = rho
        out["k_eff"] = K / (1 + (K - 1) * rho) if rho == rho else float("nan")
        out["unanimous_frac"] = float((pred == pred[:, :1]).all(dim=1).float().mean())
    return out
