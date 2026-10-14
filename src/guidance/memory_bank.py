"""Fixed bank of clean prototypes for the PCP semantic guidance.

The prototypes are spherical k-means centroids of encoder features of *clean training* images.
During the reverse process the guidance reads features of the Tweedie estimate ``x0_hat``,
which is blurry at large ``t``; a bank can therefore hold one set of prototypes per time
bucket, each computed from ``x0_hat`` of clean images noised to that time
(``build_prototype_bank``). ``PrototypeBank.at(t)`` returns the bucket nearest to ``t``.

The bank is built once (``scripts/build_prototype_bank.py``), saved with its metadata and never
depends on the evaluated images.
"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Callable, Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.diffusion.sampler import alpha_bar_cont

logger = logging.getLogger(__name__)


def spherical_kmeans(x: torch.Tensor, n_clusters: int, iters: int = 50, seed: int = 0) -> torch.Tensor:
    """Spherical k-means on unit rows of ``x`` (N, D); returns (n_clusters, D) unit centroids.

    k-means++ initialisation on cosine distance; empty clusters are re-seeded with the points
    farthest from their centroid.
    """
    if x.ndim != 2:
        raise ValueError(f"expected (N, D) features, got {tuple(x.shape)}")
    N = x.shape[0]
    if n_clusters > N:
        raise ValueError(f"cannot build {n_clusters} clusters from {N} points")
    x = F.normalize(x.float(), dim=-1)
    g = torch.Generator(device="cpu").manual_seed(seed)

    # k-means++ seeding
    idx = [int(torch.randint(N, (1,), generator=g))]
    dist = 1 - x @ x[idx[0]]
    for _ in range(1, n_clusters):
        prob = dist.clamp_min(0).cpu()
        prob = prob / prob.sum() if prob.sum() > 0 else torch.full_like(prob, 1.0 / N)
        nxt = int(torch.multinomial(prob, 1, generator=g))
        idx.append(nxt)
        dist = torch.minimum(dist, 1 - x @ x[nxt])
    c = x[idx].clone()

    for _ in range(iters):
        sim = x @ c.t()
        assign = sim.argmax(dim=1)
        new_c = torch.zeros_like(c).index_add_(0, assign, x)
        counts = torch.bincount(assign, minlength=n_clusters)
        empty = (counts == 0).nonzero().flatten()
        if len(empty):
            far = (sim.max(dim=1).values).argsort()[: len(empty)]
            new_c[empty] = x[far]
        new_c = F.normalize(new_c, dim=-1)
        if torch.allclose(new_c, c, atol=1e-6):
            c = new_c
            break
        c = new_c
    return c


class PrototypeBank(nn.Module):
    """Time-bucketed unit prototypes: ``prototypes`` (T, M, D) at times ``t_values`` (T,)."""

    def __init__(self, prototypes: torch.Tensor, t_values: Sequence[float], meta: Optional[dict] = None):
        super().__init__()
        prototypes = torch.as_tensor(prototypes, dtype=torch.float32)
        if prototypes.ndim == 2:
            prototypes = prototypes.unsqueeze(0)
        t_values = torch.as_tensor(list(t_values), dtype=torch.float64)
        if prototypes.shape[0] != len(t_values):
            raise ValueError(f"{prototypes.shape[0]} prototype sets for {len(t_values)} times")
        self.register_buffer("prototypes", F.normalize(prototypes, dim=-1))
        self.register_buffer("t_values", t_values)
        self.meta = dict(meta or {})

    @property
    def n_prototypes(self) -> int:
        return self.prototypes.shape[1]

    @property
    def dim(self) -> int:
        return self.prototypes.shape[2]

    def at(self, t: float) -> torch.Tensor:
        """Prototypes (M, D) of the bucket whose time is nearest to ``t``."""
        i = int((self.t_values - float(t)).abs().argmin())
        return self.prototypes[i]

    def fingerprint(self) -> str:
        """Short hash of the prototypes, recorded in run logs."""
        return hashlib.sha1(self.prototypes.detach().cpu().numpy().tobytes()).hexdigest()[:12]  # SAFE: hashing

    def save(self, path: str) -> None:
        torch.save({"prototypes": self.prototypes.cpu(), "t_values": self.t_values.cpu(),
                    "meta": self.meta}, path)

    @classmethod
    def load(cls, path: str, map_location="cpu") -> "PrototypeBank":
        blob = torch.load(path, map_location=map_location)
        bank = cls(blob["prototypes"], blob["t_values"].tolist(), blob.get("meta", {}))
        # the saved prototypes are already unit vectors: keep them bit-exact instead of
        # normalising a second time, so that save/load round trips and fingerprints match
        bank.prototypes.copy_(blob["prototypes"])
        return bank


@torch.no_grad()  # SAFE: offline construction of a fixed bank, not on the data path of the defense
def build_prototype_bank(encoder: nn.Module, batches: Iterable[torch.Tensor], *, n_prototypes: int,
                         t_values: Sequence[float], eps_fn: Optional[Callable] = None,
                         device="cpu", seed: int = 0, kmeans_iters: int = 50,
                         meta: Optional[dict] = None) -> PrototypeBank:
    """Build a bank from clean images in [0, 1] (an iterable of (N, C, H, W) batches).

    For ``t > 0`` every batch is noised to ``t`` with the continuous schedule and mapped back with
    the Tweedie estimate ``x0_hat = (x_t - sqrt(1-a) eps_fn(x_t, t)) / sqrt(a)``; this needs
    ``eps_fn``. ``t = 0`` uses the clean images.
    """
    if any(t > 0 for t in t_values) and eps_fn is None:
        raise ValueError("t_values > 0 need eps_fn to compute x0_hat")
    g = torch.Generator(device="cpu").manual_seed(seed)
    feats = {float(t): [] for t in t_values}
    n_seen = 0
    for xb in batches:
        xb = xb.to(device)
        n_seen += xb.shape[0]
        x = xb * 2 - 1
        for t in t_values:
            t = float(t)
            if t > 0:
                a = float(alpha_bar_cont(t))
                noise = torch.randn(x.shape, generator=g).to(device)
                x_t = math.sqrt(a) * x + math.sqrt(1 - a) * noise
                eps = eps_fn(x_t, torch.full((x.shape[0],), t, device=device, dtype=x.dtype))
                x0 = (x_t - math.sqrt(1 - a) * eps) / math.sqrt(a)
            else:
                x0 = x
            feats[t].append(F.normalize(encoder((x0 + 1) * 0.5).float(), dim=-1).cpu())
    protos = []
    for t in t_values:
        f = torch.cat(feats[float(t)])
        logger.info("bank bucket t=%.4f: k-means of %d features into %d prototypes", t, len(f), n_prototypes)
        protos.append(spherical_kmeans(f, n_prototypes, iters=kmeans_iters, seed=seed))
    meta = {**(meta or {}), "n_images": n_seen, "n_prototypes": n_prototypes,
            "t_values": [float(t) for t in t_values], "seed": seed}
    return PrototypeBank(torch.stack(protos), t_values, meta)
