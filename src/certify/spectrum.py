"""Dataset power spectrum P(f) and the Wiener gain of WA-DDS.

Conventions (shared with ``src/guidance/spectral.py``): images in [-1, 1], ``torch.fft.fft2`` with
``norm='ortho'``, frequencies in ``fft2`` order. With the orthonormal DFT, white Gaussian noise of
per-pixel standard deviation ``sigma_d`` has power ``sigma_d^2`` at *every* frequency, so

    SNR(f) = P(f) / sigma_d^2,       W(f) = P(f) / (P(f) + sigma_d^2)  in [0, 1].

``W`` is the per-frequency gain of the Wiener filter for a Gaussian prior with power spectrum ``P``:
close to 1 where the noisy input determines the signal, close to 0 where noise dominates.
``sigma_d = 2 * sigma`` because smoothing noise ``sigma`` is defined on the [0, 1] pixel scale.

``P`` must be estimated on *training* images (``scripts/spectral_stats.py``), never on the images
that are certified.
"""

from __future__ import annotations

from typing import Iterable

import torch

POWER_MODES = ("channel", "pooled")


@torch.no_grad()
def power_spectrum(batches: Iterable[torch.Tensor], device="cpu") -> torch.Tensor:
    """Mean ``|F(x)|^2`` per channel and frequency, shape (C, H, W), of images given in [0, 1].

    The mean image is *not* subtracted: the DC term carries the average colour, which the noisy
    input determines well and which WA-DDS should therefore anchor.
    """
    total, count = None, 0
    for xb in batches:
        xb = xb.to(device=device, dtype=torch.float64) * 2 - 1
        p = torch.fft.fft2(xb, norm="ortho").abs().pow(2).sum(0)
        total = p if total is None else total + p
        count += xb.shape[0]
    if count == 0:
        raise ValueError("no images given")
    return (total / count).float()


def wiener_gain(power: torch.Tensor, sigma: float, mode: str = "channel") -> torch.Tensor:
    """``W = P / (P + (2 sigma)^2)`` for smoothing noise ``sigma`` on the [0, 1] scale.

    ``mode='channel'`` keeps one gain per colour channel (C, H, W); ``'pooled'`` averages ``P``
    over channels first and broadcasts the same (1, H, W) gain to all channels.
    """
    if mode not in POWER_MODES:
        raise ValueError(f"unknown power mode '{mode}'; available: {', '.join(POWER_MODES)}")
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    p = power if mode == "channel" else power.mean(0, keepdim=True)
    nv = (2.0 * sigma) ** 2
    return p / (p + nv)


def snr_table(power: torch.Tensor, sigmas) -> list[dict]:
    """Share of frequency bins with SNR > 1 and the share of (non-DC) energy they hold."""
    p = power.mean(0)
    ac = p.clone()
    ac[0, 0] = 0
    rows = []
    for s in sigmas:
        hi = p / (2.0 * s) ** 2 > 1
        rows.append({"sigma": float(s),
                     "pct_bins_snr_gt1": float(hi.float().mean() * 100),
                     "pct_ac_energy_snr_gt1": float((ac * hi).sum() / ac.sum() * 100)})
    return rows


def save_power(path: str, power: torch.Tensor, meta: dict) -> None:
    torch.save({"power": power.cpu(), "meta": dict(meta)}, path)


def load_power(path: str) -> torch.Tensor:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    return blob["power"]
