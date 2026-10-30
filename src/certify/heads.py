"""Classifier heads of the base classifier: ensembles of off-the-shelf classifiers and flip TTA.

A head maps denoised images in [0, 1] to log-probabilities (argmax = prediction). It averages
the softmax of every member classifier on every view (the image and, with ``flip=True``, its
horizontal mirror). Being a deterministic function of the denoised image, any head keeps the
randomized-smoothing certificate valid. Nothing is trained.
"""

from __future__ import annotations

from typing import Callable, Dict, Sequence

import torch


class ClassifierHead:
    """Mean softmax over ``classifiers`` x views; returns log of the mean probability."""

    def __init__(self, classifiers: Sequence[Callable[[torch.Tensor], torch.Tensor]], flip: bool = False):
        if not classifiers:
            raise ValueError("need at least one classifier")
        self.classifiers = list(classifiers)
        self.flip = bool(flip)

    def __call__(self, x01: torch.Tensor) -> torch.Tensor:
        views = [x01, torch.flip(x01, dims=[-1])] if self.flip else [x01]
        prob = None
        for f in self.classifiers:
            for v in views:
                p = torch.softmax(f(v).float(), dim=1)
                prob = p if prob is None else prob + p
        prob = prob / (len(self.classifiers) * len(views))
        return torch.log(prob.clamp_min(1e-12))


def build_heads(main: Callable, extra: Sequence[Callable] = (), flip: bool = False) -> Dict[str, Callable]:
    """Heads keyed by their variant suffix: ``''`` (main classifier), ``'+flip'``, ``'+ens'``, ``'+ens+flip'``.

    ``'+ens'`` heads exist only when ``extra`` classifiers are given, ``'+flip'`` heads only with
    ``flip=True``.
    """
    heads = {"": main}
    if flip:
        heads["+flip"] = ClassifierHead([main], flip=True)
    if extra:
        heads["+ens"] = ClassifierHead([main, *extra], flip=False)
        if flip:
            heads["+ens+flip"] = ClassifierHead([main, *extra], flip=True)
    return heads


def certify_head_key(has_ens: bool, flip: bool) -> str:
    """Suffix of the head used by CERTIFY for the given options."""
    return ("+ens" if has_ens else "") + ("+flip" if flip else "")
