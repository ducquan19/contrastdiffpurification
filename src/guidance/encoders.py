"""Frozen feature extractors F for the PCP semantic guidance.

``build_feature_encoder(name, feature)`` resolves ``name`` as

* ``robustbench:<model_name>[:<threat_model>]`` -- a RobustBench model for the run's dataset
  (default threat model ``Linf``), e.g. an adversarially trained ResNet-18; or
* any name accepted by ``get_image_classifier`` (``cifar10-wideresnet-28-10`` ...), i.e. the
  same kind of non-robust classifier ContrastDiffPure uses.

``feature='penultimate'`` returns the input of the classifier's last ``nn.Linear`` (captured with
a forward hook, so the graph to the input is kept); ``'logits'`` returns the logits, the
representation ContrastDiffPure embeds.

The parameters are frozen (``requires_grad_(False)``), but gradients w.r.t. the *input* flow,
which the guidance and a full-gradient attack both need.

Report the accuracy of a robust encoder used on its own next to the PCP results, so that the
contribution of the purifier is not confused with the robustness of the encoder.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from src.paths import pretrained_path

logger = logging.getLogger(__name__)

FEATURES = ("penultimate", "logits")


def _last_linear(model: nn.Module) -> nn.Linear:
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if not linears:
        raise ValueError("the encoder has no nn.Linear head to take penultimate features from")
    return linears[-1]


class FeatureEncoder(nn.Module):
    """Wrap a classifier (images in [0, 1]) as a frozen feature extractor."""

    def __init__(self, model: nn.Module, feature: str = "penultimate"):
        super().__init__()
        if feature not in FEATURES:
            raise ValueError(f"unknown feature '{feature}'; available: {', '.join(FEATURES)}")
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.feature = feature
        self._head = _last_linear(self.model) if feature == "penultimate" else None

    def train(self, mode: bool = True):
        # always evaluation mode (batch norm statistics and dropout are part of F)
        return super().train(False)

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        if self.feature == "logits":
            return self.model(x01)
        captured = {}

        def hook(_module, inputs, _output):
            captured["f"] = inputs[0]

        handle = self._head.register_forward_hook(hook)
        try:
            self.model(x01)
        finally:
            handle.remove()
        f = captured["f"]
        return f.flatten(1)


def build_feature_encoder(name: str, feature: str = "penultimate", dataset: str = "cifar10",
                          device=None) -> FeatureEncoder:
    """The frozen encoder ``name`` (module docstring) on ``device``."""
    if name.startswith("robustbench:"):
        from robustbench import load_model  # heavy import, only when needed

        parts = name.split(":")
        model_name = parts[1]
        threat_model = parts[2] if len(parts) > 2 else "Linf"
        logger.info("PCP encoder: RobustBench %s (%s, %s)", model_name, dataset, threat_model)
        model = load_model(model_dir=pretrained_path("robustbench"), model_name=model_name,
                           dataset=dataset, threat_model=threat_model)
    else:
        from src.models.classifiers import get_image_classifier

        logger.info("PCP encoder: classifier %s", name)
        model = get_image_classifier(name)
    enc = FeatureEncoder(model, feature)
    return enc.to(device) if device is not None else enc
