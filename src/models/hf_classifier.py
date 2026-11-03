"""HuggingFace image classifiers on [0, 1] inputs (no robustbench import; CPU-testable)."""

import logging

import torch
import torch.nn as nn

from src.paths import pretrained_path

logger = logging.getLogger(__name__)


CIFAR10_CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def label_permutation(id2label, expected):
    """Model-output index for every expected class name (``None`` if names are generic/absent)."""
    names = [str(id2label[k]).lower() for k in sorted(id2label, key=int)] if id2label else []
    if sorted(names) != sorted(expected):
        return None
    return [names.index(c) for c in expected]


class HFImageClassifier(nn.Module):
    """A HuggingFace image classifier on [0, 1] inputs, preprocessed as in Carlini et al. (2023).

    ``x01 -> 2 x01 - 1`` (the ViT's mean = std = 0.5 normalization) and a bicubic, anti-aliased
    resize to ``size`` x ``size``; the output logits are reordered to ``expected`` class order.
    """

    def __init__(self, model, size=224, perm=None):
        super().__init__()
        self.model = model
        self.size = int(size)
        self.register_buffer('perm', torch.tensor(perm, dtype=torch.long) if perm is not None else None,
                             persistent=False)

    def preprocess(self, x01):
        x = 2.0 * x01 - 1.0
        if x.shape[-1] != self.size or x.shape[-2] != self.size:
            x = torch.nn.functional.interpolate(x, (self.size, self.size), mode='bicubic',
                                                antialias=True, align_corners=False)
        return x

    def forward(self, x01):
        logits = self.model(pixel_values=self.preprocess(x01)).logits
        return logits[:, self.perm] if self.perm is not None else logits


def load_hf_classifier(repo, expected=CIFAR10_CLASSES):
    """``hf:<repo>``; weights cached under pretrained/hf (needs ``pip install transformers``)."""
    try:
        from transformers import AutoModelForImageClassification
    except ImportError as e:
        raise ImportError("hf:<repo> classifiers need 'pip install transformers'") from e
    model = AutoModelForImageClassification.from_pretrained(repo, cache_dir=pretrained_path('hf')).eval()
    id2label = getattr(model.config, 'id2label', None)
    perm = label_permutation(id2label, list(expected))
    if perm is None:
        raise ValueError(f"{repo}: labels {id2label} do not match {expected}")
    size = getattr(model.config, 'image_size', 224)
    logger.info("HF classifier %s: input %dx%d, label permutation %s", repo, size, size, perm)
    return HFImageClassifier(model, size=size, perm=None if perm == list(range(len(perm))) else perm).eval()
