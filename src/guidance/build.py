"""Builds a ``ContrastiveGuidedDiffusion`` wrapper from a resolved ``ContrastiveConfig``.

Used by both evaluation paths (AutoAttack-style attacks and BPDA+EOT).
"""

from src.guidance.contrastive import ContrastiveGuidedDiffusion
from src.method import ContrastiveConfig
from src.models.classifiers import get_image_classifier


def build_contrastive_guidance(config: ContrastiveConfig, model, device=None, *,
                               clear_cache_on_reset: bool = False, differentiable: bool = False):
    """Return ``model`` wrapped so that its score carries the contrastive guidance.

    Loads ``config.classifier`` as the frozen backbone of F(.) and moves the wrapper to ``device``.
    ``clear_cache_on_reset`` / ``differentiable`` are evaluation fixes, not part of the published
    method (``--contrastive_fix_cache`` / ``--contrastive_differentiable``; both default off).
    """
    return ContrastiveGuidedDiffusion(
        model,
        get_image_classifier(config.classifier).eval(),
        n_classes=config.n_classes,
        embed_lr=config.embed_lr,
        drift_min=config.drift_min,
        drift_max=config.drift_max,
        tau_minus=config.tau_minus,
        beta=config.beta,
        temperature=config.temperature,
        guidance_strength=config.guidance_strength,
        estimator=config.estimator,
        clear_cache_on_reset=clear_cache_on_reset,
        differentiable=differentiable,
    ).to(device)
