# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DiffPure. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
"""Models attacked by the evaluations: a classifier with or without diffusion purification.

Both expose ``forward(x, mode)`` with ``mode`` in ``purify`` / ``classify`` /
``purify_and_classify`` (the default). The attacks of ``attacks.py`` call ``model(x)``;
``BPDA_EOT_Attack`` calls the ``purify`` and ``classify`` modes separately.
Images are in [0, 1] throughout.
"""

import logging
import time

import torch
import torch.nn as nn

from src.guidance.build import build_contrastive_guidance
from src.guidance.contrastive import ContrastiveGuidedDiffusion
from src.models.classifiers import get_image_classifier
from src.models.diffusion.ode import ODEPurifier
from src.models.diffusion.sde import SDEPurifier

PURIFIERS = {"sde": SDEPurifier, "ode": ODEPurifier}

logger = logging.getLogger(__name__)


class UndefendedClassifier(nn.Module):
    """The bare classifier, with an identity ``purify`` (the undefended baseline of BPDA)."""

    def __init__(self, args, config):
        super().__init__()
        self.classifier = get_image_classifier(args.classifier_name).to(config.device)

    def purify(self, x):
        return x

    def forward(self, x, mode='purify_and_classify'):
        if mode == 'purify':
            return self.purify(x)
        if mode in ('classify', 'purify_and_classify'):
            return self.classifier(x)
        raise NotImplementedError(f'unknown mode: {mode}')


class PurifiedClassifier(nn.Module):
    """``classifier(purify(x))``, optionally with contrastive guidance.

    ``args.diffusion_type`` picks the reverse SDE or the probability-flow ODE, and
    ``args.guidance == 'contrastive'`` wraps the score network with ``ContrastiveGuidedDiffusion``.
    ``counter`` numbers the purified batches (used for logging and image dumps).

    ``args.grad_mode``: ``straight_through`` (default, the released evaluation: the purifier is
    the identity in the backward pass) or ``full`` (gradients through the adjoint solver). The
    explicit-loop solver with direct back-propagation is ``--diffusion_type particle``
    (``src/evaluation/particle_model.py``).
    """

    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.device = config.device

        # the classifier is built before the purifier: model construction consumes the torch RNG
        self.classifier = get_image_classifier(args.classifier_name).to(config.device)

        logger.info("diffusion type: %s", args.diffusion_type)
        if args.diffusion_type not in PURIFIERS:
            raise NotImplementedError(f"unknown diffusion type: {args.diffusion_type}")
        self.runner = PURIFIERS[args.diffusion_type](args, config, device=config.device)

        if args.guidance == "contrastive" and getattr(args, "grad_mode", "straight_through") == "full":
            # the adjoint solver re-evaluates the drift in the backward pass, which would re-run the
            # stateful wrapper (step counter, cached positive view, online Adam step on F(.))
            raise ValueError("--grad_mode full with --guidance contrastive is not supported by the "
                             "adjoint SDE/ODE solvers; use --diffusion_type particle "
                             "--contrastive_differentiable true")

        if args.guidance == "contrastive":
            self.runner.set_score_model(
                build_contrastive_guidance(args.method_config, self.runner.model, device=config.device))
            logger.info("contrastive guidance attached to the score network")

        self.register_buffer("counter", torch.zeros(1, device=config.device))
        self.tag = None

    def reset_counter(self):
        """Restart batch numbering (call before each attack phase)."""
        self.counter = torch.zeros(1, dtype=torch.int, device=self.device)

    def set_tag(self, tag=None):
        """Name of the sub-folder that debug images of the next batches are saved under."""
        self.tag = tag

    def _purify_centered(self, x):
        """Purify ``x`` in [0, 1]; returns the result in [-1, 1], the diffusion model's range."""
        counter = self.counter.item()
        if counter % 5 == 0:
            logger.info("purifying batch %d", counter)

        start_time = time.time()
        x_re = self.runner.image_editing_sample(
            (x - 0.5) * 2,
            bs_id=counter,
            tag=self.tag
        )
        minutes, seconds = divmod(time.time() - start_time, 60)

        if counter % 5 == 0:
            logger.debug("input shape %s, purified shape %s", tuple(x.shape), tuple(x_re.shape))
            logger.info("purification time per batch: %02d:%05.2f", int(minutes), seconds)

        self.counter += 1

        # each purified batch starts with a fresh F(.) and step counter
        if isinstance(self.runner.model, ContrastiveGuidedDiffusion):
            self.runner.model.reset_drift_counter()

        return x_re

    def purify(self, x):
        """Purify ``x`` in [0, 1] and return the result in [0, 1]."""
        return (self._purify_centered(x) + 1) * 0.5

    def forward(self, x, mode='purify_and_classify'):
        if mode == 'purify':
            return self.purify(x)
        if mode == 'classify':
            return self.classifier(x)
        if mode != 'purify_and_classify':
            raise NotImplementedError(f'unknown mode: {mode}')

        if getattr(self.args, "grad_mode", "straight_through") == "full":
            # gradients flow through the solver (adjoint), as in the DiffPure code
            return self.classifier((self._purify_centered(x) + 1) * 0.5)

        with torch.no_grad():
            w = self._purify_centered(x) - x
        # x + w == the purified image, but w carries no gradient: the purifier is treated as the
        # identity in the backward pass (as in the DiffPure evaluation code).
        return self.classifier((x + w + 1) * 0.5)
