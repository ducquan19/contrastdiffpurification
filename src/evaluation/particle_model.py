"""``--diffusion_type particle``: K-particle purification + vote, fully differentiable.

``ParticlePurifiedClassifier`` has the interface of ``PurifiedClassifier`` (``forward(x, mode)``,
``reset_counter``, ``set_tag``) so the existing drivers and attacks run unchanged, but

* there is **no** ``torch.no_grad`` / straight-through: an attack differentiates the whole
  purifier (the explicit reverse loop of ``ParticleSampler``), including the guidance;
* ``forward`` returns a vote over the K particles: ``output_vote = 'soft'`` (the default, what
  attacks differentiate) or ``'hard'`` (majority; set by ``eval_pgd_full`` when scoring);
* ``--guidance`` selects the correction added at every reverse step:

  ``none``         DiffPure with the particle solver (K = 1: single purification; K > 1: the
                   majority-vote baseline at equal cost),
  ``contrastive``  ContrastDiffPure's guidance (``ContrastiveGuidedDiffusion.guidance_delta``);
                   pass ``--contrastive_differentiable true`` for a full-gradient evaluation and
                   ``--contrastive_fix_cache true`` for the fixed baseline,
  ``pcp``          Particle-Contrastive guidance (``src/guidance/particle.py``).

The sampler and guidance hyperparameters come from ``args.pcp_config`` (``src/method_pcp.py``).
"""

from __future__ import annotations

import logging
import os

import torch
import torch.nn as nn

from src.evaluation.voting import vote
from src.models.diffusion.sampler import ParticleSampler, alpha_bar_cont

logger = logging.getLogger(__name__)


class ContrastiveCorrection:
    """Adapter: ContrastDiffPure's stateful wrapper as a ``Guidance`` of the particle sampler."""

    def __init__(self, wrapper):
        self.wrapper = wrapper

    def reset(self):
        self.wrapper.reset_drift_counter()

    def __call__(self, *, x_t, x0_hat, eps, t, t_star, n_particles):
        return self.wrapper.guidance_delta(x_t)


class ParticlePurifiedClassifier(nn.Module):
    """``vote_k classifier(purify_k(x))`` with the differentiable particle sampler."""

    def __init__(self, args, config):
        super().__init__()
        from src.models.classifiers import get_image_classifier  # needs robustbench
        from src.models.diffusion.base import VPDynamics, load_score_model  # needs score_sde

        self.args = args
        self.device = config.device
        self.pcp = args.pcp_config
        if self.pcp is None:
            raise ValueError("--diffusion_type particle needs the PCP config (args.pcp_config)")

        # same construction order as PurifiedClassifier: classifier first (consumes the torch RNG)
        self.classifier = get_image_classifier(args.classifier_name).to(config.device)

        img_shape = (config.data.num_channels, config.data.image_size, config.data.image_size)
        score_model = load_score_model(config, config.device)
        for p in score_model.parameters():
            p.requires_grad_(False)
        self.dynamics = VPDynamics(score_model, img_shape=img_shape).to(config.device)

        guidance = self._build_guidance(args, score_model)
        self.sampler = ParticleSampler(
            self.eps_fn, t_star=args.t, steps=self.pcp.steps, solver=self.pcp.solver,
            eta=self.pcp.eta, n_particles=self.pcp.n_particles, guidance=guidance,
            checkpoint=self.pcp.checkpoint)
        self.guidance = guidance
        logger.info("particle sampler: t*=%d, K=%d, steps=%d, solver=%s, eta=%g, guidance=%s",
                    args.t, self.pcp.n_particles, self.pcp.steps, self.pcp.solver, self.pcp.eta,
                    args.guidance)

        self.output_vote = "soft"
        self.last_particle_logits = None
        self.register_buffer("counter", torch.zeros(1, device=config.device))
        self.tag = None

    # ------------------------------------------------------------------ construction
    def _build_guidance(self, args, score_model):
        if args.guidance == "none":
            return None
        if args.guidance == "contrastive":
            from src.guidance.build import build_contrastive_guidance

            wrapper = build_contrastive_guidance(
                args.method_config, score_model, device=self.device,
                clear_cache_on_reset=getattr(args, "contrastive_fix_cache", False),
                differentiable=getattr(args, "contrastive_differentiable", False))
            return ContrastiveCorrection(wrapper)
        if args.guidance == "pcp":
            from src.guidance.encoders import build_feature_encoder
            from src.guidance.memory_bank import PrototypeBank
            from src.guidance.particle import ParticleContrastiveGuidance

            c = self.pcp
            encoder, bank = None, None
            if c.sem_strength > 0:
                encoder = build_feature_encoder(c.encoder, c.encoder_feature, dataset=args.domain,
                                                device=self.device)
                path = c.bank_path()
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"prototype bank {path} not found; build it with scripts/build_prototype_bank.py")
                bank = PrototypeBank.load(path).to(self.device)
                logger.info("PCP bank %s: %d buckets x %d prototypes x %d dims, fingerprint %s",
                            path, len(bank.t_values), bank.n_prototypes, bank.dim, bank.fingerprint())
            return ParticleContrastiveGuidance(
                encoder, bank, sem_strength=c.sem_strength, rep_strength=c.rep_strength,
                tau_s=c.tau_s, tau_t=c.tau_t, sem_t_min=c.sem_t_min, sem_t_max=c.sem_t_max,
                rep_t_min=c.rep_t_min, rep_t_max=c.rep_t_max, rep_size=c.rep_size,
                scale_mode=c.scale_mode)
        raise NotImplementedError(f"unknown guidance '{args.guidance}' for --diffusion_type particle")

    def eps_fn(self, x: torch.Tensor, t_vec: torch.Tensor) -> torch.Tensor:
        """Noise prediction of the score_sde network: ``eps = -score * sqrt(1 - alpha_bar(t))``.

        Goes through ``VPDynamics.score`` (``get_score_fn(..., continuous=True)``), i.e. the same
        time-label scaling as the original SDE purifier.
        """
        score = self.dynamics.score(t_vec, x.flatten(1)).view_as(x)
        sigma = torch.sqrt(1 - alpha_bar_cont(t_vec)).to(dtype=x.dtype, device=x.device)
        return -score * sigma.view(-1, *([1] * (x.ndim - 1)))

    # ------------------------------------------------------------------ interface of PurifiedClassifier
    def reset_counter(self):
        self.counter = torch.zeros(1, dtype=torch.int, device=self.device)

    def set_tag(self, tag=None):
        self.tag = tag

    @property
    def n_particles(self) -> int:
        return self.sampler.n_particles

    def particle_logits(self, x: torch.Tensor) -> torch.Tensor:
        """(B, K, n_classes) logits of the K purified copies of every image (x in [0, 1])."""
        xp = self.sampler(x)
        B, K = xp.shape[:2]
        return self.classifier(xp.flatten(0, 1)).view(B, K, -1)

    def purify(self, x):
        """One purified image per input (BPDA interface); only defined for K = 1."""
        if self.n_particles != 1:
            raise NotImplementedError("mode='purify' (BPDA) needs --pcp_n_particles 1")
        return self.sampler(x)[:, 0]

    def forward(self, x, mode="purify_and_classify"):
        if mode == "purify":
            return self.purify(x)
        if mode == "classify":
            return self.classifier(x)
        if mode != "purify_and_classify":
            raise NotImplementedError(f"unknown mode: {mode}")
        logits = self.particle_logits(x)
        self.last_particle_logits = logits.detach()  # SAFE: diagnostics copy, the returned value keeps its graph
        self.counter += 1
        return vote(logits, self.output_vote)
