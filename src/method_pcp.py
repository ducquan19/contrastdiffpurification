"""Hyperparameters of the particle sampler and of the PCP guidance, from ``configs/method/pcp.yaml``.

Same pattern as ``src/method.py``: every field is required in the file, ``--pcp_<field>``
overrides it on the command line, and the resolved values are written to
``resolved_config.json``.

The *sampler* fields (``n_particles`` ... ``vote``) apply to every run with
``--diffusion_type particle``, i.e. also to the DiffPure (``--guidance none``) and ContrastDiffPure
(``--guidance contrastive``) baselines run with the same solver, steps and K -- which is what makes
the comparison equal-cost. The *guidance* fields apply only to ``--guidance pcp``.

====================  =========================================================
field                 meaning
====================  =========================================================
n_particles           K, purifications per image
steps                 reverse steps from t* to ~0
solver                ``ddim`` (with ``eta``) or ``em`` (Euler-Maruyama, as DiffPure)
eta                   DDIM stochasticity, 1 ~ DDPM ancestral
checkpoint            ``eps`` (recompute the score network in backward) or ``none``
vote                  prediction from the K particles: ``hard`` majority or ``soft`` mean prob.
encoder               F: ``robustbench:<name>[:<threat>]`` or a classifier name
encoder_feature       ``penultimate`` or ``logits``
bank                  prototype bank file, relative to pretrained/ (or absolute)
sem_strength          weight of the semantic pull (L_sem)
rep_strength          weight of the repulsion (L_rep)
tau_s, tau_t          temperatures of prediction and target (tau_t <= tau_s)
sem_t_min/max         L_sem active for t in [min, max] * t*
rep_t_min/max         L_rep active for t in [min, max] * t*
rep_size              side of the low-resolution space of L_rep
scale_mode            ``normalized`` (||d|| = strength ||eps||) or ``raw``
====================  =========================================================
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from typing import List

import yaml

from src.paths import config_path, pretrained_path

DEFAULT_PCP_CONFIG = "method/pcp.yaml"
SOLVERS = ("ddim", "em")
CHECKPOINT_MODES = ("none", "eps")
VOTES = ("hard", "soft")
SCALE_MODES = ("normalized", "raw")
ENCODER_FEATURES = ("penultimate", "logits")


@dataclass
class PCPConfig:
    # sampler (all --diffusion_type particle runs)
    n_particles: int
    steps: int
    solver: str
    eta: float
    checkpoint: str
    vote: str
    # PCP guidance (--guidance pcp)
    encoder: str
    encoder_feature: str
    bank: str
    sem_strength: float
    rep_strength: float
    tau_s: float
    tau_t: float
    sem_t_min: float
    sem_t_max: float
    rep_t_min: float
    rep_t_max: float
    rep_size: int
    scale_mode: str

    def __post_init__(self):
        def choice(name, value, options):
            if value not in options:
                raise ValueError(f"unknown {name} '{value}'; available: {', '.join(options)}")

        choice("solver", self.solver, SOLVERS)
        choice("checkpoint", self.checkpoint, CHECKPOINT_MODES)
        choice("vote", self.vote, VOTES)
        choice("scale_mode", self.scale_mode, SCALE_MODES)
        choice("encoder_feature", self.encoder_feature, ENCODER_FEATURES)
        if self.n_particles < 1:
            raise ValueError(f"n_particles must be >= 1, got {self.n_particles}")
        if self.steps < 1:
            raise ValueError(f"steps must be >= 1, got {self.steps}")
        if self.eta < 0:
            raise ValueError(f"eta must be >= 0, got {self.eta}")
        if self.sem_strength < 0 or self.rep_strength < 0:
            raise ValueError("guidance strengths must be >= 0")
        if not 0 < self.tau_t <= self.tau_s:
            raise ValueError(f"need 0 < tau_t <= tau_s, got tau_t={self.tau_t}, tau_s={self.tau_s}")
        for lo, hi, what in ((self.sem_t_min, self.sem_t_max, "sem"), (self.rep_t_min, self.rep_t_max, "rep")):
            if not 0 <= lo <= hi <= 1:
                raise ValueError(f"need 0 <= {what}_t_min <= {what}_t_max <= 1, got {lo}, {hi}")
        if self.rep_size < 1:
            raise ValueError(f"rep_size must be >= 1, got {self.rep_size}")

    def as_dict(self):
        return asdict(self)

    def bank_path(self) -> str:
        """Absolute path of the prototype bank (relative names live under pretrained/)."""
        return self.bank if os.path.isabs(self.bank) else pretrained_path(self.bank)


PCP_FIELDS = {f.name: f.type for f in fields(PCPConfig)}
_TYPES = {"int": int, "float": float, "str": str}


def pcp_cli_dest(field_name: str) -> str:
    return f"pcp_{field_name}"


def load_pcp_config(name_or_path: str = DEFAULT_PCP_CONFIG) -> PCPConfig:
    """Load ``configs/method/pcp.yaml`` (or a path); every field must be present."""
    source = config_path(name_or_path)
    with open(source, "r") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{source}: expected a mapping at the top level")
    problems: List[str] = []
    missing = sorted(set(PCP_FIELDS) - set(raw))
    unknown = sorted(set(raw) - set(PCP_FIELDS))
    if missing:
        problems.append(f"missing key(s): {', '.join(missing)}")
    if unknown:
        problems.append(f"unknown key(s): {', '.join(unknown)}")
    if problems:
        raise ValueError(f"{source}:\n  - " + "\n  - ".join(problems))
    return PCPConfig(**raw)


def resolve_pcp(args) -> PCPConfig:
    """File first (``--pcp_method``), then every ``--pcp_<field>`` that was given."""
    config = load_pcp_config(getattr(args, "pcp_method", None) or DEFAULT_PCP_CONFIG)
    overrides = {name: getattr(args, pcp_cli_dest(name), None) for name in PCP_FIELDS}
    resolved = {**config.as_dict(), **{k: v for k, v in overrides.items() if v is not None}}
    return PCPConfig(**resolved)


def add_pcp_args(parser) -> None:
    """``--pcp_method`` and one ``--pcp_<field>`` flag per field (default None = keep the file)."""
    group = parser.add_argument_group(
        "particle sampler / PCP (--diffusion_type particle)",
        "Unset values come from --pcp_method.")
    group.add_argument("--pcp_method", type=str, default=DEFAULT_PCP_CONFIG,
                       help="PCP config under configs/ (default: %(default)s)")
    choices = {"solver": SOLVERS, "checkpoint": CHECKPOINT_MODES, "vote": VOTES,
               "scale_mode": SCALE_MODES, "encoder_feature": ENCODER_FEATURES}
    for name, type_ in PCP_FIELDS.items():
        py_type = _TYPES[type_] if isinstance(type_, str) else type_
        group.add_argument(f"--{pcp_cli_dest(name)}", type=py_type, default=None,
                           choices=choices.get(name))
