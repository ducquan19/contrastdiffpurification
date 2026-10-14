"""Hyperparameters of the contrastive guidance, loaded from ``configs/method/``.

These are the quantities Algorithm 1 of the paper requires (Bai et al., ICML 2024,
"Diffusion Models Demand Contrastive Guidance for Adversarial Purification to Advance"):

====================  ==================================
paper                 field here
====================  ==================================
lambda                ``guidance_strength``
tau                   ``temperature``
tau^-                 ``tau_minus``
beta                  ``beta``
HNM / InfoNCE         ``estimator`` (hard / soft)
F(.) training         ``embed_lr``
(not in the paper)    ``drift_min`` / ``drift_max``
====================  ==================================

Each field ``<f>`` can be overridden on the command line with ``--contrastive_<f>``.

``configs/method/contrastive.yaml`` holds the values used by the CIFAR-10 experiment scripts of
the original release (the paper gives no numbers for these hyperparameters).

``drift_min`` / ``drift_max`` select *which* reverse-diffusion steps receive the guidance.
Algorithm 1 guides every step; the window comes from the released code.

Every field in the file is required; command-line flags override individual values
(see ``resolve_contrastive``).
"""

from dataclasses import asdict, dataclass, fields
from typing import List

import yaml

from src.data.specs import get_dataset_spec
from src.paths import config_path

DEFAULT_METHOD_CONFIG = "method/contrastive.yaml"
ESTIMATORS = ("hard", "soft")


@dataclass
class ContrastiveConfig:
    """Algorithm 1's hyperparameters, plus the two that gate the guidance window."""

    temperature: float          # tau: softmax temperature of the contrastive loss
    tau_minus: float            # tau^-: assumed probability of observing another class
    beta: float                 # concentration, i.e. hardness of the negative mining
    estimator: str              # 'hard' = HNM loss (Eq. 10), 'soft' = InfoNCE (Eq. 9)
    guidance_strength: float    # lambda: scale of the contrastive gradient in Eq. (8)
    embed_lr: float             # Adam learning rate of the feature extractor F(.)
    drift_min: int              # first reverse step that receives guidance
    drift_max: int              # one past the last reverse step that receives guidance

    # Filled in from the dataset and the command line rather than the method file.
    classifier: str = ""        # backbone whose logits F(.) embeds
    n_classes: int = 0          # F(.) input width; the dataset's number of classes

    def __post_init__(self):
        if self.estimator not in ESTIMATORS:
            raise ValueError(
                f"unknown estimator '{self.estimator}'; available: {', '.join(ESTIMATORS)}")
        if self.drift_min < 0:
            raise ValueError(f"drift_min must be >= 0, got {self.drift_min}")
        if self.drift_max <= self.drift_min:
            raise ValueError(
                f"drift_max ({self.drift_max}) must be greater than drift_min ({self.drift_min})")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")

    def as_dict(self):
        return asdict(self)


_FILE_FIELDS = tuple(f.name for f in fields(ContrastiveConfig) if f.name not in ("classifier", "n_classes"))

# Every field and its type, in declaration order; each one is a ``--contrastive_<field>`` flag.
CLI_FIELDS = {f.name: f.type for f in fields(ContrastiveConfig)}


def cli_dest(field_name: str) -> str:
    """argparse ``dest`` (and flag name without ``--``) of a ``ContrastiveConfig`` field."""
    return f"contrastive_{field_name}"


def load_method_config(name_or_path: str = DEFAULT_METHOD_CONFIG) -> ContrastiveConfig:
    """Load ``configs/method/<name>.yaml``; every hyperparameter must be present."""
    source = config_path(name_or_path)
    with open(source, "r") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"{source}: expected a mapping at the top level")

    problems: List[str] = []
    missing = sorted(set(_FILE_FIELDS) - set(raw))
    unknown = sorted(set(raw) - set(_FILE_FIELDS))
    if missing:
        problems.append(f"missing key(s): {', '.join(missing)}")
    if unknown:
        problems.append(f"unknown key(s): {', '.join(unknown)}")
    if problems:
        raise ValueError(f"{source}:\n  - " + "\n  - ".join(problems))

    return ContrastiveConfig(**raw)


def resolve_contrastive(args) -> ContrastiveConfig:
    """The method config for this run: file first, then any command-line override.

    ``n_classes`` comes from the dataset spec unless given explicitly, so the number of classes
    is not repeated between the dataset registry and the command line.
    """
    config = load_method_config(getattr(args, "method", None) or DEFAULT_METHOD_CONFIG)

    # an unset flag is None and keeps the file value
    overrides = {name: getattr(args, cli_dest(name), None) for name in CLI_FIELDS}
    resolved = {**config.as_dict(), **{k: v for k, v in overrides.items() if v is not None}}
    if not resolved["n_classes"]:
        resolved["n_classes"] = get_dataset_spec(args.domain).num_classes
    if not resolved["classifier"]:
        raise ValueError("--contrastive_classifier is required with --guidance contrastive")

    return ContrastiveConfig(**resolved)
