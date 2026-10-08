"""Score-SDE (NCSN++) configuration loader, shared by evaluation and training.

A config file in ``configs/`` (e.g. ``cifar10.yml``) has five sections -- ``data``, ``model``,
``training``, ``optim`` and ``sampling`` -- each mirrored by a dataclass below. Every field is
required: ``load_config`` raises ``ValueError`` listing every missing or unknown key, with the
file and section it belongs to.

The schema holds exactly the keys read by ``src`` (including ``src/models/score_sde``) and ``scripts``.
``data.channels`` and ``model.num_classes`` are not part of it: only NCSNv2 and the
``InstanceNorm++`` normalization read them, and the models here are NCSN++ with GroupNorm.
"""

from dataclasses import dataclass, fields
from typing import Any, List
import sys

import yaml

from src.paths import config_path


@dataclass
class DataConfig:
    dataset: str          # selects the checkpoint in src/models/diffusion/{sde,ode}.py
    image_size: int
    num_channels: int
    centered: bool        # True => the model works on data scaled to [-1, 1]


@dataclass
class ModelConfig:
    # score network
    name: str
    nf: int
    ch_mult: List[int]
    num_res_blocks: int
    attn_resolutions: List[int]
    resamp_with_conv: bool
    conditional: bool
    dropout: float
    normalization: str
    nonlinearity: str
    init_scale: float
    skip_rescale: bool
    resblock_type: str
    progressive: str
    progressive_input: str
    progressive_combine: str
    embedding_type: str
    fourier_scale: float
    fir: bool
    fir_kernel: List[int]
    scale_by_sigma: bool
    # noise schedule
    sigma_min: float
    sigma_max: float
    num_scales: int
    beta_min: float
    beta_max: float
    # weight averaging
    ema_rate: float


@dataclass
class TrainingConfig:
    sde: str              # vpsde | subvpsde | vesde
    continuous: bool
    reduce_mean: bool
    likelihood_weighting: bool


@dataclass
class OptimConfig:
    optimizer: str
    lr: float
    beta1: float
    eps: float
    weight_decay: float
    warmup: int
    grad_clip: float


@dataclass
class SamplingConfig:
    method: str           # pc | ode
    predictor: str
    corrector: str
    n_steps_each: int
    noise_removal: bool
    probability_flow: bool
    snr: float


_SECTIONS = {
    "data": DataConfig,
    "model": ModelConfig,
    "training": TrainingConfig,
    "optim": OptimConfig,
    "sampling": SamplingConfig,
}


@dataclass
class ConfigMain:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    optim: OptimConfig
    sampling: SamplingConfig
    # Set by the evaluation / training entry points, not by the YAML file.
    device: Any = None


def _check_section(section: str, raw: Any) -> List[str]:
    """Return the problems (wrong type, missing or unknown keys) of one raw YAML section."""
    cls = _SECTIONS[section]
    if not isinstance(raw, dict):
        return [f"section '{section}' must be a mapping, got {type(raw).__name__}"]

    expected = {f.name for f in fields(cls)}
    problems = []
    missing = sorted(expected - set(raw))
    unknown = sorted(set(raw) - expected)
    if missing:
        problems.append(f"section '{section}' missing key(s): {', '.join(missing)}")
    if unknown:
        problems.append(f"section '{section}' unknown key(s): {', '.join(unknown)}")
    return problems


def load_config(name_or_path: str) -> ConfigMain:
    """Load and validate a config given as a bare file name (``cifar10.yml``) or a path."""
    source = config_path(name_or_path)
    with open(source, "r") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"{source}: expected a mapping at the top level")

    problems = []
    unknown = sorted(set(raw) - set(_SECTIONS))
    if unknown:
        problems.append(f"unknown top-level section(s): {', '.join(unknown)}")
    missing = sorted(set(_SECTIONS) - set(raw))
    if missing:
        problems.append(f"missing section(s): {', '.join(missing)}")

    for name in _SECTIONS:
        if name in raw:
            problems += _check_section(name, raw[name])

    if problems:
        raise ValueError(f"{source}:\n  - " + "\n  - ".join(problems))

    return ConfigMain(**{name: _SECTIONS[name](**raw[name]) for name in _SECTIONS})


# Existing Lightning checkpoints pickle these config classes under the module name
# ``basic_config_setting``; the alias lets them be unpickled (scripts/convert_lightning_checkpoint.py).
sys.modules.setdefault("basic_config_setting", sys.modules[__name__])
