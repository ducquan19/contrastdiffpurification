"""The strict config loader: no silent defaults, no unnoticed typos."""
import dataclasses

import pytest
import yaml

from src.config import ConfigMain, DataConfig, ModelConfig, SamplingConfig, load_config
from src.data.specs import DATASETS
from src.paths import config_path


def _cifar10_raw():
    with open(config_path("cifar10.yml")) as f:
        return yaml.safe_load(f)


def _write(tmp_path, raw):
    p = tmp_path / "config.yml"
    p.write_text(yaml.safe_dump(raw))
    return str(p)


def test_experiment_configs_load():
    for spec in DATASETS.values():
        config = load_config(spec.config_file)
        assert isinstance(config, ConfigMain)
        assert config.model.name == "ncsnpp"
        assert config.training.sde == "vpsde"
        assert config.training.continuous is True
        assert config.device is None  # set by the entry point, never by the YAML


def test_every_required_field_is_present_in_the_shipped_configs():
    """A field the code reads must come from the file, not from a default."""
    for section_cls in (DataConfig, ModelConfig, SamplingConfig):
        for field in dataclasses.fields(section_cls):
            assert field.default is dataclasses.MISSING, (
                f"{section_cls.__name__}.{field.name} has a default; a missing key would pass "
                "unnoticed instead of being reported")


def test_missing_key_is_reported(tmp_path):
    raw = _cifar10_raw()
    del raw["training"]["likelihood_weighting"]
    with pytest.raises(ValueError, match="missing key"):
        load_config(_write(tmp_path, raw))


def test_unknown_key_is_reported(tmp_path):
    raw = _cifar10_raw()
    raw["model"]["nff"] = 128
    with pytest.raises(ValueError, match="unknown key"):
        load_config(_write(tmp_path, raw))


def test_all_problems_are_reported_together(tmp_path):
    raw = _cifar10_raw()
    del raw["model"]["nf"]
    raw["data"]["typo"] = 1
    del raw["sampling"]
    with pytest.raises(ValueError) as excinfo:
        load_config(_write(tmp_path, raw))
    message = str(excinfo.value)
    assert "missing section(s): sampling" in message
    assert "'data' unknown key(s): typo" in message
    assert "'model' missing key(s): nf" in message


def test_unknown_section_is_reported(tmp_path):
    raw = _cifar10_raw()
    raw["trainig"] = {}
    with pytest.raises(ValueError, match="unknown top-level section"):
        load_config(_write(tmp_path, raw))
