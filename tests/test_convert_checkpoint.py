"""Checkpoints whose config objects were pickled under the old module name still load."""
import sys

import pytest

torch = pytest.importorskip("torch")

import src.config as config_module  # noqa: E402
from scripts.convert_lightning_checkpoint import LEGACY_CONFIG_MODULE, load_lightning_checkpoint  # noqa: E402
from src.config import DataConfig  # noqa: E402


def _save_legacy_checkpoint(path):
    """Write a checkpoint the way the old code did: DataConfig living in `basic_config_setting`."""
    cfg = DataConfig(dataset="GTSRB", image_size=32, num_channels=3, centered=True)
    original = DataConfig.__module__
    sys.modules[LEGACY_CONFIG_MODULE] = config_module
    DataConfig.__module__ = LEGACY_CONFIG_MODULE
    try:
        torch.save({"state_dict": {"model.w": torch.ones(2)}, "hyper_parameters": {"data": cfg}}, path)
    finally:
        DataConfig.__module__ = original
        del sys.modules[LEGACY_CONFIG_MODULE]
    return cfg


def test_legacy_checkpoint_loads_without_a_module_alias(tmp_path):
    path = tmp_path / "old.ckpt"
    cfg = _save_legacy_checkpoint(path)
    assert LEGACY_CONFIG_MODULE not in sys.modules

    ckpt = load_lightning_checkpoint(path)

    assert ckpt["hyper_parameters"]["data"] == cfg
    assert isinstance(ckpt["hyper_parameters"]["data"], DataConfig)
    assert torch.equal(ckpt["state_dict"]["model.w"], torch.ones(2))
    assert LEGACY_CONFIG_MODULE not in sys.modules
