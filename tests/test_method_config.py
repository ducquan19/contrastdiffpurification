"""The method config layer: the published hyperparameters, and CLI overrides."""
import pytest
import yaml

from src.method import ContrastiveConfig, load_method_config, resolve_contrastive
from src.paths import config_path


def test_shipped_config_holds_the_published_cifar10_values():
    """configs/method/contrastive.yaml must match the original CIFAR-10 scripts."""
    c = load_method_config()
    assert c.guidance_strength == 500      # lambda
    assert c.temperature == 1.0            # tau
    assert c.drift_min == 0
    assert c.drift_max == 100000
    assert c.embed_lr == 0.01
    assert c.tau_minus == 0.1 and c.beta == 1.0 and c.estimator == "hard"


def test_missing_and_unknown_keys_are_reported(tmp_path):
    raw = yaml.safe_load(open(config_path("method/contrastive.yaml")))
    del raw["beta"]
    raw["lambda"] = 1
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError) as e:
        load_method_config(str(p))
    assert "missing key(s): beta" in str(e.value) and "unknown key(s): lambda" in str(e.value)


def test_invalid_values_are_rejected():
    base = load_method_config().as_dict()
    for bad in ({"estimator": "nope"}, {"drift_min": -1},
                {"drift_min": 10, "drift_max": 5}, {"temperature": 0}):
        with pytest.raises(ValueError):
            ContrastiveConfig(**{**base, **bad})


class _Args:
    domain = "cifar10"
    method = None
    contrastive_classifier = "cifar10-wideresnet-28-10"
    contrastive_n_classes = None
    contrastive_embed_lr = None
    contrastive_drift_min = None
    contrastive_drift_max = None
    contrastive_tau_minus = None
    contrastive_beta = None
    contrastive_temperature = None
    contrastive_guidance_strength = None
    contrastive_estimator = None


def test_resolution_falls_back_to_the_file_and_the_dataset():
    c = resolve_contrastive(_Args())
    assert c.guidance_strength == 500 and c.temperature == 1.0
    assert c.n_classes == 10  # from the CIFAR-10 dataset spec, not repeated on the CLI


def test_command_line_overrides_the_file():
    args = _Args()
    args.contrastive_guidance_strength = 1234.0
    args.contrastive_tau_minus = 0.25
    args.contrastive_drift_min = 40
    c = resolve_contrastive(args)
    assert c.guidance_strength == 1234.0 and c.tau_minus == 0.25 and c.drift_min == 40
    assert c.temperature == 1.0  # untouched, still from the file


def test_classifier_is_required():
    args = _Args()
    args.contrastive_classifier = None
    with pytest.raises(ValueError, match="contrastive_classifier"):
        resolve_contrastive(args)
