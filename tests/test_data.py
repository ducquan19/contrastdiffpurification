"""Dataset specs, evaluation-subset selection and test-time preprocessing.

The transform tests pin down the two equivalences that justified replacing the three original
per-dataset pipelines with one ``Resize + ToTensor`` (see src/data/specs.py).
"""
import numpy as np
import pytest

from src.data.specs import DATASET_NAMES, DATASETS, get_dataset_spec
from src.data.loaders import select_subset_indices


# --------------------------------------------------------------------------- specs
def test_registry_metadata():
    expected = {"cifar10": 10, "cifar100": 100, "gtsrb": 43}
    assert set(DATASET_NAMES) == set(expected)
    for name, num_classes in expected.items():
        spec = DATASETS[name]
        assert spec.num_classes == num_classes
        assert spec.image_size == 32
        assert spec.img_shape == (3, 32, 32)
        assert spec.root.name == name


def test_unknown_dataset_is_rejected_by_name():
    with pytest.raises(ValueError, match="unknown dataset"):
        get_dataset_spec("cifar-10")  # near-miss spelling, no substring matching


def test_spec_image_size_matches_its_config():
    from src.config import load_config
    for spec in DATASETS.values():
        config = load_config(spec.config_file)
        assert config.data.image_size == spec.image_size
        assert config.data.num_channels == spec.num_channels


# --------------------------------------------------------------- subset selection
def test_subset_is_reproducible_and_seed_dependent():
    a = select_subset_indices(10000, 64, data_seed=0)
    assert np.array_equal(a, select_subset_indices(10000, 64, data_seed=0))
    assert not np.array_equal(a, select_subset_indices(10000, 64, data_seed=1))


def test_subset_is_a_valid_sample_without_repeats():
    idx = select_subset_indices(10000, 200, data_seed=3)
    assert len(idx) == 200 and len(set(idx.tolist())) == 200
    assert idx.min() >= 0 and idx.max() < 10000


def test_subset_matches_the_original_formula():
    """Regression guard: the evaluated images must not shift under refactoring."""
    for data_seed in (0, 1, 7):
        for num_sub in (64, 200):
            expected = np.random.RandomState(data_seed).choice(10000, num_sub, replace=False)
            assert np.array_equal(select_subset_indices(10000, num_sub, data_seed), expected)


def test_subset_size_is_validated():
    with pytest.raises(ValueError, match="exceeds the test split size"):
        select_subset_indices(100, 101, data_seed=0)
    with pytest.raises(ValueError, match="must be positive"):
        select_subset_indices(100, 0, data_seed=0)


# ------------------------------------------------------------------ preprocessing
def test_pil_to_tensor_path_equals_to_tensor():
    """CIFAR-100 originally used PILToTensor + ConvertImageDtype instead of ToTensor."""
    transforms = pytest.importorskip("torchvision.transforms")
    import torch
    from PIL import Image

    img = Image.fromarray(np.random.RandomState(0).randint(0, 256, (32, 32, 3), dtype=np.uint8))
    to_tensor = transforms.ToTensor()(img)
    legacy = transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
    ])(img)
    assert torch.equal(to_tensor, legacy)


def test_resize_is_a_noop_at_the_native_size():
    """CIFAR-10 originally had no Resize; the shared pipeline adds one."""
    transforms = pytest.importorskip("torchvision.transforms")
    import torch
    from PIL import Image

    img = Image.fromarray(np.random.RandomState(1).randint(0, 256, (32, 32, 3), dtype=np.uint8))
    assert torch.equal(transforms.ToTensor()(img),
                       DATASETS["cifar10"].eval_transform()(img))


@pytest.mark.parametrize("name", sorted(DATASET_NAMES))
def test_eval_transform_outputs_unit_range_images(name):
    pytest.importorskip("torchvision.transforms")
    from PIL import Image

    spec = DATASETS[name]
    # a non-native size, as GTSRB images come in varying sizes
    img = Image.fromarray(np.random.RandomState(2).randint(0, 256, (47, 39, 3), dtype=np.uint8))
    out = spec.eval_transform()(img)
    assert out.shape == spec.img_shape
    assert out.dtype.is_floating_point
    assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0
