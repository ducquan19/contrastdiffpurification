"""Dataset registry: image geometry, class count, config file and test-set factory.

Every dataset is evaluated with ``Resize((S, S)) + ToTensor()``, giving floats in [0, 1].
This is numerically equivalent to the original per-dataset pipelines (see tests/test_data.py).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import Dataset

from src.paths import data_path


@dataclass(frozen=True)
class DatasetSpec:
    """Everything the pipeline needs to know about one dataset."""

    name: str
    num_classes: int
    image_size: int
    config_file: str          # the score-SDE config in configs/
    root_subdir: str          # location under the data root
    num_channels: int = 3

    @property
    def root(self) -> Path:
        """Directory the dataset is read from."""
        return Path(data_path(self.root_subdir)) if self.root_subdir else Path(data_path())

    @property
    def img_shape(self) -> tuple[int, int, int]:
        """``(C, H, W)`` of an evaluated image."""
        return (self.num_channels, self.image_size, self.image_size)

    def eval_transform(self) -> Callable:
        """Test-time preprocessing: resize to ``image_size``, then convert to a [0, 1] tensor."""
        import torchvision.transforms as transforms

        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])

    def build_split(
        self,
        train: bool,
        transform: Callable | None = None,
        root: str | Path | None = None,
    ) -> Dataset:
        """The full training or test split.

        ``transform`` defaults to ``eval_transform()`` and ``root`` to ``self.root``. CIFAR is
        downloaded on first use; GTSRB must already be on disk (scripts/download_gtsrb.sh).
        """
        import torchvision.datasets

        if transform is None:
            transform = self.eval_transform()
        root = Path(root) if root is not None else self.root

        if self.name == "cifar10":
            return torchvision.datasets.CIFAR10(
                root=str(root), train=train, download=True, transform=transform)

        if self.name == "cifar100":
            return torchvision.datasets.CIFAR100(
                root=str(root), train=train, download=True, transform=transform)

        if self.name == "gtsrb":
            from src.data.gtsrb import GTSRBDatasetTest, GTSRBDatasetTrain

            dataset_cls = GTSRBDatasetTrain if train else GTSRBDatasetTest
            # enable_crop=False keeps the full sign image, as in the original code
            return dataset_cls(root=root, transform=transform, enable_crop=False)

        raise NotImplementedError(f"no dataset builder for '{self.name}'")


DATASETS: dict[str, DatasetSpec] = {
    spec.name: spec
    for spec in (
        DatasetSpec(
            name="cifar10",
            num_classes=10,
            image_size=32,
            config_file="cifar10.yml",
            root_subdir="cifar10"
        ),

        DatasetSpec(
            name="cifar100",
            num_classes=100,
            image_size=32,
            config_file="cifar100.yml",
            root_subdir="cifar100"
        ),

        DatasetSpec(
            name="gtsrb",
            num_classes=43,
            image_size=32,
            config_file="gtsrb.yml",
            root_subdir="gtsrb"
        ),
    )
}

DATASET_NAMES: tuple[str, ...] = tuple(DATASETS)


def get_dataset_spec(name: str) -> DatasetSpec:
    """Look up a dataset by exact name (``--domain``)."""
    try:
        return DATASETS[name]
    except KeyError:
        raise ValueError(
            f"unknown dataset '{name}'; available: {', '.join(DATASET_NAMES)}"
        ) from None
