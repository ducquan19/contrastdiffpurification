"""Readers for the German Traffic Sign Recognition Benchmark (43 classes, RGB, native size 32).

Expected layout under ``root``::

    GTSRB/Training/<class_id>/*.ppm  + GT-<class_id>.csv   training images and labels
    GTSRB/Final_Test/Images/*.ppm                          test images
    GT-final_test.csv                                      test labels

Label files are ``;``-separated with the columns
``Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId``. The ROI locates the sign inside an
image that carries a border of about 10% of the sign size around it, so ``enable_crop`` trades
that border for a tighter crop.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from torchvision.datasets import VisionDataset


@dataclass(frozen=True)
class _Item:
    """One training image: path relative to the training folder, label and sign ROI."""

    filename: str
    class_id: int
    roi: tuple[int, int, int, int] | None = None


def _roi(row: pd.Series) -> tuple[int, int, int, int]:
    """The sign's bounding box ``(x1, y1, x2, y2)`` from a label row."""
    return (int(row["Roi.X1"]), int(row["Roi.Y1"]), int(row["Roi.X2"]), int(row["Roi.Y2"]))


class GTSRBDatasetTrain(VisionDataset):
    """The GTSRB training split, ordered by file name."""

    def __init__(
        self,
        root: str | Path,
        transforms: Callable | None = None,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        enable_crop: bool = False,
    ) -> None:
        super().__init__(str(root), transforms, transform, target_transform)
        self.enable_crop = enable_crop
        self._img_folder = Path(root) / "GTSRB" / "Training"
        self.images = self._read_items()

    def _read_items(self) -> list[_Item]:
        """Every training image that exists on disk, in a deterministic order."""
        items: list[_Item] = []
        for class_dir in sorted(p for p in self._img_folder.iterdir() if p.is_dir()):
            label_file = class_dir / f"GT-{class_dir.name}.csv"
            if label_file.is_file():
                items.extend(self._items_from_labels(class_dir, label_file))
            else:
                warnings.warn(f"no label file in {class_dir}; labelling by folder name, no ROI")
                items.extend(
                    _Item(f"{class_dir.name}/{path.name}", int(class_dir.name))
                    for path in sorted(class_dir.glob("*.ppm"))
                )
        items.sort(key=lambda item: item.filename)
        return items

    def _items_from_labels(self, class_dir: Path, label_file: Path) -> Iterator[_Item]:
        """The items of one class folder, skipping rows whose image is missing."""
        for _, row in pd.read_csv(label_file, sep=";").iterrows():
            filename = row["Filename"]
            if not (class_dir / filename).is_file():
                continue
            yield _Item(f"{class_dir.name}/{filename}", int(row["ClassId"]), _roi(row))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        item = self.images[index]
        img = Image.open(self._img_folder / item.filename).convert("RGB")
        if self.enable_crop and item.roi is not None:
            img = img.crop(item.roi)
        if self.transform is not None:
            img = self.transform(img)
        label = item.class_id
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label


class GTSRBDatasetTest(VisionDataset):
    """The GTSRB test split, ordered by the label file."""

    def __init__(
        self,
        root: str | Path,
        transforms: Callable | None = None,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        enable_crop: bool = False,
    ) -> None:
        super().__init__(str(root), transforms, transform, target_transform)
        self.enable_crop = enable_crop
        self._img_folder = Path(root) / "GTSRB" / "Final_Test" / "Images"
        self._label_file = Path(root) / "GT-final_test.csv"
        self.labels = self._read_labels()

    def _read_labels(self) -> pd.DataFrame:
        """The label rows whose image is present on disk."""
        labels = pd.read_csv(self._label_file, sep=";")
        on_disk = {path.name for path in self._img_folder.glob("*.ppm")}
        keep = labels["Filename"].isin(on_disk)
        unlabelled = len(on_disk) - int(keep.sum())
        if unlabelled or not keep.all():
            warnings.warn(
                f"{self._img_folder} and {self._label_file} disagree: {int((~keep).sum())} "
                f"label(s) without an image, {unlabelled} image(s) without a label. The dataset "
                f"is incomplete; consider downloading it again."
            )
        return labels[keep].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        row = self.labels.iloc[index]
        img = Image.open(self._img_folder / row["Filename"]).convert("RGB")
        if self.enable_crop:
            img = img.crop(_roi(row))
        if self.transform is not None:
            img = self.transform(img)
        label = int(row["ClassId"])
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label
