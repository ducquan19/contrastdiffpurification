"""Filesystem layout: roots are independent of the working directory, entry points exist."""
import os

from src import paths
from src.data.specs import DATASETS


def test_roots_are_under_the_repo_by_default():
    if any(k in os.environ for k in ("CDP_DATA_ROOT", "CDP_PRETRAINED_ROOT", "CDP_OUTPUT_ROOT")):
        return  # redirected on purpose
    for root in (paths.DATA_ROOT, paths.PRETRAINED_ROOT, paths.OUTPUT_ROOT):
        assert os.path.commonpath([root, paths.REPO_ROOT]) == paths.REPO_ROOT


def test_config_path_resolves_bare_names():
    assert paths.config_path("cifar10.yml") == os.path.join(paths.CONFIG_ROOT, "cifar10.yml")


def test_every_dataset_config_exists():
    for spec in DATASETS.values():
        assert os.path.isfile(paths.config_path(spec.config_file)), spec.config_file


def test_entry_points_and_experiments_exist():
    for rel in ("scripts/eval_robustness.py", "scripts/eval_bpda.py", "scripts/train_score_sde.py",
                "scripts/train_classifier.py", "scripts/convert_lightning_checkpoint.py",
                "scripts/download_gtsrb.sh"):
        assert os.path.isfile(os.path.join(paths.REPO_ROOT, rel)), rel
    n = sum(len([f for f in os.listdir(os.path.join(paths.REPO_ROOT, "experiments", d))
                 if f.endswith(".sh")])
            for d in ("baseline", "contrastive", "ablation_t"))
    assert n == 21
