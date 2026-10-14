"""Filesystem locations used by the code.

Every location defaults to a folder under the repository root, independent of the current
working directory. The data, pretrained and output roots can be redirected with an environment
variable, e.g. to a shared dataset or checkpoint store on a cluster.

    data/          datasets                     (CDP_DATA_ROOT)
    pretrained/    checkpoints, RobustBench cache (CDP_PRETRAINED_ROOT)
    outputs/       experiment results and logs  (CDP_OUTPUT_ROOT)
    configs/       diffusion-model YAML configs
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_ROOT = os.environ.get("CDP_DATA_ROOT", os.path.join(REPO_ROOT, "data"))
PRETRAINED_ROOT = os.environ.get("CDP_PRETRAINED_ROOT", os.path.join(REPO_ROOT, "pretrained"))
OUTPUT_ROOT = os.environ.get("CDP_OUTPUT_ROOT", os.path.join(REPO_ROOT, "outputs"))
CONFIG_ROOT = os.path.join(REPO_ROOT, "configs")


def data_path(*parts):
    """Path under the dataset root."""
    return os.path.join(DATA_ROOT, *parts)


def pretrained_path(*parts):
    """Path under the checkpoint root."""
    return os.path.join(PRETRAINED_ROOT, *parts)


def output_path(*parts):
    """Path under the experiment-output root."""
    return os.path.join(OUTPUT_ROOT, *parts)


def config_path(name):
    """Resolve a config given as a bare file name (``cifar10.yml``) or as a path."""
    if os.path.isabs(name) or os.path.exists(name):
        return name
    return os.path.join(CONFIG_ROOT, name)
