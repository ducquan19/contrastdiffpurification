#!/bin/bash
# One-time setup, run on the login node: conda environment and packages, all under /media/<user>.
#
#   bash experiments/slurm/setup_env.sh
#
# TORCH_INDEX selects the CUDA build of PyTorch. It must not exceed the "CUDA Version" that
# nvidia-smi reports on a compute node (check with: srun --mem=4G --gres=gpu:1 nvidia-smi).
# Only the GPU driver is needed: the CIFAR-10 / CIFAR-100 / GTSRB configs (fir: False) never use
# the compiled CUDA ops of score_sde, so nvcc is not required.
set -euo pipefail
cd "$(dirname "$0")/../.."
source experiments/slurm/env.sh

: "${PYTHON_VERSION:=3.10}"
: "${TORCH_INDEX:=https://download.pytorch.org/whl/cu121}"

mkdir -p "$CONDA_PKGS_DIRS" "$XDG_CACHE_HOME" logs/slurm

cdp_conda_init
if [ ! -d "$CDP_ENV" ]; then
  conda create -y -p "$CDP_ENV" python="$PYTHON_VERSION"
fi
conda activate "$CDP_ENV"

pip install torch torchvision --index-url "$TORCH_INDEX"
pip install -r requirements.txt
pip install gdown pytest          # checkpoint download, CPU test suite

python -m pytest tests -q -p no:cacheprovider
python -c "import src.models.score_sde, src.attacks.autoattack, robustbench, torchsde; print('imports ok')"

# record the exact environment next to the results, for reproducibility
pip freeze > "logs/env-$(date +%Y%m%d).txt"
echo "environment ready: $CDP_ENV (package list in logs/env-$(date +%Y%m%d).txt)"
