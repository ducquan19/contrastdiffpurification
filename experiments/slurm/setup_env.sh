#!/bin/bash
# One-time setup, run on the login node: a Python venv and its packages, all under /media/<user>.
#
#   TORCH_INDEX=https://download.pytorch.org/whl/cu124 bash experiments/slurm/setup_env.sh
#
# PYTHON     interpreter the venv is created from (default: python3; needs >= 3.9). If it comes
#            from an environment module, load that module first and set CDP_MODULES to it so the
#            Slurm jobs load it too.
# TORCH_INDEX selects the CUDA build of PyTorch. It must not exceed the "CUDA Version" that
#            nvidia-smi reports on a compute node (check with: srun --mem=4G --gres=gpu:1 nvidia-smi).
#
# external/score_sde is not installed here: it compiles CUDA ops and must be built on a GPU node
# afterwards, with experiments/slurm/build_score_sde.sh.
set -euo pipefail
cd "$(dirname "$0")/../.."
source experiments/slurm/env.sh

: "${PYTHON:=python3}"
: "${TORCH_INDEX:=https://download.pytorch.org/whl/cu124}"

mkdir -p "$XDG_CACHE_HOME" "$(dirname "$CDP_ENV")" logs/slurm

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "error: $PYTHON not found; set PYTHON=<path to python3.9+> or load a Python module" >&2
  exit 1
fi
if ! "$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 9))'; then
  echo "error: $("$PYTHON" -V 2>&1) is too old; Python >= 3.9 is required" >&2
  exit 1
fi

if [ ! -f "$CDP_ENV/bin/activate" ]; then
  echo "creating venv $CDP_ENV from $("$PYTHON" -V 2>&1) ($(command -v "$PYTHON"))"
  if ! "$PYTHON" -m venv "$CDP_ENV"; then
    echo "error: '$PYTHON -m venv' failed. On Debian/Ubuntu the venv module is a separate package" \
         "(python3-venv); ask the administrators, or use another Python via PYTHON=..." >&2
    exit 1
  fi
fi
cdp_activate

python -m pip install --upgrade pip
pip install torch torchvision --index-url "$TORCH_INDEX"
pip install -r requirements.txt
pip install ./external/auto-attack
pip install gdown pytest wheel    # checkpoint download, CPU test suite, score_sde build (bdist_wheel)

python -m pytest tests -q -p no:cacheprovider
python -c "import autoattack, robustbench, torchsde; print('imports ok')"
echo "next: srun --mem=16G --gres=gpu:1 --cpus-per-task=4 bash experiments/slurm/build_score_sde.sh"

# record the exact environment next to the results, for reproducibility
python -V > "logs/env-$(date +%Y%m%d).txt"
pip freeze >> "logs/env-$(date +%Y%m%d).txt"
echo "environment ready: $CDP_ENV (package list in logs/env-$(date +%Y%m%d).txt)"
