#!/bin/bash
# Build and install external/score_sde, whose import loads two compiled C++/CUDA ops.
# Run it on a GPU node, after setup_env.sh:
#
#   srun --mem=16G --gres=gpu:1 --cpus-per-task=4 bash experiments/slurm/build_score_sde.sh
#
# score_sde's setup.py compiles the CUDA kernels only when it sees a GPU; on the login node it
# would build the C++ part alone and `import score_sde` would then fail. The build needs nvcc
# (same CUDA major version as torch.version.cuda): load the cluster's CUDA module first, e.g.
#   module avail cuda; module load cuda/12.4; export CDP_MODULES="cuda/12.4"
# or point CUDA_HOME at a CUDA toolkit.
set -euo pipefail
cd "$(dirname "$0")/../.."
source experiments/slurm/env.sh
cdp_activate

python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 'error: no GPU visible; run this through srun --gres=gpu:1')"
if ! command -v nvcc >/dev/null 2>&1 && [ -z "${CUDA_HOME:-}" ]; then
  echo "error: nvcc not found and CUDA_HOME unset; load a CUDA module (module avail cuda) or set CUDA_HOME" >&2
  exit 1
fi
python -c "import torch; print('torch', torch.__version__, 'built for CUDA', torch.version.cuda)"
nvcc --version 2>/dev/null | tail -1 || true

# setup.py imports torch, so build against the torch already installed in the venv
pip install --no-build-isolation --no-deps ./external/score_sde
python -c "import score_sde; from score_sde.op import upfirdn2d; print('score_sde ok')"
