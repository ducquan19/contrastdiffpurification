#!/bin/bash
# Build and install external/score_sde, whose import loads two compiled C++/CUDA ops.
# Run it on a GPU node, after setup_env.sh:
#
#   srun --mem=16G --gres=gpu:1 --cpus-per-task=4 bash experiments/slurm/build_score_sde.sh
#
# score_sde's setup.py compiles the CUDA kernels only when it sees a GPU or when FORCE_CUDA=1.
# Without a GPU and without FORCE_CUDA it would build the C++ part alone and `import score_sde`
# would then fail. To build on the login node (needs nvcc there, no GPU queue):
#
#   CUDA_HOME=/usr/local/cuda FORCE_CUDA=1 bash experiments/slurm/build_score_sde.sh The build needs nvcc
# (same CUDA major version as torch.version.cuda): load the cluster's CUDA module first, e.g.
#   module avail cuda; module load cuda/12.4; export CDP_MODULES="cuda/12.4"
# or point CUDA_HOME at a CUDA toolkit.
set -euo pipefail
cd "$(dirname "$0")/../.."
source experiments/slurm/env.sh
cdp_activate

if [ "${FORCE_CUDA:-0}" = "1" ]; then
  # Build the CUDA ops without a GPU (e.g. on the login node): compiling only needs nvcc.
  # Default targets: V100 (sm_70) and A100 (sm_80); override TORCH_CUDA_ARCH_LIST for other GPUs.
  export FORCE_CUDA
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0;8.0}"
  echo "FORCE_CUDA=1: building CUDA ops for TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST without a GPU"
else
  python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 'error: no GPU visible; run this through srun --gres=gpu:1, or set FORCE_CUDA=1 to build without a GPU')"
fi
if ! command -v nvcc >/dev/null 2>&1 && [ -z "${CUDA_HOME:-}" ]; then
  echo "error: nvcc not found and CUDA_HOME unset; load a CUDA module (module avail cuda) or set CUDA_HOME" >&2
  exit 1
fi
python -c "import torch; print('torch', torch.__version__, 'built for CUDA', torch.version.cuda)"
nvcc --version 2>/dev/null | tail -1 || true

# setup.py imports torch, so build against the torch already installed in the venv
pip install --no-build-isolation --no-deps ./external/score_sde
if [ "${FORCE_CUDA:-0}" = "1" ]; then
  # without a GPU driver the import may fail even though the build succeeded; the first GPU job
  # (experiments/slurm/smoke.sbatch) is the real check
  python -c "import score_sde; from score_sde.op import upfirdn2d; print('score_sde ok')" \
    || echo "warning: 'import score_sde' failed here (no GPU driver?); verify it in a GPU job (smoke.sbatch)"
else
  python -c "import score_sde; from score_sde.op import upfirdn2d; print('score_sde ok')"
fi
