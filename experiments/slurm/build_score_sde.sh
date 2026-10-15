#!/bin/bash
# Build and install external/score_sde, whose import loads two compiled C++/CUDA ops.
#
# score_sde's setup.py compiles the CUDA kernels only when it sees a GPU or when FORCE_CUDA=1;
# otherwise it builds the C++ part alone and `import score_sde` fails. Two ways to run it:
#
#   # on the login node, no GPU queue (needs a CUDA toolkit there):
#   CUDA_HOME=/usr/local/cuda-12.3 FORCE_CUDA=1 bash experiments/slurm/build_score_sde.sh
#
#   # on a GPU node:
#   srun --mem=16G --gres=gpu:1 --cpus-per-task=4 bash experiments/slurm/build_score_sde.sh
#
# The nvcc used is $CUDA_HOME/bin/nvcc (else the nvcc on PATH). Its CUDA *major* version must equal
# torch.version.cuda's (12 for a cu12x torch, 11 for cu118); the script stops otherwise.
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

# the nvcc that torch.utils.cpp_extension will use
if [ -n "${CUDA_HOME:-}" ]; then
  NVCC="$CUDA_HOME/bin/nvcc"
  export PATH="$CUDA_HOME/bin:$PATH"
elif command -v nvcc >/dev/null 2>&1; then
  NVCC="$(command -v nvcc)"
  export CUDA_HOME="$(dirname "$(dirname "$NVCC")")"
else
  echo "error: no nvcc: set CUDA_HOME to a CUDA toolkit (e.g. /usr/local/cuda-12.3) or load a CUDA module" >&2
  exit 1
fi
[ -x "$NVCC" ] || { echo "error: $NVCC not found or not executable; check CUDA_HOME" >&2; exit 1; }

NVCC_VERSION="$("$NVCC" --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p')"
TORCH_CUDA="$(python -c "import torch; print(torch.version.cuda or '')")"
echo "torch $(python -c 'import torch; print(torch.__version__)') built for CUDA $TORCH_CUDA"
echo "nvcc  $NVCC  (CUDA $NVCC_VERSION), CUDA_HOME=$CUDA_HOME"
if [ -z "$TORCH_CUDA" ]; then
  echo "error: this torch is a CPU-only build; reinstall it from a CUDA index (setup_env.sh TORCH_INDEX)" >&2
  exit 1
fi
if [ "${NVCC_VERSION%%.*}" != "${TORCH_CUDA%%.*}" ]; then
  echo "error: nvcc is CUDA $NVCC_VERSION but torch was built for CUDA $TORCH_CUDA (major versions" \
       "must match). Point CUDA_HOME at a matching toolkit, e.g. CUDA_HOME=/usr/local/cuda-12.3" \
       "for a cu12x torch, or a CUDA 11.8 module for a cu118 torch." >&2
  exit 1
fi

# --no-build-isolation builds against the torch in the venv; it then needs wheel in the venv too
python -c "import wheel" 2>/dev/null || pip install wheel
pip install --no-build-isolation --no-deps ./external/score_sde

if [ "${FORCE_CUDA:-0}" = "1" ]; then
  # without a GPU driver the import may fail even though the build succeeded; the first GPU job
  # (experiments/slurm/smoke.sbatch) is the real check
  python -c "import score_sde; from score_sde.op import upfirdn2d; print('score_sde ok')" \
    || echo "warning: 'import score_sde' failed here (no GPU driver?); verify it in a GPU job (smoke.sbatch)"
else
  python -c "import score_sde; from score_sde.op import upfirdn2d; print('score_sde ok')"
fi
