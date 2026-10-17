#!/bin/bash
# Download the test set and the checkpoints, on the login node (needs internet; light work only).
#
#   bash experiments/slurm/prepare_assets.sh          # CIFAR-10 + WRN-28-10 + score model
#   bash experiments/slurm/prepare_assets.sh --all    # also ResNet-50 and WRN-70-16-dropout
#
# The Google Drive files are those linked from the DiffPure README. If gdown hits a Drive quota,
# download them in a browser and copy them to the paths printed below.
set -euo pipefail
cd "$(dirname "$0")/../.."
source experiments/slurm/env.sh
cdp_activate

# gdown (used by RobustBench and fetch_file below) saves cookies to ~/.cache/gdown/cookies.txt
# without creating the directory. If $HOME is not writable, keep that cache under $CDP_WORK.
if ! mkdir -p "$HOME/.cache/gdown" 2>/dev/null; then
  export HOME="$CDP_WORK"
  mkdir -p "$HOME/.cache/gdown"
  echo "note: \$HOME not writable; using HOME=$HOME for the gdown cache"
fi

# CIFAR-10 test split (torchvision) and the RobustBench "Standard" WRN-28-10 classifier
python - <<'EOF'
from robustbench import load_model

from src.data.specs import get_dataset_spec
from src.models.classifiers import ROBUSTBENCH_MODEL_DIR

print("CIFAR-10 test images:", len(get_dataset_spec("cifar10").build_split(train=False)))
load_model(model_name="Standard", model_dir=ROBUSTBENCH_MODEL_DIR, dataset="cifar10", threat_model="Linf")
print("RobustBench Standard WRN-28-10 ready in", ROBUSTBENCH_MODEL_DIR)
EOF

fetch_file() {  # <drive file id> <destination>
  if [ -f "$2" ]; then echo "have $2"; return; fi
  mkdir -p "$(dirname "$2")"
  gdown "$1" -O "$2" || echo "MANUAL: download https://drive.google.com/file/d/$1 to $2"
}

fetch_folder() {  # <drive folder id> <destination dir>; the folder must contain weights.pt
  if [ -f "$2/weights.pt" ]; then echo "have $2/weights.pt"; return; fi
  mkdir -p "$2"
  gdown --folder "https://drive.google.com/drive/folders/$1" -O "$2" \
    || echo "MANUAL: download https://drive.google.com/drive/folders/$1 into $2"
  [ -f "$2/weights.pt" ] || echo "WARNING: $2/weights.pt not found; move the downloaded file there"
}

# score-SDE model used by every CIFAR-10 run (score_sde_pytorch vp/cifar10_ddpmpp_deep_continuous)
fetch_file 16_-Ahc6ImZV5ClUc0vM5Iivf8OJ1VSif pretrained/score_sde/checkpoint_8.pth

if [ "${1:-}" = "--all" ]; then
  fetch_folder 1SEGilIEAnx9OC1JVhmOynreCF3oog7Fi pretrained/cifar10/resnet-50          # *_rn50* scripts
  fetch_folder 1istqcnPNXJ-TQFdxRqNJuInzOtlYOG1b pretrained/cifar10/wrn-70-16-dropout  # *_70-16-dp* scripts
fi
