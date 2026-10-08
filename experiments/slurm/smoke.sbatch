#!/bin/bash
# Short end-to-end check on one GPU: baseline and contrastive purification under AutoAttack rand,
# on 4 images with EOT 1. Submit from the repository root:
#
#   sbatch experiments/slurm/smoke.sbatch
#
# The elapsed time printed per run is the basis for --time of the full runs.
#SBATCH --job-name=cdp-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[ -f experiments/slurm/env.sh ] || { echo "error: submit from the repository root" >&2; exit 1; }
source experiments/slurm/env.sh
cdp_module_purge
cdp_activate

echo "host: $(hostname)  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"

for guidance in none contrastive; do
  start=$(date +%s)
  python -m scripts.eval_robustness --guidance "$guidance" --config cifar10.yml \
    --classifier_name cifar10-wideresnet-28-10 --contrastive_classifier cifar10-wideresnet-28-10 \
    --attack_version rand --eot_iter 1 --num_sub 4 --adv_batch_size 4 \
    --exp smoke -i "$guidance"
  echo "== $guidance finished in $(( $(date +%s) - start )) s"
done
echo "results: outputs/smoke/"
