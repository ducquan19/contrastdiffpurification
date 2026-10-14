#!/bin/bash
# Run one experiment script as a job array: one task per (seed, data_seed) pair, one GPU each.
# Submit from the repository root:
#
#   sbatch experiments/slurm/run_array.sbatch experiments/contrastive/run_cifar_rand_inf.sh
#
# The default array 0-23 covers SEEDS="121 122 123" x DATA_SEEDS="0 1 2 3 4 5 6 7"
# (task id = seed index * number of data seeds + data seed index). Other grids:
#
#   SEEDS="121" DATA_SEEDS="0 1" sbatch --array=0-1 experiments/slurm/run_array.sbatch <script>
#   T_LIST="50 100 150" sbatch experiments/slurm/run_array.sbatch experiments/ablation_t/<script>
#
# A task that hits the time limit can be resubmitted alone (sbatch --array=<id> ...): AutoAttack
# resumes from adv_*_eval_state.json in the run folder.
#SBATCH --job-name=cdp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --array=0-23
#SBATCH --output=logs/slurm/%x-%A_%a.out

set -euo pipefail
SCRIPT=${1:?usage: sbatch experiments/slurm/run_array.sbatch experiments/<group>/<script>.sh}
cd "$SLURM_SUBMIT_DIR"
[ -f experiments/slurm/env.sh ] || { echo "error: submit from the repository root" >&2; exit 1; }
source experiments/slurm/env.sh
cdp_module_purge
cdp_activate

read -ra SEED_LIST <<< "${SEEDS:-121 122 123}"
read -ra DATA_SEED_LIST <<< "${DATA_SEEDS:-0 1 2 3 4 5 6 7}"
TASK=${SLURM_ARRAY_TASK_ID:-0}
N_DATA=${#DATA_SEED_LIST[@]}
if (( TASK >= ${#SEED_LIST[@]} * N_DATA )); then
  echo "error: task $TASK is outside the ${#SEED_LIST[@]} x $N_DATA grid; fix --array" >&2
  exit 1
fi
SEED=${SEED_LIST[$(( TASK / N_DATA ))]}
DATA_SEED=${DATA_SEED_LIST[$(( TASK % N_DATA ))]}

echo "script: $SCRIPT  seed: $SEED  data_seed: $DATA_SEED  t: ${T_LIST:-script default}"
echo "host: $(hostname)  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

# The experiment scripts set CUDA_VISIBLE_DEVICES from their first argument: pass the GPU Slurm
# allocated. bash -e makes a failing python run fail the task instead of being reported COMPLETED.
bash -e "$SCRIPT" "${CUDA_VISIBLE_DEVICES:-0}" "$SEED" "$DATA_SEED" ${T_LIST:+"$T_LIST"}
