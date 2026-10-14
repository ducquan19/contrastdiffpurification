# Helper for one-off GPU runs on Slurm (pilot, tuning, extra tables). Source it from the repo root:
#
#   source experiments/pcp/gpu_job.sh
#   gpu_job <job name> <hours> <arguments of scripts.eval_robustness ...>
#   gpu_cmd <job name> <hours> <any shell command, e.g. python -m scripts.build_prototype_bank ...>
#
# Each call submits one job (1 GPU, 4 CPUs, 32 GB) that activates the environment of
# experiments/slurm/env.sh and writes its log to logs/slurm/<name>-<jobid>.out.
# Arguments must not contain spaces.

gpu_cmd() {
  local name=$1 hours=$2
  shift 2
  mkdir -p logs/slurm
  sbatch --job-name="$name" --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=4 --mem=32G \
    --time="${hours}:00:00" --output="logs/slurm/%x-%j.out" \
    --wrap "set -euo pipefail; source experiments/slurm/env.sh; cdp_module_purge; cdp_activate; $*"
}

gpu_job() {
  local name=$1 hours=$2
  shift 2
  gpu_cmd "$name" "$hours" python -m scripts.eval_robustness "$@"
}
