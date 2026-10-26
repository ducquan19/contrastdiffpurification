#!/bin/bash
# WA-DDS on CIFAR-10 (docs/wadds.md). Run from the repository root on the login node:
#
#   source /media02/lhbac34/cdp.sh                  # env + experiments/pcp/gpu_job.sh
#   bash experiments/wadds/run_cifar10.sh stats     # CPU, ~1 min, on the login node
#   bash experiments/wadds/run_cifar10.sh g0        # gate G0: band diagnosis on 200 dev images, 1 job
#   bash experiments/wadds/run_cifar10.sh dev       # gate G1: CERTIFY WA-DDS on 100 dev images, 1 job
#   bash experiments/wadds/run_cifar10.sh main      # main table: 500 test images, N=10k, 12 jobs
#   bash experiments/wadds/run_cifar10.sh summary   # (re)compute every summary.json, no GPU
#
# Run 'main' only after G0 / G1 pass. Only WA-DDS is certified here; G0 also reports the one-shot
# and multi-step variants because they come for free from the same noisy samples.
# Baselines (onestep / multistep) for the main table: same commands with --method onestep|multistep.
#
# --batch 125: the score net peaked at 33 GiB for batch 256 (calibration), too much for gpu04 (V100-32GB).
# Two lanes (one running job each, one job per node; see run-guide): laneA on gpu02, laneB elsewhere.
set -euo pipefail

source experiments/pcp/gpu_job.sh

EXP=./wadds
COMMON="--config cifar10.yml --domain cifar10 --classifier_name cifar10-wideresnet-28-10 --exp $EXP --resume"
SIGMAS="0.25 0.5 1.0"
S=10

laneA() { GPU_DEP=singleton GPU_NODE=gpu02 gpu_cmd laneA "$@"; }
laneB() { GPU_DEP=singleton GPU_EXCLUDE=gpu02 gpu_cmd laneB "$@"; }

case "${1:-}" in
  stats)
    python -m scripts.spectral_stats --domain cifar10
    ;;

  g0)
    # 200 dev images x 100 noisy copies x 3 sigmas; 19 NFE per copy -> ~1.1M NFE, well under 1 h on A100
    cmd=""
    for s in $SIGMAS; do
      cmd+="python -m scripts.certify_dds --mode diagnose --sigma $s --steps $S --eta 1.0 --gamma 1.0 \
--eval_split dev --num_sub 200 --data_seed 0 --n_diag 100 --batch 125 $COMMON; "
    done
    laneA 4 "$cmd"
    ;;

  dev)
    # CERTIFY WA-DDS, 100 dev images, N0=100, N=1000; 100 x 1100 x 10 NFE = 1.1M NFE per sigma
    cmd=""
    for s in $SIGMAS; do
      cmd+="python -m scripts.certify_dds --mode certify --method wadds --sigma $s --steps $S --eta 1.0 \
--gamma 1.0 --eval_split dev --num_sub 100 --data_seed 0 --n0 100 --n 1000 --alpha 0.001 --batch 125 $COMMON; "
    done
    laneB 4 "$cmd"
    ;;

  main)
    # 500 test images (the first 500 of the reported subset, data_seed 0), N0=100, N=10000.
    # 500 x 10100 x 10 NFE ~ 50M NFE per sigma (~14 GPU-h on A100) -> 4 shards per sigma.
    i=0
    for s in $SIGMAS; do
      for sh in 0 1 2 3; do
        run="python -m scripts.certify_dds --mode certify --method wadds --sigma $s --steps $S --eta 1.0 \
--gamma 1.0 --eval_split test --num_sub 500 --data_seed 0 --n0 100 --n 10000 --alpha 0.001 --batch 125 \
--shard $sh/4 $COMMON"
        if (( i % 2 == 0 )); then laneA 10 "$run"; else laneB 10 "$run"; fi
        i=$((i + 1))
      done
    done
    ;;

  summary)
    # recompute summary.json of every run folder (also partial / sharded runs), no GPU
    python -m scripts.certify_dds --summarize_dir outputs/wadds/*/
    ;;

  *)
    echo "usage: $0 {stats|g0|dev|main|summary}" >&2
    exit 1
    ;;
esac
