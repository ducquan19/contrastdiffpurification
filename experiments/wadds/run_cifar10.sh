#!/bin/bash
# WA-DDS on CIFAR-10 (docs/wadds.md). Run from the repository root on the login node:
#
#   source /media02/lhbac34/cdp.sh                  # env + experiments/pcp/gpu_job.sh
#   bash experiments/wadds/run_cifar10.sh stats     # CPU, ~1 min, on the login node
#   bash experiments/wadds/run_cifar10.sh g0        # gate G0: band diagnosis on 200 dev images, 1 job
#   bash experiments/wadds/run_cifar10.sh dev       # gate G1: CERTIFY WA-DDS on 100 dev images, 1 job
#   bash experiments/wadds/run_cifar10.sh tune      # G0 grid around the default, sigma 0.5 / 1.0, 1 job
#   bash experiments/wadds/run_cifar10.sh devbase   # onestep + multistep CERTIFY on the same 100 dev images, 1 job
#   bash experiments/wadds/run_cifar10.sh main [wadds|multistep|onestep]   # 500 test images, N=10k, 24 jobs
#   bash experiments/wadds/run_cifar10.sh summary   # (re)compute every summary.json, no GPU
#
# WA-DDS settings of 'dev' and 'main' come from WADDS_ARGS (default "--steps 10 --eta 0 --gamma 1.0");
# after 'tune', export the chosen ones, e.g. WADDS_ARGS="--steps 10 --eta 0 --gamma 1.0".
#
# Run 'main' only after G0 / G1 pass. Only WA-DDS is certified here; G0 also reports the one-shot
# and multi-step variants because they come for free from the same noisy samples.
# Baselines (onestep / multistep) for the main table: same commands with --method onestep|multistep.
#
# Measured (Oct 26): peak GPU memory 2.3 GiB at --batch 125, ~860 NFE/s on gpu02 (A100), ~330 NFE/s on gpu04 (V100).
# 'main' therefore uses --batch 500 and 8 shards of ~63 images (~2 h on A100, ~5.5 h on V100).
# Two lanes (one running job each, one job per node; see run-guide): laneA on gpu02, laneB elsewhere.
set -euo pipefail

source experiments/pcp/gpu_job.sh

EXP=./wadds
COMMON="--config cifar10.yml --domain cifar10 --classifier_name cifar10-wideresnet-28-10 --exp $EXP --resume"
SIGMAS="${SIGMAS:-0.25 0.5 1.0}"   # e.g. SIGMAS="0.5 1.0" bash ... main
S=10
# Tune (27/10): eta 0 beats eta 1 at sigma 0.5 and 1.0 (ACR CI excludes 0); gamma 0.5, partial
# windows: worse; pooled: same; S 20: n.s. -> default eta 0. MS_ARGS: the multi-step baseline at the same settings.
WADDS_ARGS="${WADDS_ARGS:---steps 10 --eta 0 --gamma 1.0}"
MS_ARGS="${MS_ARGS:---steps 10 --eta 0}"

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
      cmd+="python -m scripts.certify_dds --mode certify --method wadds --sigma $s $WADDS_ARGS \
--eval_split dev --num_sub 100 --data_seed 0 --n0 100 --n 1000 --alpha 0.001 --batch 125 $COMMON; "
    done
    laneB 4 "$cmd"
    ;;

  tune)
    # G0 grid (200 dev images, 100 copies) around the default, one factor at a time; ~7 min per run
    # with S=10 on A100. The default itself is the existing g0 run. Compare with compare_dds diag.
    cmd=""
    for s in 0.5 1.0; do
      for cfg in "--steps 10 --eta 1.0 --gamma 0.5" "--steps 10 --eta 0 --gamma 1.0" \
                 "--steps 10 --eta 1.0 --gamma 1.0 --anchor_t_min 0.5" \
                 "--steps 10 --eta 1.0 --gamma 1.0 --anchor_t_max 0.5" \
                 "--steps 10 --eta 1.0 --gamma 1.0 --power_mode pooled" "--steps 20 --eta 1.0 --gamma 1.0"; do
        cmd+="python -m scripts.certify_dds --mode diagnose --sigma $s $cfg --eval_split dev --num_sub 200 \
--data_seed 0 --n_diag 100 --batch 125 $COMMON; "
      done
    done
    laneA 4 "$cmd"
    ;;

  devbase)
    # baselines of gate G1 on the same 100 dev images and N as 'dev': onestep (1 NFE) and multistep (S NFE)
    cmd=""
    for s in $SIGMAS; do
      cmd+="python -m scripts.certify_dds --mode certify --method onestep --sigma $s --eval_split dev \
--num_sub 100 --data_seed 0 --n0 100 --n 1000 --alpha 0.001 --batch 125 $COMMON; "
      cmd+="python -m scripts.certify_dds --mode certify --method multistep --sigma $s --steps $S --eta 1.0 \
--eval_split dev --num_sub 100 --data_seed 0 --n0 100 --n 1000 --alpha 0.001 --batch 125 $COMMON; "
    done
    laneB 5 "$cmd"
    ;;

  main)
    # 500 test images (the first 500 of the reported subset, data_seed 0), N0=100, N=10000,
    # 8 shards per sigma. METHOD = wadds (default), multistep or onestep (baselines, same settings).
    METHOD="${2:-wadds}"
    case "$METHOD" in
      wadds) MARGS="--method wadds $WADDS_ARGS" ;;
      multistep) MARGS="--method multistep $MS_ARGS" ;;
      onestep) MARGS="--method onestep" ;;
      *) echo "unknown method $METHOD" >&2; exit 1 ;;
    esac
    for s in $SIGMAS; do
      for sh in 0 1 2 3 4 5 6 7; do
        run="python -m scripts.certify_dds --mode certify $MARGS --sigma $s --eval_split test --num_sub 500 \
--data_seed 0 --n0 100 --n 10000 --alpha 0.001 --batch 500 --shard $sh/8 $COMMON"
        # A100 (laneA, ~2 h/shard) takes shards 0-5, the slower lane (V100 ~5.3 h/shard) shards 6-7
        if (( sh < 6 )); then laneA 8 "$run"; else laneB 8 "$run"; fi
      done
    done
    ;;

  summary)
    # recompute summary.json of every run folder (also partial / sharded runs), no GPU
    python -m scripts.certify_dds --summarize_dir outputs/wadds/*/
    ;;

  *)
    echo "usage: $0 {stats|g0|dev|tune|devbase|main [method]|summary}" >&2
    exit 1
    ;;
esac
