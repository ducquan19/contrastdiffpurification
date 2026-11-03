#!/bin/bash
# Confidence-routed cascade DDS on CIFAR-10 (docs/cascade.md). Run from the repository root on the login node:
#
#   source /media02/lhbac34/cdp.sh                    # env + experiments/pcp/gpu_job.sh
#   bash experiments/cascade/run_cifar10.sh fetch     # login node: download the classifiers, clean dev accuracy
#   bash experiments/cascade/run_cifar10.sh screen    # 1 GPU job: probabilities on 200 dev images, 3 sigmas
#   bash experiments/cascade/run_cifar10.sh analyze   # login node, CPU: every head / cascade + the gate
#   CLF_ARGS="..." bash experiments/cascade/run_cifar10.sh main onestep|multistep|cascade
#                                                     # CERTIFY 500 test images, N=10k, 8 shards per sigma
#
# 'main' examples (fair comparison with Carlini et al. / Consistency Purification = same ViT-B/16):
#   CLF_ARGS="--classifier_name hf:$VIT --clf_tag vitb16" bash ... main onestep
#   CLF_ARGS="--classifier_name hf:$VIT --clf_tag vitb16" bash ... main multistep
# the head chosen by 'analyze', e.g. a router of the standard WRN and an l2-robust WRN, as a cascade:
#   CLF_ARGS="--route_classifiers robustbench:Wang2023Better_WRN-28-10:L2" TAU=0.95 bash ... main cascade
set -euo pipefail

source experiments/pcp/gpu_job.sh

VIT=aaraki/vit-base-patch16-224-in21k-finetuned-cifar10
CLASSIFIERS="cifar10-wideresnet-28-10,robustbench:Wang2023Better_WRN-28-10:L2,robustbench:Rebuffi2021Fixing_28_10_cutmix_ddpm:L2,hf:$VIT"
TAGS="wrn,wang23l2,rebuffi21l2,vitb16"
SIGMAS="${SIGMAS:-0.25 0.5 1.0}"
STEPS_LIST="${STEPS_LIST:-5,10}"
COMMON="--config cifar10.yml --domain cifar10 --exp ./wadds --resume"
CLF_ARGS="${CLF_ARGS:-}"
TAU="${TAU:-}"

laneA() { GPU_DEP=singleton GPU_NODE=gpu02 gpu_cmd laneA "$@"; }
laneB() { GPU_DEP=singleton GPU_EXCLUDE=gpu02 gpu_cmd laneB "$@"; }

case "${1:-}" in
  fetch)
    # compute nodes may have no internet: download every checkpoint here (pretrained/robustbench, pretrained/hf).
    # The ViT runs on src/models/vit.py (plain PyTorch): transformers is NOT needed.
    python - <<EOF
import torch
from src.models.classifiers import get_image_classifier
from src.data.loaders import load_eval_subset
x, y, _ = load_eval_subset("cifar10", 200, 0, split="dev")
for name in "$CLASSIFIERS".split(","):
    f = get_image_classifier(name).eval()
    with torch.no_grad():
        acc = sum(int((f(x[i:i + 50]).argmax(1) == y[i:i + 50]).sum()) for i in range(0, len(x), 50)) / len(x)
    print(f"{name}: clean accuracy on 200 dev images = {acc:.3f}")
EOF
    ;;

  screen)
    # 200 dev images x 200 draws; onestep + multi-step S=5 and S=10 (eta 0) share the noisy draws;
    # 4 classifiers on every output. ~14 NFE per draw -> ~0.56M NFE per sigma, ~15 min per sigma on A100.
    cmd=""
    for s in $SIGMAS; do
      cmd+="python -m scripts.screen_cascade --sigma $s --classifiers $CLASSIFIERS --tags $TAGS \
--steps_list $STEPS_LIST --eta 0 --eval_split dev --num_sub 200 --data_seed 0 --n_draws 200 --batch 100 \
--exp ./cascade || echo 'SCREEN RUN FAILED: sigma $s'; "
    done
    laneA 3 "$cmd"
    ;;

  analyze)
    python -m scripts.analyze_cascade outputs/cascade/screen_*/
    ;;

  main)
    METHOD="${2:-}"
    case "$METHOD" in
      onestep) MARGS="--method onestep" ;;
      multistep) MARGS="--method multistep --steps 10 --eta 0" ;;
      cascade)
        [ -n "$TAU" ] || { echo "main cascade needs TAU=<threshold chosen by analyze>" >&2; exit 1; }
        MARGS="--method cascade --steps 10 --eta 0 --exit_tau $TAU" ;;
      *) echo "usage: $0 main {onestep|multistep|cascade}" >&2; exit 1 ;;
    esac
    for s in $SIGMAS; do
      for sh in 0 1 2 3 4 5 6 7; do
        run="python -m scripts.certify_dds --mode certify $MARGS --sigma $s --eval_split test --num_sub 500 \
--data_seed 0 --n0 100 --n 10000 --alpha 0.001 --batch 500 --shard $sh/8 $COMMON $CLF_ARGS"
        if (( sh < 6 )); then laneA 8 "$run"; else laneB 8 "$run"; fi
      done
    done
    ;;

  *)
    echo "usage: $0 {fetch|screen|analyze|main onestep|multistep|cascade}" >&2
    exit 1
    ;;
esac
