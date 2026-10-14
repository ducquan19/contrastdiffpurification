#!/usr/bin/env bash
# Throughput of the score network and GPU-day estimates of the full-gradient evaluations.
# Usage: bash experiments/pcp/calibrate.sh <gpu>
cd "$(dirname "$0")/../.." || exit 1
CUDA_VISIBLE_DEVICES=$1 python -m scripts.calibrate_throughput --config cifar10.yml --batch_sizes 64,128,256