#!/usr/bin/env bash
# Build the PCP prototype bank (training images only) at the path set in configs/method/pcp.yaml.
# Usage: bash experiments/pcp/build_bank.sh <gpu>
cd "$(dirname "$0")/../.." || exit 1
CUDA_VISIBLE_DEVICES=$1 python -m scripts.build_prototype_bank --config cifar10.yml --domain cifar10 \
  --n_prototypes 500 --t_values 0,0.025,0.05,0.075,0.1 --n_images 20000 --seed 0