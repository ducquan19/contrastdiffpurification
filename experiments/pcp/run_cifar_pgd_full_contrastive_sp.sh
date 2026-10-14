#!/usr/bin/env bash
# Full-gradient PGD+EOT (Linf 8/255), CIFAR-10 / WRN-28-10, differentiable particle solver.
# ContrastDiffPure (published hyperparameters, configs/method/contrastive.yaml), K = 1, with the
# guidance visible to the attacker and the cached-view leak fixed.
# Usage: bash experiments/pcp/run_cifar_pgd_full_contrastive_sp.sh <gpu> <seed> <data_seed> [shard i/n]
cd "$(dirname "$0")/../.." || exit 1  # run from the repository root
cuda_index=$1
SEED1=$2
SEED2=$3
SHARD=${4:-${SHARD:-0/1}}   # 4th argument, else the SHARD environment variable (Slurm)

for seed in $SEED1; do
  for data_seed in $SEED2; do
        CUDA_VISIBLE_DEVICES=$cuda_index python -m scripts.eval_robustness --guidance contrastive \
          --exp ./exp_results_pcp -i cifar10-pgdfull-linf-contrastive_sp \
          --config cifar10.yml --domain cifar10 --classifier_name cifar10-wideresnet-28-10 \
          --diffusion_type particle --t 100 \
          --attack_version pgd_full --lp_norm Linf --adv_eps 0.031373 \
          --pgd_steps 200 --pgd_eot_iter 4 --pgd_loss ce \
          --num_sub 512 --adv_batch_size 32 \
          --seed $seed --data_seed $data_seed --shard $SHARD --resume \
          --pcp_n_particles 1 \
          --contrastive_classifier cifar10-wideresnet-28-10 \
          --contrastive_differentiable true --contrastive_fix_cache true

  done
done
