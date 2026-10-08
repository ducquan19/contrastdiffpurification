#! /bin/bash
# Train classifiers used by the CIFAR-100 / GTSRB experiments (not needed for CIFAR-10).
# Afterwards: python -m scripts.convert_lightning_checkpoint --ckpt_path <ckpt> -x resnet \
#                 --out_path pretrained/cifar100/cifar100_wrn_28-10_standard.pt
cd "$(dirname "$0")/../.." || exit 1  # run from the repository root

# python -m scripts.train_classifier \
#     --backbone wrideresnet-28-10 \
#     --data_root "./data/gtsrb"\
#     --ckpt_root "./outputs/training/gtsrb_wrn-28-10_checkpoint" \
#     --dataset "gtsrb" \
#     --train_max_epochs 256 \
#     --checkpoint "./outputs/training/gtsrb_model_final_wrn-28-10.ckpt" \

# python -m scripts.train_classifier \
#     --backbone wrideresnet-70-16 \
#     --data_root "./data/cifar100"\
#     --ckpt_root "./outputs/training/cifar100-70-16_checkpoint" \
#     --dataset "cifar100" \
#     --train_max_epochs 256 \
#     --checkpoint "./outputs/training/cifar100_model_final_wrn-70-16.ckpt" \
#     --train_batch_size 64 \
#     --val_batch_size 64 \

# The original script resumed from a machine-specific checkpoint
# (--resume_ckpt "./cifar100-28-10_checkpoint/model-epoch=168-val_loss=1.74.ckpt"); the default
# "last" resumes from the newest checkpoint in --ckpt_root, if any.
python -m scripts.train_classifier \
    --backbone "wrideresnet-28-10" \
    --data_root "./data/cifar100"\
    --ckpt_root "./outputs/training/cifar100-28-10_checkpoint" \
    --dataset "cifar100" \
    --train_max_epochs 1024 \
    --checkpoint "./outputs/training/cifar100_model_final_wrn-28-10.ckpt" \
    --train_batch_size 64 \
    --val_batch_size 64 \

# python -m scripts.train_classifier \
#     --backbone resnet18 \
#     --data_root "./data/gtsrb"\
#     --ckpt_root "./outputs/training/gtsrb_resnet18_checkpoint" \
#     --dataset "gtsrb" \
#     --train_max_epochs 128 \
#     --checkpoint "./outputs/training/gtsrb_model_final_resnet18.ckpt"
