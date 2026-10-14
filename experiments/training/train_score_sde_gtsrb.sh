#! /bin/bash
# Train the GTSRB score-SDE (NCSN++) diffusion model.
# Afterwards: python -m scripts.convert_lightning_checkpoint --ckpt_path outputs/training/gtsrb_ncsnpp_score_sde.ckpt \
#                 -x model --out_path pretrained/gtsrb/gtsrb_score_sde.pt
cd "$(dirname "$0")/../.." || exit 1  # run from the repository root

# NOTE: --resume_checkpoint points at the final checkpoint, as in the original script; remove it
# for a first run (Lightning fails if the file does not exist).
python -m scripts.train_score_sde \
    --mode "train" \
    --ckpt_root "./outputs/training/gtsrb_ncsnpp" \
    --dataset "gtsrb" \
    --data_root "./data/gtsrb" \
    --checkpoint "./outputs/training/gtsrb_ncsnpp_score_sde.ckpt" \
    --config_path "./configs/gtsrb.yml" \
    --train_max_epochs 1800 \
    --resume_checkpoint "./outputs/training/gtsrb_ncsnpp_score_sde.ckpt" \
