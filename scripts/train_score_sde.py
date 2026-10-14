"""Train (or sample from) a score-SDE NCSN++ model on CIFAR-100 or GTSRB.

Run from the repository root:

    python -m scripts.train_score_sde --mode train --dataset gtsrb --config_path configs/gtsrb.yml ...
"""
import argparse
import logging
import os

import pytorch_lightning as pl
import torch
import torchvision.utils
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.specs import DATASET_NAMES, get_dataset_spec
from src.paths import config_path, output_path
from src.config import load_config
from src.training.score_sde_module import ScoreSDETrain, get_data_scaler
from src.utils import setup_logging

logger = logging.getLogger(__name__)


def main():
    _args = argparse.ArgumentParser()
    _args.add_argument(
        "--mode",
        type=str,
        default="train",
        help="train, test, generate"
    )
    _args.add_argument(
        "--checkpoint",
        type=str,
        default=output_path(
            "training",
            "cifar100_ncsnpp_scoresde_model.ckpt"
        ),
        help="checkpoint path"
    )
    _args.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="checkpoint path for continuous train"
    )
    _args.add_argument(
        "--generate_batch_size",
        type=int,
        default=64,
        help="batch size for generate"
    )
    _args.add_argument(
        "--generate_file",
        type=str,
        default=output_path(
            "generate",
            "samples.png"
        ),
        help="directory to save generate samples"
    )
    _args.add_argument(
        "--config_path",
        type=str,
        default=config_path("cifar100.yml"),
        help="config file path"
    )
    _args.add_argument(
        "--ckpt_root",
        type=str,
        default=None,
        help="directory to save checkpoints"
    )
    _args.add_argument(
        "--dataset",
        type=str,
        default="cifar100",
        choices=DATASET_NAMES,
        help="dataset to train on"
    )
    _args.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="root directory of the dataset (default: its folder under the data root)"
    )
    _args.add_argument(
        "--train_max_epochs",
        type=int,
        default=1600,
        help="max epochs for train"
    )
    args = _args.parse_args()
    setup_logging()
    main_config = load_config(args.config_path)

    if args.mode == "generate":
        logger.info("generating samples with checkpoint %s", args.checkpoint)

        try:
            model = ScoreSDETrain.load_from_checkpoint(args.checkpoint, config=main_config).cuda()
        except Exception:
            model = ScoreSDETrain(main_config)
            model.model.load_state_dict(
                torch.load(
                    args.checkpoint,
                    map_location=torch.device('cpu')
                )
            )
            model = model.cuda()

        model.eval()

        samples = model.generate_sample(args.generate_batch_size, model.get_sample_func())

        # save with torch vision
        torchvision.utils.save_image(samples, args.generate_file)
        logger.info("samples saved to %s", args.generate_file)

    elif args.mode == "train":

        _transfrom_pip = [
            transforms.Resize((
                main_config.data.image_size,
                main_config.data.image_size
            )),
            transforms.ToTensor(),
        ]

        if main_config.data.centered:
            _transfrom_pip.append(get_data_scaler(True))

        dataset = get_dataset_spec(args.dataset).build_split(
            train=True,
            transform=transforms.Compose(_transfrom_pip),
            root=args.data_root,
        )

        train_dataloader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=True,
            num_workers=4,
        )
        if args.ckpt_root is not None:
            os.makedirs(args.ckpt_root, exist_ok=True)

        # configure train
        trainer = pl.Trainer(
            max_epochs=args.train_max_epochs,
            enable_checkpointing=True,
            enable_progress_bar=True,
            log_every_n_steps=1,
            default_root_dir=args.ckpt_root,
        )

        model = ScoreSDETrain(main_config)
        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            ckpt_path=args.resume_checkpoint
        )

        trainer.save_checkpoint(args.checkpoint)
    else:
        raise ValueError(f"mode ({args.mode}) must be 'train', 'test' or 'generate'")


if __name__ == "__main__":
    main()
