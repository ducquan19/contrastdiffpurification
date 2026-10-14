"""Train or test a classifier on CIFAR-100 or GTSRB (WRN-28-10 / WRN-70-16 / ResNet-18).

Run from the repository root:

    python -m scripts.train_classifier --backbone wrideresnet-28-10 --dataset cifar100 ...
"""
import argparse
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.specs import DATASET_NAMES, get_dataset_spec
from src.paths import output_path
from src.training.classifier_module import DMClassifier


def build_transform(train: bool):
    """Resize to 32x32, (random flip when training), ImageNet normalization."""
    steps = [transforms.Resize(size=(32, 32))]
    if train:
        steps.append(transforms.RandomHorizontalFlip())
    steps += [
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ]
    return transforms.Compose(steps)


def main():
    _args = argparse.ArgumentParser()
    _args.add_argument(
        "--backbone",
        type=str,
        default="wrideresnet-28-10"
    )
    _args.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="root directory of the dataset (default: its folder under the data root)"
    )
    _args.add_argument(
        "--checkpoint",
        type=str,
        default=output_path(
            "training",
            "cifar100_model_final_{backbone}.ckpt"
        ),
        help="The saved checkpoint"
    )
    _args.add_argument(
        "--ckpt_root",
        type=str,
        default=output_path(
            "training",
            "cifar100_checkpoints"
        ),
        help="the root path to save checkpoints"
    )
    _args.add_argument(
        "--resume_ckpt",
        type=str,
        default="last",
        help="the checkpoint path to resume training"
    )
    _args.add_argument(
        "--optim",
        type=str,
        default="adam",
        help="the optimizer"
    )
    _args.add_argument(
        "--dataset",
        type=str,
        default="cifar100",
        choices=DATASET_NAMES,
        help="dataset to train on"
    )
    _args.add_argument(
        "--test_mode",
        action="store_true"
    )
    _args.add_argument(
        "--train_max_epochs",
        type=int,
        default=512,
        help="max epochs for train"
    )
    _args.add_argument(
        "--train_batch_size",
        type=int,
        default=1024,
        help="batch size for train"
    )
    _args.add_argument(
        "--val_batch_size",
        type=int,
        default=1024,
        help="batch size for val"
    )
    args = _args.parse_args()
    args.checkpoint = args.checkpoint.format(backbone=args.backbone)

    print(args)

    # test mode evaluates without augmentation; training validates with the training transform
    spec = get_dataset_spec(args.dataset)
    transform = build_transform(train=not args.test_mode)
    if not args.test_mode:
        train_dataset = spec.build_split(
            train=True,
            transform=transform,
            root=args.data_root
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=8
        )

    num_classes = spec.num_classes
    val_dataset = spec.build_split(
        train=False,
        transform=transform,
        root=args.data_root
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=8
    )

    os.makedirs(args.ckpt_root, exist_ok=True)

    trainer = pl.Trainer(
        max_epochs=args.train_max_epochs,
        # enable_checkpointing=False,
        enable_progress_bar=True,
        log_every_n_steps=1,
        default_root_dir=args.ckpt_root,
        callbacks=[
            ModelCheckpoint(
                monitor='val_loss',
                dirpath=args.ckpt_root,
                filename='model-{epoch:02d}-{val_loss:.2f}',
                save_top_k=3,
                mode='min'
            ),
            ModelCheckpoint(
                dirpath=args.ckpt_root,
                filename='model-latest',
            ),
        ],
    )

    if args.test_mode:
        try:
            model = DMClassifier.load_from_checkpoint(
                args.checkpoint,
                backbone=args.backbone,
                num_classes=num_classes,
            )
        except Exception:
            model = DMClassifier(args.backbone, num_classes=num_classes)
            model.resnet.load_state_dict(torch.load(args.checkpoint))

        trainer.test(model, val_loader)
    else:
        model = DMClassifier(
            args.backbone,
            num_classes=num_classes,
            optim=args.optim
        )
        trainer.fit(
            model,
            train_loader,
            val_loader,
            ckpt_path=args.resume_ckpt
        )
        trainer.save_checkpoint(args.checkpoint)


if __name__ == "__main__":
    main()
