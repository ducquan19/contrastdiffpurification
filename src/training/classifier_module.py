"""Lightning classifier (WRN-28-10 / WRN-70-16 / ResNet-18) for CIFAR-100 and GTSRB."""

import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim

from src.models.classifiers import build_dm_wide_resnet, build_resnet18

# backbone name -> (depth, width) of the DMWideResNet
_WIDE_RESNETS = {"wrideresnet-28-10": (28, 10), "wrideresnet-70-16": (70, 16)}


class DMClassifier(pl.LightningModule):
    """Cross-entropy classifier with a RobustBench backbone.

    ``backbone`` is ``wrideresnet-28-10``, ``wrideresnet-70-16`` or ``resnet18``; ``optim`` is
    ``adam`` (lr 2e-4) or ``sgd`` (lr 5e-4) and is only needed for training.
    """

    def __init__(self, backbone, num_classes=100, optim=""):
        super(DMClassifier, self).__init__()
        if backbone in _WIDE_RESNETS:
            depth, width = _WIDE_RESNETS[backbone]
            self.resnet = build_dm_wide_resnet(num_classes, depth=depth, width=width)
        elif backbone == "resnet18":
            self.resnet = build_resnet18(num_classes)
        else:
            raise NotImplementedError(f"{backbone} is not supported")

        self.resnet.train()
        self.criterion = nn.CrossEntropyLoss()
        self.__optim = optim.lower()

    def forward(self, x):
        return self.resnet(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, labels)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)
        predict_labels = outputs.argmax(dim=1)
        acc = (predict_labels == labels).sum().item() / labels.size(0)
        self.log('test_acc', acc)
        return acc

    def configure_optimizers(self):
        if self.__optim == 'sgd':
            optimizer = optim.SGD(self.trainer.model.parameters(), lr=5e-4)
        elif self.__optim == 'adam':
            optimizer = optim.Adam(self.trainer.model.parameters(), lr=2e-4)
        else:
            raise ValueError(f"unknown optimizer '{self.__optim}'; expected 'sgd' or 'adam'")
        return optimizer
