# The original license:
# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DiffPure. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
"""Image classifiers attacked and protected in the experiments, selected by name.

Every returned model takes images in [0, 1] and normalizes internally.
"""

import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import robustbench.model_zoo.architectures.dm_wide_resnet as dm_wide_resnet
import robustbench.model_zoo.architectures.resnet as resnet

from robustbench import load_model

from src.models.classifiers.cifar10_resnet import resnet50, wide_resnet_70_16_dropout
from src.paths import pretrained_path

# RobustBench downloads its checkpoints here (instead of ./models relative to the cwd).
ROBUSTBENCH_MODEL_DIR = pretrained_path('robustbench')

logger = logging.getLogger(__name__)


class ImageNetNormalized(nn.Module):
    """Applies the ImageNet mean/std before ``model``.

    scripts/train_classifier.py normalizes its inputs this way, so the CIFAR-100 and GTSRB models
    trained there need it to accept [0, 1] images.
    """

    def __init__(self, model):
        super().__init__()
        self.resnet = model
        self.mu = torch.Tensor([0.485, 0.456, 0.406]).float().view(3, 1, 1)
        self.sigma = torch.Tensor([0.229, 0.224, 0.225]).float().view(3, 1, 1)

    def forward(self, x):
        x = (x - self.mu.to(x.device)) / self.sigma.to(x.device)
        return self.resnet(x)


def build_dm_wide_resnet(num_classes, depth=28, width=10):
    """RobustBench ``DMWideResNet`` with Swish; CIFAR-10 statistics for 10 classes, else CIFAR-100's."""
    if num_classes == 10:
        mean, std = dm_wide_resnet.CIFAR10_MEAN, dm_wide_resnet.CIFAR10_STD
    else:
        mean, std = dm_wide_resnet.CIFAR100_MEAN, dm_wide_resnet.CIFAR100_STD
    return dm_wide_resnet.DMWideResNet(
        num_classes=num_classes,
        depth=depth,
        width=width,
        activation_fn=dm_wide_resnet.Swish,
        mean=mean,
        std=std,
    )


def build_resnet18(num_classes):
    """RobustBench ResNet-18 (no internal normalization)."""
    return resnet.ResNet(block=resnet.BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=num_classes)


def update_state_dict(state_dict, idx_start=9):
    """Strip the first ``idx_start`` characters of every key (e.g. a ``module.`` DataParallel prefix)."""
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k[idx_start:]] = v
    return new_state_dict


def _load_weights(model, relative_path, strip_prefix=0):
    """Load ``pretrained/<relative_path>`` into ``model`` and return it in eval mode."""
    model_path = pretrained_path(relative_path)
    logger.info("loading checkpoint %s", model_path)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    if strip_prefix:
        state_dict = update_state_dict(state_dict, idx_start=strip_prefix)
    model.load_state_dict(state_dict)
    return model.eval()


def get_image_classifier(classifier_name: str):
    """Build the classifier ``<dataset>-<architecture>`` with its pretrained weights loaded.

    Supported names: ``cifar10-{wideresnet-28-10, wrn-70-16-dropout, resnet-50}``,
    ``cifar100-wideresnet-28-10``, ``gtsrb-{wrideresnet-28-10, resnet18}`` (the ``wrideresnet``
    spelling is part of the GTSRB name). Matching is by substring.
    """
    logger.info("classifier: %s", classifier_name)

    # robustbench:<model_name>[:<threat_model>[:<dataset>]] -- any RobustBench model, e.g. an
    # adversarially trained one (the robust PCP encoder evaluated on its own)
    if classifier_name.startswith('robustbench:'):
        parts = classifier_name.split(':')
        threat_model = parts[2] if len(parts) > 2 else 'Linf'
        dataset = parts[3] if len(parts) > 3 else 'cifar10'
        return load_model(model_dir=ROBUSTBENCH_MODEL_DIR, model_name=parts[1], dataset=dataset,
                          threat_model=threat_model).eval()

    # 'cifar100' must be checked before 'cifar10', which is a substring of it
    if 'cifar100' in classifier_name:
        if 'wideresnet-28-10' in classifier_name:
            model = _load_weights(build_dm_wide_resnet(100), 'cifar100/cifar100_wrn_28-10_standard.pt')
            return ImageNetNormalized(model)

    elif 'cifar10' in classifier_name:
        if 'wideresnet-28-10' in classifier_name:
            return load_model(model_dir=ROBUSTBENCH_MODEL_DIR, model_name='Standard',
                              dataset='cifar10', threat_model='Linf')
        if 'resnet-50' in classifier_name:
            return _load_weights(resnet50(), 'cifar10/resnet-50/weights.pt', strip_prefix=7)
        if 'wrn-70-16-dropout' in classifier_name:
            return _load_weights(wide_resnet_70_16_dropout(), 'cifar10/wrn-70-16-dropout/weights.pt',
                                 strip_prefix=7)

    elif 'gtsrb' in classifier_name:
        if 'wrideresnet-28-10' in classifier_name:
            model = _load_weights(build_dm_wide_resnet(43), 'gtsrb/gtsrb_wrn_28-10_standard.pt')
            return ImageNetNormalized(model).eval()
        if 'resnet18' in classifier_name:
            model = _load_weights(build_resnet18(43), 'gtsrb/gtsrb_resnet18_standard.pt')
            return ImageNetNormalized(model).eval()

    raise NotImplementedError(f'unknown classifier {classifier_name}')
