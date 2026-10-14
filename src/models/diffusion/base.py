# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DiffPure. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
"""Parts shared by the SDE and ODE purifiers.

* ``load_score_model`` -- the dataset's pretrained NCSN++ score network;
* ``VPDynamics`` -- the VP-SDE noise schedule and the score of flattened images;
* ``DiffusionPurifier`` -- diffuse the input to t* and integrate it back; subclasses supply the
  reverse-time dynamics and the solver.
"""

import logging
import os
import random

import numpy as np
import torch
import torchvision.utils as tvu

from score_sde import sde_lib
from score_sde.losses import get_optimizer
from score_sde.models import utils as mutils
from score_sde.models.ema import ExponentialMovingAverage

from src.paths import pretrained_path

logger = logging.getLogger(__name__)


def restore_checkpoint(ckpt_dir, state, device):
    """Load a score_sde training checkpoint into ``state`` (optimizer, model, EMA and step)."""
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']


def load_score_model(config, device):
    """Build the score network of ``config`` and load the pretrained weights of its dataset.

    CIFAR-10 uses the official score_sde checkpoint (EMA weights); CIFAR-100 and GTSRB use a
    plain ``state_dict`` produced by scripts/train_score_sde.py.
    """
    dataset = config.data.dataset.lower()
    logger.debug('score model config: %s', config)
    model = mutils.create_model(config)

    if dataset == 'cifar10':
        optimizer = get_optimizer(config, model.parameters())
        ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
        state = dict(step=0, optimizer=optimizer, model=model, ema=ema)
        restore_checkpoint(pretrained_path('score_sde', 'checkpoint_8.pth'), state, device)
        ema.copy_to(model.parameters())
    elif dataset in ('cifar100', 'gtsrb'):
        state_dict = torch.load(pretrained_path(dataset, f'{dataset}_score_sde.pt'), map_location='cpu')
        model.load_state_dict(state_dict)
    else:
        raise NotImplementedError(f'Unknown dataset {config.data.dataset}!')

    return model.eval().to(device)


class VPDynamics(torch.nn.Module):
    """Variance-preserving SDE ``dx = -0.5 beta(t) x dt + sqrt(beta(t)) dw`` around a score network.

    Works on flattened images of shape (B, C*H*W), the layout torchsde / torchdiffeq expect.

    Args:
        model: the score_sde (NCSN++) eps-prediction network; replaced in place when contrastive
               guidance is attached.
        beta_min / beta_max: beta(0) and beta(1).
        N: number of discrete steps of the schedule.
        img_shape: (C, H, W) of one image.
    """

    def __init__(self, model, beta_min=0.1, beta_max=20, N=1000,
                 img_shape=(3, 32, 32)):
        super().__init__()
        self.model = model
        self.img_shape = img_shape

        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)

    def vpsde_fn(self, t, x):
        """Forward-SDE drift ``-0.5 beta(t) x`` and diffusion ``sqrt(beta(t))``."""
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def score(self, t, x):
        """Score ``grad_x log p_t(x)`` of flattened ``x``, returned with the same shape."""
        assert x.ndim == 2 and np.prod(self.img_shape) == x.shape[1], x.shape
        x_img = x.view(-1, *self.img_shape)

        # the network predicts epsilon; get_score_fn turns it into a score
        sde = sde_lib.VPSDE(beta_min=self.beta_0, beta_max=self.beta_1, N=self.N)
        score_fn = mutils.get_score_fn(sde, self.model, train=False, continuous=True)
        score = score_fn(x_img, t)
        assert x_img.shape == score.shape, f'{x_img.shape}, {score.shape}'
        return score.view(x.shape[0], -1)


class DiffusionPurifier(torch.nn.Module):
    """Purify images by diffusing them to t* = ``args.t`` and integrating the reverse process.

    Subclasses implement ``_build_dynamics`` and ``_integrate``.
    """

    def __init__(self, args, config, device=None):
        super().__init__()
        self.args = args
        self.config = config
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        img_shape = (config.data.num_channels, config.data.image_size, config.data.image_size)
        self.model = load_score_model(config, self.device)
        self.dynamics = self._build_dynamics(self.model, img_shape).to(self.device)
        self.betas = self.dynamics.discrete_betas.float().to(self.device)

    def _build_dynamics(self, model, img_shape) -> VPDynamics:
        raise NotImplementedError

    def _integrate(self, x):
        """Integrate the noised batch ``x`` (B, C, H, W) from t* back to t ~ 0."""
        raise NotImplementedError

    def set_score_model(self, model):
        """Replace the score network, e.g. by its contrastive-guided wrapper."""
        self.model = model
        self.dynamics.model = model

    def _initial_noise(self, x0):
        """Gaussian noise for the forward diffusion; with ``--fix_rand`` one fixed sample per batch."""
        if self.args.fix_rand:
            noise_fixed = torch.FloatTensor(1, *x0.shape[1:]).\
                normal_(0, 1, generator=torch.manual_seed(self.args.seed)).to(self.device)
            logger.debug('fixed noise (first values): %s', noise_fixed[0, 0, 0, :3])
            return noise_fixed.repeat(x0.shape[0], 1, 1, 1)
        return torch.randn_like(x0).to(self.device)

    def image_editing_sample(self, img, bs_id=0, tag=None):
        """Purify ``img`` (in [-1, 1], NCHW) and return the result, also in [-1, 1].

        Each of ``args.sample_step`` rounds diffuses the input to t* = ``args.t`` (randomized by
        ``+-args.t_delta`` with ``--rand_t``) and integrates back to t ~ 0; the rounds are chained
        and all intermediate results are concatenated along the batch dimension. For
        ``bs_id < 2`` the inputs and outputs are saved under ``args.log_dir/bs<bs_id>_<tag>``.
        """
        assert isinstance(img, torch.Tensor)

        if tag is None:
            tag = 'rnd' + str(random.randint(0, 10000))
        out_dir = os.path.join(self.args.log_dir, 'bs' + str(bs_id) + '_' + tag)

        assert img.ndim == 4, img.ndim
        img = img.to(self.device)
        x0 = img

        if bs_id < 2:
            os.makedirs(out_dir, exist_ok=True)
            tvu.save_image((x0 + 1) * 0.5, os.path.join(out_dir, 'original_input.png'))

        xs = []
        for it in range(self.args.sample_step):
            e = self._initial_noise(x0)
            assert e.shape == x0.shape

            total_noise_levels = self.args.t
            if self.args.rand_t:
                total_noise_levels = self.args.t + np.random.randint(-self.args.t_delta, self.args.t_delta)
                logger.debug('randomized t*: %d', total_noise_levels)
            # closed-form forward diffusion q(x_t | x_0)
            a = (1 - self.betas).cumprod(dim=0).to(self.device)
            x = x0 * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()

            if bs_id < 2:
                tvu.save_image((x + 1) * 0.5, os.path.join(out_dir, f'init_{it}.png'))

            x0 = self._integrate(x)

            if bs_id < 2:
                torch.save(x0, os.path.join(out_dir, f'samples_{it}.pth'))
                tvu.save_image((x0 + 1) * 0.5, os.path.join(out_dir, f'samples_{it}.png'))

            xs.append(x0)

        return torch.cat(xs, dim=0)
