"""Lightning module for training a score-SDE (NCSN++) diffusion model and sampling from it.

The CLI is in scripts/train_score_sde.py.
"""
import numpy as np
import pytorch_lightning as pl
import torch

import src.models.score_sde.losses as losses
import src.models.score_sde.sampling as sampling
from src.models.score_sde import sde_lib
from src.models.score_sde.models import utils as mutils

from src.config import ConfigMain


def get_data_scaler(is_data_centered: bool):
    """Data normalizer. Assume data are always in [0, 1]."""
    if is_data_centered:
        # Rescale to [-1, 1]
        return lambda x: x * 2. - 1.
    else:
        return lambda x: x


def get_data_inverse_scaler(is_data_centered: bool):
    """Inverse data normalizer."""
    if is_data_centered:
        # Rescale [-1, 1] to [0, 1]
        return lambda x: (x + 1.) / 2.
    else:
        return lambda x: x


class ScoreSDETrain(pl.LightningModule):
    """Trains ``mutils.create_model(config)`` with the score_sde loss selected by ``config.training``."""

    def __init__(self, config: ConfigMain):
        super().__init__()
        self.config = config
        self.model = mutils.create_model(config)

        sde, sampling_eps = self.build_sde()
        self.sampling_eps = sampling_eps
        self.loss_func = self.get_sde_loss_func(sde)

    def build_sde(self):
        """Return ``(sde, sampling_eps)`` for ``config.training.sde`` (vpsde / subvpsde / vesde)."""
        config = self.config
        sde_type = config.training.sde.lower()

        # Setup SDEs
        if sde_type == 'vpsde':
            sde = sde_lib.VPSDE(
                beta_min=config.model.beta_min,
                beta_max=config.model.beta_max,
                N=config.model.num_scales
            )
            sampling_eps = 1e-3
        elif sde_type == 'subvpsde':
            sde = sde_lib.subVPSDE(
                beta_min=config.model.beta_min,
                beta_max=config.model.beta_max,
                N=config.model.num_scales
            )
            sampling_eps = 1e-3
        elif sde_type == 'vesde':
            sde = sde_lib.VESDE(
                sigma_min=config.model.sigma_min,
                sigma_max=config.model.sigma_max,
                N=config.model.num_scales
            )
            sampling_eps = 1e-5
        else:
            raise NotImplementedError(f"SDE {config.training.sde} unknown.")
        return sde, sampling_eps

    def get_sde_loss_func(self, sde):
        """Continuous-time SDE loss, or the discrete SMLD / DDPM loss when ``continuous`` is false."""
        continuous = self.config.training.continuous
        reduce_mean = self.config.training.reduce_mean
        likelihood_weighting = self.config.training.likelihood_weighting
        if continuous:
            loss_fn_train = losses.get_sde_loss_fn(
                sde,
                True,
                reduce_mean=reduce_mean,
                continuous=True,
                likelihood_weighting=likelihood_weighting
            )
        else:
            assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
            if isinstance(sde, sde_lib.VESDE):
                loss_fn_train = losses.get_smld_loss_fn(
                    sde, True, reduce_mean=reduce_mean)
            elif isinstance(sde, sde_lib.VPSDE):
                loss_fn_train = losses.get_ddpm_loss_fn(
                    sde, True, reduce_mean=reduce_mean)
            else:
                raise ValueError(
                    f"Discrete training for {sde.__class__.__name__} is not recommended.")

        return loss_fn_train

    def training_step(self, batch, batch_idx):
        images, labels = batch

        loss = self.loss_func(self.model, images)

        self.log('train_loss', loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return losses.get_optimizer(self.config, self.trainer.model.parameters())

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """Optimizer step with a linear learning-rate warm-up over ``config.optim.warmup`` steps."""
        warmup = self.config.optim.warmup
        steps = self.trainer.global_step
        if warmup > 0:
            for pg in optimizer.param_groups:
                pg['lr'] = self.config.optim.lr * \
                    np.minimum(steps / warmup, 1.0)
        optimizer.step(closure=optimizer_closure)

    @torch.no_grad()
    def get_sample_func(self):
        """Return ``sample(batch_size, model)`` using the sampler in ``config.sampling`` (pc / ode)."""
        sde, sampling_eps = self.build_sde()
        inverse_scaler = get_data_inverse_scaler(self.config.data.centered)
        _config = self.config
        device = self.device

        def __inner(batch_size, model):

            sampling_shape = (
                batch_size,
                _config.data.num_channels,
                _config.data.image_size,
                _config.data.image_size
            )

            sampler_name = _config.sampling.method

            # from score_sde's run_lib
            # Probability flow ODE sampling with black-box ODE solvers
            if sampler_name.lower() == 'ode':
                sampling_fn = sampling.get_ode_sampler(
                    sde=sde,
                    shape=sampling_shape,
                    inverse_scaler=inverse_scaler,
                    denoise=_config.sampling.noise_removal,
                    eps=sampling_eps,
                    device=device
                )
            # Predictor-Corrector sampling. Predictor-only and Corrector-only samplers are special cases.
            elif sampler_name.lower() == 'pc':
                predictor = sampling.get_predictor(_config.sampling.predictor.lower())
                corrector = sampling.get_corrector(_config.sampling.corrector.lower())
                sampling_fn = sampling.get_pc_sampler(
                    sde=sde,
                    shape=sampling_shape,
                    predictor=predictor,
                    corrector=corrector,
                    inverse_scaler=inverse_scaler,
                    snr=_config.sampling.snr,
                    n_steps=_config.sampling.n_steps_each,
                    probability_flow=_config.sampling.probability_flow,
                    continuous=_config.training.continuous,
                    denoise=_config.sampling.noise_removal,
                    eps=sampling_eps,
                    device=device
                )
            else:
                raise ValueError(f"Sampler name {sampler_name} unknown.")

            return sampling_fn(model)

        return __inner

    @torch.no_grad()
    def generate_sample(self, batch_size: int, sample_func: callable = None):
        """Draw ``batch_size`` images in [0, 1] from the current (non-EMA) weights."""
        if sample_func is None:
            sample_func = self.get_sample_func()

        sample, _ = sample_func(batch_size, self.model)
        return sample
