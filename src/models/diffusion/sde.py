# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DiffPure. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
"""Purification by the reverse VP-SDE (DiffPure).

Solver settings are fixed here rather than exposed as flags, because they are part of the
published method: ``torchsde.sdeint_adjoint`` with ``method='euler'`` over the two end points
``ts = [1 - t*/1000, 1 - 1e-5]`` (the solver subdivides that interval itself). The adjoint
solver is what makes exact gradients available to an adaptive attack. ``--t`` sets t*, and
``--use_bm`` adds a fixed Brownian interval.
"""

import numpy as np
import torch
import torchsde

from src.models.diffusion.base import DiffusionPurifier, VPDynamics


class RevVPSDE(VPDynamics):
    """Reverse-time VP-SDE in the ``f`` / ``g`` form torchsde expects (time runs as 1 - t)."""

    noise_type = "diagonal"
    sde_type = "ito"

    def rvpsde_fn(self, t, x, return_type='drift'):
        """Reverse-SDE drift ``f - g^2 * score`` (``return_type='drift'``) or diffusion ``g``."""
        drift, diffusion = self.vpsde_fn(t, x)
        if return_type == 'drift':
            return drift - diffusion[:, None] ** 2 * self.score(t, x)
        return diffusion

    def f(self, t, x):
        """Drift -f(x, 1 - t) for flattened ``x`` (B, C*H*W)."""
        t = t.expand(x.shape[0])  # (batch_size, )
        drift = self.rvpsde_fn(1 - t, x, return_type='drift')
        assert drift.shape == x.shape
        return -drift

    def g(self, t, x):
        """Diffusion g(1 - t), broadcast to the shape of ``x``."""
        t = t.expand(x.shape[0])  # (batch_size, )
        diffusion = self.rvpsde_fn(1 - t, x, return_type='diffusion')
        assert diffusion.shape == (x.shape[0], )
        return diffusion[:, None].expand(x.shape)


class SDEPurifier(DiffusionPurifier):
    """Purifier that integrates the reverse SDE with ``torchsde.sdeint_adjoint``."""

    def __init__(self, args, config, device=None):
        super().__init__(args, config, device)
        print(f't: {args.t}, rand_t: {args.rand_t}, t_delta: {args.t_delta}')
        print(f'use_bm: {args.use_bm}')

    def _build_dynamics(self, model, img_shape):
        return RevVPSDE(model=model, img_shape=img_shape)

    def _integrate(self, x):
        batch_size = x.shape[0]
        state_size = int(np.prod(x.shape[1:]))  # c*h*w

        # fixed solver window; see the module docstring
        t0, t1 = 1 - self.args.t / 1000, 1 - 1e-5
        ts = torch.linspace(t0, t1, 2).to(self.device)

        x_ = x.view(batch_size, -1)  # (batch_size, state_size)
        if self.args.use_bm:
            bm = torchsde.BrownianInterval(t0=t0, t1=t1, size=(batch_size, state_size), device=self.device)
            xs_ = torchsde.sdeint_adjoint(self.dynamics, x_, ts, method='euler', bm=bm)
        else:
            xs_ = torchsde.sdeint_adjoint(self.dynamics, x_, ts, method='euler')
        return xs_[-1].view(x.shape)  # (batch_size, c, h, w)
