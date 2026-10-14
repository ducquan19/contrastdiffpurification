# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DiffPure. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
"""Purification by the probability-flow ODE.

Solver settings are fixed here rather than exposed as flags, because they are part of the
published method: ``torchdiffeq.odeint_adjoint`` with ``method='euler'`` and
``atol = rtol = 1e-3`` over ``ts = [t*/1000, 1e-5]``. Only ``--step_size`` (the Euler step) and
``--t`` (t*) are configurable.
"""

import logging

import torch
from torchdiffeq import odeint_adjoint

from src.models.diffusion.base import DiffusionPurifier, VPDynamics

logger = logging.getLogger(__name__)


class VPODE(VPDynamics):
    """Probability-flow ODE of the VP-SDE, in the ``func(t, states)`` form torchdiffeq expects."""

    def ode_fn(self, t, x):
        """ODE drift ``f(x, t) - 0.5 * g(t)^2 * score(x, t)`` for flattened ``x`` (B, C*H*W)."""
        drift, diffusion = self.vpsde_fn(t, x)
        return drift - 0.5 * diffusion[:, None] ** 2 * self.score(t, x)

    def forward(self, t, states):
        x = states[0]

        t = t.expand(x.shape[0])  # (batch_size, )
        dx_dt = self.ode_fn(t, x)
        assert dx_dt.shape == x.shape

        return dx_dt,


class ODEPurifier(DiffusionPurifier):
    """Purifier that integrates the probability-flow ODE with ``torchdiffeq.odeint_adjoint``."""

    def __init__(self, args, config, device=None):
        super().__init__(args, config, device)
        # fixed solver settings; see the module docstring
        self.atol, self.rtol = 1e-3, 1e-3
        self.method = 'euler'
        logger.info('ODE solver: %s, atol %g, rtol %g, step size %g', self.method, self.atol, self.rtol,
                    self.args.step_size)

    def _build_dynamics(self, model, img_shape):
        return VPODE(model=model, img_shape=img_shape)

    def _integrate(self, x):
        t0, t1 = self.args.t / 1000, 1e-5
        ts = torch.linspace(t0, t1, 2).to(self.device)

        state_t = odeint_adjoint(
            self.dynamics,
            (x.view(x.shape[0], -1), ),  # (batch_size, state_size)
            ts,
            atol=self.atol,
            rtol=self.rtol,
            method=self.method,
            options=dict(step_size=self.args.step_size),  # Euler step
        )
        return state_t[0][-1].view(x.shape)  # (batch_size, c, h, w)
