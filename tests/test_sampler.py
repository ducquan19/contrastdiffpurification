"""CPU tests of the differentiable particle sampler (torch only)."""
import math

import pytest
import torch
import torch.nn as nn

from src.guidance.memory_bank import PrototypeBank
from src.guidance.particle import ParticleContrastiveGuidance
from src.models.diffusion.sampler import (ParticleSampler, alpha_bar_cont, beta_cont,
                                          discrete_alpha_bar, make_time_grid, per_sample_noise)

SHAPE = (3, 8, 8)


class ToyEps(nn.Module):
    """Smooth, input-dependent noise predictor."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv = nn.Conv2d(3, 3, 3, padding=1)

    def forward(self, x, t):
        return torch.tanh(self.conv(x)) * (1 + t.view(-1, 1, 1, 1))


class ToyEnc(nn.Module):
    def __init__(self, d=6):
        super().__init__()
        torch.manual_seed(1)
        self.fc = nn.Linear(3 * 8 * 8, d)

    def forward(self, x01):
        return torch.tanh(self.fc(x01.flatten(1)))


def _pcp_guidance(sem=0.2, rep=0.2):
    torch.manual_seed(2)
    bank = PrototypeBank(torch.randn(2, 5, 6), [0.0, 0.1])
    return ParticleContrastiveGuidance(ToyEnc(), bank, sem_strength=sem, rep_strength=rep,
                                       tau_s=0.5, tau_t=0.25, sem_t_min=0.0, rep_t_min=0.0, rep_size=4)


def gaussian_eps(s):
    """Exact noise prediction for data ~ N(0, s^2 I): x_t ~ N(0, a s^2 + 1 - a)."""
    def eps_fn(x, t_vec):
        a = alpha_bar_cont(t_vec).to(x.dtype).view(-1, 1, 1, 1)
        v = a * s ** 2 + (1 - a)
        return torch.sqrt(1 - a) * x / v
    return eps_fn


# ------------------------------------------------------------------------------ schedule
def test_schedule_matches_the_original_formulas():
    assert abs(float(alpha_bar_cont(0.1)) - math.exp(-0.5 * 19.9 * 0.01 - 0.1 * 0.1)) < 1e-12
    assert abs(float(beta_cont(0.5)) - (0.1 + 0.5 * 19.9)) < 1e-12
    # forward noising (discrete, as base.py) and reverse start (continuous) agree at t* = 100
    assert abs(discrete_alpha_bar(100) - float(alpha_bar_cont(0.1))) < 2e-3
    g = make_time_grid(0.1, 4)
    assert len(g) == 5 and float(g[0]) == pytest.approx(0.1) and float(g[-1]) == pytest.approx(1e-5)


def test_per_sample_noise_depends_only_on_the_own_seed():
    seeds = torch.tensor([11, 22, 33])
    full = per_sample_noise(seeds, 2, 3, SHAPE, "cpu")
    alone = per_sample_noise(seeds[1:2], 2, 3, SHAPE, "cpu")
    assert full.shape == (3, 6, *SHAPE)
    assert torch.equal(full[:, 2:4], alone)


# ------------------------------------------------------------------------------ solvers
@pytest.mark.parametrize("solver", ["em", "ddim"])
def test_solvers_preserve_a_gaussian_data_distribution(solver):
    """With the exact score of N(0, s^2), purifying samples of the data must keep variance s^2."""
    s = 0.5
    torch.manual_seed(0)
    x0 = s * torch.randn(1500, *SHAPE).clamp(-3, 3)
    sampler = ParticleSampler(gaussian_eps(s), t_star=500, steps=200, solver=solver, eta=1.0,
                              checkpoint="none")
    out = sampler.purify_centered(x0, seeds=torch.arange(1500))
    assert out.var().item() == pytest.approx(s ** 2, rel=0.08)


def test_ddim_eta0_with_a_point_mass_returns_the_point():
    """Data = {0}: eps(x_t) = x_t / sqrt(1 - a), so every Tweedie estimate is 0."""
    def eps_fn(x, t_vec):
        a = alpha_bar_cont(t_vec).to(x.dtype).view(-1, 1, 1, 1)
        return x / torch.sqrt(1 - a)
    sampler = ParticleSampler(eps_fn, t_star=300, steps=10, solver="ddim", eta=0.0)
    out = sampler.purify_centered(torch.randn(4, *SHAPE), seeds=torch.arange(4))
    assert out.abs().max().item() < 1e-2


def test_em_step_is_the_reverse_vp_sde():
    """One Euler-Maruyama step equals x + h (0.5 b x + b score) + sqrt(b h) z (DiffPure's drift)."""
    sampler = ParticleSampler(ToyEps(), t_star=100, steps=1, solver="em", checkpoint="none")
    x, z = torch.randn(2, *SHAPE), torch.randn(2, *SHAPE)
    t, s = 0.1, 0.05
    eps = ToyEps()(x, torch.full((2,), t))
    a, b = float(alpha_bar_cont(t)), float(beta_cont(t))
    expected = x + (t - s) * (0.5 * b * x + b * (-eps / math.sqrt(1 - a))) + math.sqrt(b * (t - s)) * z
    assert torch.allclose(sampler._step(x, eps, t, s, z), expected, atol=1e-6)


# ------------------------------------------------------------------------------ batch independence
@pytest.mark.parametrize("guidance", [None, "pcp"])
def test_purification_of_an_image_does_not_depend_on_the_batch(guidance):
    """T2 + T3: same per-image seeds -> same output, whatever the batch and its order."""
    g = _pcp_guidance() if guidance == "pcp" else None
    sampler = ParticleSampler(ToyEps(), t_star=100, steps=6, n_particles=4, guidance=g)
    torch.manual_seed(3)
    x, seeds = torch.rand(5, *SHAPE), torch.tensor([5, 6, 7, 8, 9])
    full = sampler(x, seeds)
    assert full.shape == (5, 4, *SHAPE)
    for i in range(5):
        assert torch.allclose(sampler(x[i:i + 1], seeds[i:i + 1])[0], full[i], atol=1e-5)
    perm = torch.tensor([3, 0, 4, 1, 2])
    assert torch.allclose(sampler(x[perm], seeds[perm]), full[perm], atol=1e-5)


def test_contrastive_guidance_couples_the_batch():
    """ContrastDiffPure (in-batch negatives): the same image purifies differently in another batch."""
    from src.evaluation.particle_model import ContrastiveCorrection
    from src.guidance.contrastive import ContrastiveGuidedDiffusion

    class ToyCls(nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(4)
            self.fc = nn.Linear(3 * 8 * 8, 10)

        def forward(self, x):
            return self.fc(x.flatten(1))

    def run(xb, sb):
        torch.manual_seed(0)  # same initialisation of F(.) in both runs
        wrapper = ContrastiveGuidedDiffusion(ToyEps(), ToyCls().eval(), n_classes=10, drift_min=0,
                                             temperature=1.0, guidance_strength=500,
                                             clear_cache_on_reset=True)
        sampler = ParticleSampler(ToyEps(), t_star=100, steps=6, guidance=ContrastiveCorrection(wrapper))
        return sampler(xb, sb)

    torch.manual_seed(5)
    x, seeds = torch.rand(6, *SHAPE), torch.arange(6)
    together = run(x, seeds)[0]
    other_batch = run(torch.cat([x[:1], torch.rand(5, *SHAPE)]), seeds)[0]
    assert not torch.allclose(together, other_batch, atol=1e-4)


# ------------------------------------------------------------------------------ gradients
def test_checkpointing_does_not_change_the_gradient():
    """T8: recomputing the score network in backward gives the same input gradient."""
    grads = []
    for mode in ("none", "eps"):
        sampler = ParticleSampler(ToyEps(), t_star=100, steps=4, n_particles=3,
                                  guidance=_pcp_guidance(), checkpoint=mode)
        x = torch.rand(2, *SHAPE, requires_grad=True)
        sampler(x, torch.tensor([1, 2])).pow(2).sum().backward()
        grads.append(x.grad.clone())
    assert torch.allclose(grads[0], grads[1], atol=1e-6)


def test_the_attacker_sees_the_guidance():
    """T6: the input gradient changes when the guidance is switched on, and is finite."""
    def grad_with(g):
        sampler = ParticleSampler(ToyEps(), t_star=100, steps=4, n_particles=3, guidance=g)
        x = torch.rand(2, *SHAPE, requires_grad=True)
        sampler(x, torch.tensor([1, 2])).pow(2).sum().backward()
        return x.grad
    g0, g1 = grad_with(None), grad_with(_pcp_guidance(sem=0.5, rep=0.5))
    assert torch.isfinite(g1).all()
    assert not torch.allclose(g0, g1, atol=1e-6)


def test_inference_without_grad_runs_the_guidance():
    sampler = ParticleSampler(ToyEps(), t_star=100, steps=3, n_particles=2, guidance=_pcp_guidance())
    with torch.no_grad():
        out = sampler(torch.rand(2, *SHAPE), torch.tensor([1, 2]))
    assert not out.requires_grad
    assert sampler.last_stats["guided_steps"] == 3


def test_invalid_arguments():
    with pytest.raises(ValueError):
        ParticleSampler(ToyEps(), solver="rk4")
    with pytest.raises(ValueError):
        ParticleSampler(ToyEps(), n_particles=0)
    with pytest.raises(ValueError):
        ParticleSampler(ToyEps()).purify_centered(torch.rand(2, *SHAPE), seeds=torch.tensor([1]))
