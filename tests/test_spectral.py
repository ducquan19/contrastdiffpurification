"""SRSG, FreqPure, the low-frequency PGD and the dev split (CPU, no score network)."""

import math

import numpy as np
import pytest
import torch

from src.data.loaders import REPORT_SEED, REPORT_SIZE, dev_pool_indices, select_subset_indices
from src.evaluation.attacks_full import lowpass_project, pgd_eot_full
from src.guidance.spectral import (FreqPureConfig, FreqPureGuidance, SRSGConfig, SpectralGuidance,
                                   gaussian_lowpass, radial_frequency, soft_lowpass)
from src.models.diffusion.sampler import ParticleSampler, alpha_bar_cont


def _srsg(seeds, K=2, **kw):
    g = SpectralGuidance(SRSGConfig(**kw))
    x_ref = torch.rand(len(seeds) * K, 3, 8, 8, dtype=torch.float64) * 2 - 1
    g.set_reference(x_ref=x_ref, seeds=torch.tensor(seeds), n_particles=K)
    return g, x_ref


def test_radial_frequency_is_symmetric_and_centred():
    d = radial_frequency(32, 32)
    assert d[0, 0] == 0 and torch.isclose(d.max(), torch.tensor(16 * math.sqrt(2)))
    assert torch.allclose(d, torch.roll(torch.flip(d, dims=(0, 1)), shifts=(1, 1), dims=(0, 1)))


def test_soft_mask_has_no_hard_zero_or_one():
    m = soft_lowpass(radial_frequency(32, 32), torch.tensor([4.0]), tau=1.0)
    assert (m > 0).all() and (m < 1).all()


def test_srsg_limits_gamma_zero_and_full_band():
    g, x_ref = _srsg([1, 2], gamma_min=0.0, gamma_max=0.0)
    x0 = torch.rand_like(x_ref)
    assert torch.allclose(g.corrected_x0(x0), x0)
    g, x_ref = _srsg([1, 2], gamma_min=1.0, gamma_max=1.0, r_min=100.0, r_max=100.0, tau=1e-3)
    assert torch.allclose(g.corrected_x0(torch.rand_like(x_ref)), x_ref, atol=1e-8)


def test_srsg_draws_are_per_image_and_reproducible():
    g1, _ = _srsg([7, 8, 9])
    g2, _ = _srsg([7, 8, 9])
    g3, _ = _srsg([8])
    assert torch.equal(g1.r, g2.r) and torch.equal(g1.gamma, g2.gamma)
    assert torch.equal(g1.r[2:4], g3.r) and torch.equal(g1.gamma[2:4], g3.gamma)   # image 8, any batch
    assert not torch.equal(g1.r[0:2], g1.r[2:4])
    c = SRSGConfig()
    assert ((g1.r >= c.r_min) & (g1.r <= c.r_max)).all()
    assert ((g1.gamma >= c.gamma_min) & (g1.gamma <= c.gamma_max)).all()


@pytest.mark.parametrize("mode", ["complex", "amp"])
def test_srsg_is_differentiable_in_both_inputs(mode):
    g, x_ref = _srsg([3], K=1, mode=mode)
    x_ref = x_ref.clone().requires_grad_(True)
    g.x_ref = x_ref
    x0 = torch.rand_like(x_ref, requires_grad=True)
    out = g.corrected_x0(x0)
    gx0, gref = torch.autograd.grad(out.pow(2).sum(), (x0, x_ref))
    assert gx0.abs().sum() > 0 and gref.abs().sum() > 0
    assert torch.autograd.gradcheck(lambda a, b: (setattr(g, "x_ref", b), g.corrected_x0(a))[1],
                                    (x0[:, :1, :4, :4].detach().clone().requires_grad_(True),
                                     x_ref[:, :1, :4, :4].detach().clone().requires_grad_(True)))


def test_noise_correction_reproduces_the_corrected_estimate():
    g, x_ref = _srsg([5])
    x_t, eps = torch.randn_like(x_ref), torch.randn_like(x_ref)
    t = 0.05
    ab = float(alpha_bar_cont(t))
    x0_hat = (x_t - math.sqrt(1 - ab) * eps) / math.sqrt(ab)
    d = g(x_t=x_t, x0_hat=x0_hat, eps=eps, t=t, t_star=0.1, n_particles=2)
    x0_used = (x_t - math.sqrt(1 - ab) * (eps + d)) / math.sqrt(ab)
    assert torch.allclose(x0_used, g.corrected_x0(x0_hat), atol=1e-9)


def test_freqpure_exchanges_low_amplitude_and_projects_low_phase():
    torch.manual_seed(0)
    g = FreqPureGuidance(FreqPureConfig(d_a=3, d_p=2, delta=0.2))
    # same sign of the mean, as for natural images: the DC phase (0 or pi) then agrees and the
    # exchanged spectrum stays Hermitian, so the amplitudes survive the real part of the iDFT
    x_ref = torch.rand(2, 3, 16, 16, dtype=torch.float64) * 0.8 + 0.1
    g.set_reference(x_ref=x_ref, seeds=torch.tensor([0, 1]), n_particles=1)
    x0 = torch.rand_like(x_ref) * 0.8 + 0.1
    out = g.corrected_x0(x0)
    F = lambda z: torch.fft.fft2(z, norm="ortho")
    d = radial_frequency(16, 16, dtype=torch.float64)
    low_a, low_p, high = d < 3, d < 2, d >= 3
    assert torch.allclose(F(out).abs()[..., low_a], F(x_ref).abs()[..., low_a], atol=1e-6)
    assert torch.allclose(F(out).abs()[..., high], F(x0).abs()[..., high], atol=1e-6)
    dphi = torch.atan2(torch.sin(F(out).angle() - F(x_ref).angle()), torch.cos(F(out).angle() - F(x_ref).angle()))
    assert (dphi[..., low_p].abs() <= 0.2 + 1e-6).all()


def test_sampler_passes_the_input_as_reference():
    class Recorder:
        def set_reference(self, *, x_ref, seeds, n_particles):
            self.x_ref, self.K = x_ref, n_particles

        def __call__(self, **kw):
            return None

    rec = Recorder()
    s = ParticleSampler(lambda x, t: torch.zeros_like(x), t_star=100, steps=2, n_particles=3,
                        guidance=rec, checkpoint="none")
    x = torch.rand(2, 3, 8, 8)
    s(x, seeds=torch.tensor([1, 2]))
    assert rec.K == 3 and torch.allclose(rec.x_ref, (x * 2 - 1).repeat_interleave(3, dim=0))


def test_gaussian_lowpass_keeps_dc_and_the_linf_bound():
    torch.manual_seed(0)
    d = (torch.rand(4, 3, 32, 32) * 2 - 1) * (8 / 255)
    lp = gaussian_lowpass(d, sigma=4.0)
    assert lp.abs().max() <= d.abs().max() * 1.05
    assert torch.allclose(lp.mean(dim=(-2, -1)), d.mean(dim=(-2, -1)), atol=1e-6)
    hf = radial_frequency(32, 32) > 12
    energy = lambda z: torch.fft.fft2(z, norm="ortho").abs()[..., hf].pow(2).sum()
    assert energy(lp) < 1e-3 * energy(d)


def test_lowpass_pgd_stays_in_the_ball():
    torch.manual_seed(0)
    fc = torch.nn.Linear(3 * 16 * 16, 10)
    model = lambda z: fc(z.flatten(1))
    x = torch.rand(4, 3, 16, 16)
    y = model(x).argmax(1)
    eps = 8 / 255
    x_last, x_best, _ = pgd_eot_full(model, x, y, eps=eps, steps=5, eot_iter=1, log_every=0,
                                     lowpass_sigma=3.0)
    for z in (x_last, x_best):
        assert (z - x).abs().max() <= eps + 1e-6 and z.min() >= 0 and z.max() <= 1
    p = lowpass_project(x + 0.5, x, eps, "Linf", 3.0)
    assert (p - x).abs().max() <= eps + 1e-6


def test_dev_pool_excludes_the_reported_images():
    report = set(select_subset_indices(10000, REPORT_SIZE, REPORT_SEED).tolist())
    dev = dev_pool_indices(10000)
    assert len(dev) == 10000 - REPORT_SIZE and not report & set(dev.tolist())
    assert np.all(np.diff(dev) > 0)
