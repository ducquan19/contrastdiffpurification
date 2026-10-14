"""CPU tests of the PCP losses and guidance (torch only)."""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.guidance.memory_bank import PrototypeBank
from src.guidance.particle import (ParticleContrastiveGuidance, rep_bandwidth, rep_features, rep_grad_phi,
                                   rep_loss, sem_grad_z, sem_loss, sem_target)


def _unit(*shape, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(*shape, generator=g, dtype=dtype), dim=-1)


class TinyEnc(nn.Module):
    def __init__(self, d=4, dtype=torch.float32):
        super().__init__()
        torch.manual_seed(7)
        self.fc = nn.Linear(3 * 4 * 4, d).to(dtype)

    def forward(self, x01):
        return torch.tanh(self.fc(x01.flatten(1)))


def _guidance(sem=0.3, rep=0.3, dtype=torch.float32, scale_mode="normalized"):
    bank = PrototypeBank(_unit(1, 6, 4, seed=3), [0.05])
    g = ParticleContrastiveGuidance(TinyEnc(dtype=dtype), bank, sem_strength=sem, rep_strength=rep,
                                    tau_s=0.5, tau_t=0.25, sem_t_min=0.0, rep_t_min=0.0, rep_size=2,
                                    scale_mode=scale_mode)
    g.bank.prototypes = g.bank.prototypes.to(dtype)
    return g


# ------------------------------------------------------------------------------ analytic gradients
@pytest.mark.parametrize("K", [1, 4])
def test_sem_gradient_equals_autograd_with_fixed_target(K):
    z = _unit(3, K, 5, seed=1).requires_grad_(True)
    P = _unit(7, 5, seed=2)
    q = sem_target(z, P, 0.2).detach()
    loss = sem_loss(z, P, q, 0.4).sum()
    auto, = torch.autograd.grad(loss, z)
    assert torch.allclose(sem_grad_z(z, P, 0.4, 0.2), auto, atol=1e-6)


def test_rep_gradient_equals_autograd_with_fixed_bandwidth():
    phi = torch.randn(2, 5, 12, requires_grad=True)
    d2 = (phi.unsqueeze(2) - phi.unsqueeze(1)).pow(2).sum(-1)
    h = rep_bandwidth(d2).detach()
    auto, = torch.autograd.grad(rep_loss(phi, h).sum(), phi)
    assert torch.allclose(rep_grad_phi(phi), auto, atol=1e-6)


def test_descending_the_guidance_lowers_both_losses():
    torch.manual_seed(0)
    phi = torch.randn(2, 4, 12)
    # rep_grad_phi holds the bandwidth fixed, so compare the losses at that same bandwidth
    h = rep_bandwidth((phi.unsqueeze(2) - phi.unsqueeze(1)).pow(2).sum(-1))
    before = rep_loss(phi, h)
    after = rep_loss(phi - 0.05 * rep_grad_phi(phi) / rep_grad_phi(phi).norm(), h)
    assert (after < before).all()

    z = _unit(2, 4, 5, seed=4)
    P = _unit(6, 5, seed=5)
    q = sem_target(z, P, 0.2)
    g = sem_grad_z(z, P, 0.4, 0.2)
    z2 = F.normalize(z - 0.01 * g / g.norm(), dim=-1)
    assert (sem_loss(z2, P, q, 0.4) < sem_loss(z, P, q, 0.4)).all()


def test_losses_are_per_image():
    """Adding an image to the batch does not change the loss or gradient of the others."""
    phi = torch.randn(3, 4, 6)
    assert torch.allclose(rep_grad_phi(phi)[:2], rep_grad_phi(phi[:2]), atol=1e-7)
    z, P = _unit(3, 4, 5, seed=6), _unit(6, 5, seed=7)
    assert torch.allclose(sem_grad_z(z, P, 0.4, 0.2)[:2], sem_grad_z(z[:2], P, 0.4, 0.2), atol=1e-7)


# ------------------------------------------------------------------------------ the guidance module
def _call(g, x0, eps, t=0.05, K=3):
    return g(x_t=x0, x0_hat=x0, eps=eps, t=t, t_star=0.1, n_particles=K)


def test_normalized_correction_has_the_requested_norm():
    g = _guidance(sem=0.3, rep=0.0)
    x0, eps = torch.randn(6, 3, 4, 4), torch.randn(6, 3, 4, 4)
    d = _call(g, x0, eps)
    ratio = d.flatten(1).norm(dim=1) / eps.flatten(1).norm(dim=1)
    assert torch.allclose(ratio, torch.full_like(ratio, 0.3), atol=1e-5)


def test_windows_and_zero_strength_disable_the_guidance():
    g = _guidance()
    g.sem_window, g.rep_window = (0.0, 0.2), (0.0, 0.2)
    x0, eps = torch.randn(6, 3, 4, 4), torch.randn(6, 3, 4, 4)
    assert _call(g, x0, eps, t=0.05) is None          # 0.05 > 0.2 * t_star = 0.02
    assert _call(_guidance(sem=0.0, rep=0.3), x0, eps, K=1) is None  # repulsion needs K > 1


def test_sem_requires_encoder_and_bank():
    with pytest.raises(ValueError):
        ParticleContrastiveGuidance(None, None, sem_strength=0.1)
    with pytest.raises(ValueError):
        ParticleContrastiveGuidance(TinyEnc(), PrototypeBank(_unit(1, 3, 4), [0.0]), tau_s=0.1, tau_t=0.2)


def test_inference_branch_returns_a_constant():
    g = _guidance()
    x0 = torch.randn(6, 3, 4, 4, requires_grad=True)
    with torch.no_grad():
        d = _call(g, x0, torch.randn(6, 3, 4, 4))
    assert not d.requires_grad


@pytest.mark.parametrize("scale_mode", ["normalized", "raw"])
def test_full_gradient_through_the_guidance_is_exact(scale_mode):
    """T4: gradcheck of x -> guidance correction, including the input dependence of q and h."""
    g = _guidance(dtype=torch.float64, scale_mode=scale_mode)

    def f(x):
        x0 = torch.tanh(x)           # some differentiable map to the Tweedie estimate
        eps = 0.5 * x + 0.1
        return _call(g, x0, eps)

    x = torch.randn(3, 3, 4, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(f, (x,), eps=1e-6, atol=1e-5)


def test_rep_features_layout():
    x = torch.arange(2 * 3 * 3 * 4 * 4, dtype=torch.float32).view(6, 3, 4, 4)
    phi = rep_features(x, 3, 2)
    assert phi.shape == (2, 3, 12)
    assert torch.allclose(phi[1, 0], F.adaptive_avg_pool2d(x[3:4], 2).flatten())
