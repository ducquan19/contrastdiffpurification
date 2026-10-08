"""CPU tests of the method module (torch only)."""
import pytest
import torch
import torch.nn as nn

from src.guidance.contrastive import ContrastiveGuidedDiffusion, ContrastiveLoss


class ToyEps(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 6, 3, padding=1)  # learned-sigma style output: 2*C channels

    def forward(self, x, t):
        return self.conv(x)


class ToyCls(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3 * 8 * 8, 10)

    def forward(self, x):
        return self.fc(x.flatten(1))


def _make(drift_min=1):
    torch.manual_seed(0)
    return ContrastiveGuidedDiffusion(ToyEps(), ToyCls().eval(), n_classes=10,
                                      drift_min=drift_min, temperature=1.0,
                                      guidance_strength=500)


def test_first_call_equals_wrapped_model():
    cg = _make()
    x, t = torch.randn(4, 3, 8, 8), torch.full((4,), 500)
    assert torch.equal(cg(x, t), cg.model(x, t))  # no cached x_t yet -> no drift


def test_drift_only_touches_mean_channels_after_first_step():
    cg = _make()
    x, t = torch.randn(4, 3, 8, 8), torch.full((4,), 500)
    cg(x, t)
    x2 = x + 0.01
    out, base = cg(x2, t), cg.model(x2, t)
    assert not torch.allclose(out[:, :3], base[:, :3])
    assert torch.equal(out[:, 3:], base[:, 3:])
    assert cg.drift_counter == 2


def test_reset_drift_counter():
    cg = _make()
    x, t = torch.randn(4, 3, 8, 8), torch.full((4,), 500)
    cg(x, t)
    cg.reset_drift_counter()
    assert cg.drift_counter == 0


def test_hard_loss_is_finite():
    torch.manual_seed(0)
    loss = ContrastiveLoss(estimator="hard")(torch.randn(6, 10), torch.randn(6, 10))
    assert torch.isfinite(loss)


@pytest.mark.xfail(raises=RuntimeError, strict=True,
                   reason="known bug in the original code: soft estimator has a shape mismatch")
def test_soft_loss_known_bug():
    ContrastiveLoss(estimator="soft")(torch.randn(6, 10), torch.randn(6, 10))


def test_guidance_strength_scales_the_correction():
    """lambda in Eq. (8) multiplies the contrastive gradient."""
    x, t = torch.randn(4, 3, 8, 8), torch.full((4,), 500)
    deltas = []
    for strength in (500, 1000):
        torch.manual_seed(0)
        cg = ContrastiveGuidedDiffusion(ToyEps(), ToyCls().eval(), n_classes=10,
                                        drift_min=1, temperature=1.0,
                                        guidance_strength=strength)
        cg(x, t)
        x2 = x + 0.01
        deltas.append((cg(x2, t) - cg.model(x2, t))[:, :3])
    assert torch.allclose(deltas[1], deltas[0] * 2, atol=1e-5)


def test_no_guidance_outside_the_drift_window():
    cg = _make(drift_min=10)
    x, t = torch.randn(4, 3, 8, 8), torch.full((4,), 500)
    cg(x, t)
    x2 = x + 0.01
    assert torch.equal(cg(x2, t), cg.model(x2, t))  # still before drift_min
