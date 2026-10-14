"""CPU tests of the PCP bank, config and voting (torch + yaml only)."""
import argparse

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from src.evaluation.metrics import wilson_interval
from src.evaluation.voting import hard_vote_scores, particle_agreement, soft_vote_logits
from src.guidance.memory_bank import PrototypeBank, build_prototype_bank, spherical_kmeans
from src.method_pcp import PCPConfig, add_pcp_args, load_pcp_config, resolve_pcp
from src.paths import config_path


# ------------------------------------------------------------------------------ memory bank
def test_spherical_kmeans_recovers_separated_clusters():
    g = torch.Generator().manual_seed(0)
    centres = F.normalize(torch.eye(3, 8), dim=-1)
    x = torch.cat([F.normalize(c + 0.05 * torch.randn(200, 8, generator=g), dim=-1) for c in centres])
    found = spherical_kmeans(x, 3, seed=1)
    sim = found @ centres.t()
    assert (sim.max(dim=0).values > 0.99).all()


def test_bank_picks_the_nearest_bucket_and_round_trips(tmp_path):
    bank = PrototypeBank(torch.randn(3, 4, 5), [0.0, 0.05, 0.1], {"encoder": "toy"})
    assert torch.equal(bank.at(0.06), bank.prototypes[1])
    assert torch.equal(bank.at(0.2), bank.prototypes[2])
    assert torch.allclose(bank.prototypes.norm(dim=-1), torch.ones(3, 4))
    p = tmp_path / "bank.pt"
    bank.save(str(p))
    loaded = PrototypeBank.load(str(p))
    assert torch.equal(loaded.prototypes, bank.prototypes) and loaded.meta["encoder"] == "toy"
    assert loaded.fingerprint() == bank.fingerprint()


def test_build_prototype_bank_shapes():
    enc = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 6))
    eps_fn = lambda x, t: torch.zeros_like(x)  # noqa: E731 - x0_hat = x_t / sqrt(a)
    batches = [torch.rand(20, 3, 4, 4) for _ in range(2)]
    bank = build_prototype_bank(enc, batches, n_prototypes=5, t_values=[0.0, 0.05], eps_fn=eps_fn)
    assert bank.prototypes.shape == (2, 5, 6) and bank.meta["n_images"] == 40
    with pytest.raises(ValueError):
        build_prototype_bank(enc, batches, n_prototypes=5, t_values=[0.05])


# ------------------------------------------------------------------------------ config
def test_shipped_pcp_config_loads():
    c = load_pcp_config()
    assert c.n_particles == 8 and c.steps == 20 and c.solver == "ddim" and c.vote == "hard"
    assert 0 < c.tau_t <= c.tau_s


def test_pcp_config_reports_missing_and_unknown_keys(tmp_path):
    raw = yaml.safe_load(open(config_path("method/pcp.yaml")))
    del raw["eta"]
    raw["lambda"] = 1
    p = tmp_path / "pcp.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError) as e:
        load_pcp_config(str(p))
    assert "missing key(s): eta" in str(e.value) and "unknown key(s): lambda" in str(e.value)


def test_pcp_config_rejects_invalid_values():
    base = load_pcp_config().as_dict()
    for bad in ({"solver": "rk4"}, {"n_particles": 0}, {"tau_t": 1.0, "tau_s": 0.5},
                {"sem_t_min": 0.8, "sem_t_max": 0.2}, {"vote": "median"}, {"eta": -1}):
        with pytest.raises(ValueError):
            PCPConfig(**{**base, **bad})


def test_pcp_cli_overrides_the_file():
    parser = argparse.ArgumentParser()
    add_pcp_args(parser)
    args = parser.parse_args(["--pcp_n_particles", "16", "--pcp_rep_strength", "0.25", "--pcp_solver", "em"])
    c = resolve_pcp(args)
    assert c.n_particles == 16 and c.rep_strength == 0.25 and c.solver == "em"
    assert c.steps == load_pcp_config().steps  # untouched


# ------------------------------------------------------------------------------ voting + metrics
def test_soft_vote_is_the_log_mean_probability():
    logits = torch.randn(4, 5, 10)
    expected = torch.log(F.softmax(logits, -1).mean(1))
    assert torch.allclose(soft_vote_logits(logits), expected, atol=1e-6)


def test_hard_vote_is_the_majority():
    logits = torch.full((1, 5, 3), -5.0)
    for k, c in enumerate([2, 2, 2, 0, 0]):
        logits[0, k, c] = 5.0
    assert hard_vote_scores(logits).argmax(1).item() == 2


def test_particle_agreement():
    y = torch.tensor([0, 1, 2, 3])
    same = torch.tensor([[0, 0], [9, 9], [2, 2], [9, 9]])      # errors perfectly correlated
    stats = particle_agreement(same, y)
    assert stats["error_corr"] == pytest.approx(1.0) and stats["k_eff"] == pytest.approx(1.0)
    assert stats["particle_acc"] == pytest.approx(0.5)


def test_wilson_interval():
    lo, hi = wilson_interval(50, 100)
    assert lo < 0.5 < hi and hi - lo == pytest.approx(0.192, abs=0.005)
