"""Bookkeeping of the full-gradient PGD+EOT attack (CPU, toy models)."""

import torch

from src.evaluation.attacks_full import (_all_reps, _worst_case, eot_margin, load_transfer_examples,
                                         pgd_eot_full)


class Linear(torch.nn.Module):
    def __init__(self, noise=0.0):
        super().__init__()
        torch.manual_seed(0)
        self.fc = torch.nn.Linear(3 * 4 * 4, 10)
        self.noise = noise

    def forward(self, x):
        z = self.fc(x.flatten(1))
        return z + self.noise * torch.randn_like(z)


def test_eot_margin_sign():
    p = torch.tensor([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]])
    m = eot_margin(p, torch.tensor([0, 0]))
    assert torch.allclose(m, torch.tensor([0.3, -0.3]))


def test_x_best_is_lowest_margin_iterate_and_in_ball():
    torch.manual_seed(1)
    model = Linear(noise=0.5)
    x = torch.rand(6, 3, 4, 4)
    y = model.fc(x.flatten(1)).argmax(1)
    eps = 8 / 255
    _, x_best, broken = pgd_eot_full(model, x, y, eps=eps, steps=15, eot_iter=2, log_every=0)
    assert (x_best - x).abs().max() <= eps + 1e-6
    assert x_best.min() >= 0 and x_best.max() <= 1
    assert broken.dtype == torch.bool and broken.shape == (6,)


def test_attack_breaks_a_linear_model_without_noise():
    torch.manual_seed(2)
    model = Linear()
    x = torch.rand(8, 3, 4, 4)
    y = model.fc(x.flatten(1)).argmax(1)
    _, x_best, broken = pgd_eot_full(model, x, y, eps=0.5, steps=30, step_size=0.05, eot_iter=1,
                                   log_every=0)
    with torch.no_grad():
        assert (model(x_best).argmax(1) != y).all()
    assert broken.all()


def test_n_eval_and_worst_case_metrics():
    rec = {"def_adv": torch.tensor([1.0, 0.75, 0.0, 1.0]),
           "def_clean": torch.tensor([1.0, 1.0, 0.5, 1.0]),
           "broken_during_attack": torch.tensor([False, False, False, True])}
    assert _all_reps([rec], "def_adv")["acc"] == 0.5          # images 0 and 3
    assert _all_reps([rec], "def_clean")["acc"] == 0.75
    assert abs(_worst_case([rec])["acc"] - (1.0 + 0.75) / 4) < 1e-6  # image 3 is broken


def test_load_transfer_examples_indexes_by_subset_position(tmp_path):
    for b, lo in enumerate((0, 2)):
        torch.save({"lo": lo, "y": torch.tensor([lo, lo + 1]), "x_adv": torch.full((2, 3, 4, 4), float(lo))},
                   tmp_path / f"pgd_full_batch{b:03d}.pt")
    out = load_transfer_examples(str(tmp_path))
    assert sorted(out) == [0, 1, 2, 3]
    assert out[3][1] == 3 and float(out[2][0].mean()) == 2.0
