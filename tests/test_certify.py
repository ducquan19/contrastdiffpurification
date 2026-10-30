"""CPU tests of src/certify (randomized smoothing, DDS / WA-DDS denoisers, spectrum) and of the
summaries of scripts/certify_dds.py. No score_sde / robustbench needed."""

import csv
import math
import os

import numpy as np
import pytest
import torch
from scipy.stats import norm

from src.certify.denoisers import DIAG_VARIANTS, DDSDenoiser, sigma_to_time, spectral_blend
from src.certify.smoothing import (ABSTAIN, average_certified_radius, certified_accuracy,
                                   certified_radius, certify, clopper_pearson_lower,
                                   image_generator, plugin_radius)
from src.certify.spectrum import power_spectrum, snr_table, wiener_gain
from src.models.diffusion.sampler import alpha_bar_cont

SHAPE = (3, 8, 8)


def oracle_eps(x_star):
    """eps of a point-mass prior at x_star (in [-1, 1]): every denoiser must return x_star."""
    def eps_fn(x, t_vec):
        ab = alpha_bar_cont(t_vec).to(x).view(-1, 1, 1, 1)
        return (x - ab.sqrt() * x_star) / (1 - ab).sqrt()
    return eps_fn


def toy_eps(x, t_vec):
    """A non-trivial, input-dependent noise prediction (not a real score)."""
    return 0.5 * torch.tanh(x) + 0.1 * torch.roll(x, 1, dims=-1)


# --------------------------------------------------------------------------------------------
# smoothing
# --------------------------------------------------------------------------------------------

def test_clopper_pearson_all_successes_closed_form():
    for n, alpha in [(1000, 0.001), (100, 0.01), (10000, 0.001)]:
        assert clopper_pearson_lower(n, n, alpha) == pytest.approx(alpha ** (1.0 / n), rel=1e-9)
    assert clopper_pearson_lower(0, 10, 0.001) == 0.0


def test_clopper_pearson_is_below_the_estimate_and_monotone():
    vals = [clopper_pearson_lower(k, 1000, 0.001) for k in (500, 700, 900, 990)]
    assert all(v < k / 1000 for v, k in zip(vals, (500, 700, 900, 990)))
    assert vals == sorted(vals)


def test_certified_radius():
    assert certified_radius(0.5, 0.25) == 0.0
    assert certified_radius(0.3, 0.25) == 0.0
    assert certified_radius(float(norm.cdf(1.0)), 0.5) == pytest.approx(0.5)


def test_certify_constant_classifier():
    x = torch.rand(SHAPE)
    base = lambda xb: torch.full((xb.shape[0],), 3, dtype=torch.long)
    r = certify(base, x, sigma=0.5, n0=10, n=1000, alpha=0.001, n_classes=10, batch=128,
                generator=image_generator(0, 0, "cpu"))
    assert r.predict == 3
    assert r.p_lower == pytest.approx(0.001 ** (1 / 1000))
    assert r.radius == pytest.approx(0.5 * norm.ppf(0.001 ** (1 / 1000)))
    assert r.counts_estimate.sum() == 1000 and r.counts_select.sum() == 10


def test_certify_abstains_on_a_coin_flip():
    x = torch.zeros(SHAPE)
    # label = sign of the first noise coordinate: P = 1/2 for both classes
    base = lambda xb: (xb[:, 0, 0, 0] > 0).long()
    r = certify(base, x, sigma=1.0, n0=50, n=500, alpha=0.001, n_classes=2, batch=64,
                generator=image_generator(1, 2, "cpu"))
    assert r.predict == ABSTAIN and r.radius == 0.0


def test_image_generator_is_reproducible_and_index_dependent():
    a = torch.randn(5, generator=image_generator(0, 7, "cpu"))
    b = torch.randn(5, generator=image_generator(0, 7, "cpu"))
    c = torch.randn(5, generator=image_generator(0, 8, "cpu"))
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_certified_accuracy_and_acr():
    labels, predicts, radii = [0, 1, 2, 3], [0, 1, ABSTAIN, 0], [0.6, 0.2, 0.0, 0.9]
    ca = certified_accuracy(labels, predicts, radii, at=[0.0, 0.25, 0.5])
    assert ca == {0.0: 0.5, 0.25: 0.25, 0.5: 0.25}
    assert average_certified_radius(labels, predicts, radii) == pytest.approx((0.6 + 0.2) / 4)


def test_plugin_radius_is_finite_at_one():
    assert plugin_radius(1.0, 0.5, 100) == pytest.approx(0.5 * norm.ppf(1 - 0.005))
    assert plugin_radius(0.4, 0.5, 100) == 0.0


# --------------------------------------------------------------------------------------------
# denoisers
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("sigma", [0.12, 0.25, 0.5, 1.0])
def test_sigma_to_time_matches_alpha_bar(sigma):
    t = sigma_to_time(sigma)
    assert float(alpha_bar_cont(t)) == pytest.approx(1 / (1 + (2 * sigma) ** 2), rel=1e-9)


def test_spectral_blend_limits():
    a, b = torch.randn(2, *SHAPE), torch.randn(2, *SHAPE)
    assert torch.allclose(spectral_blend(a, b, torch.ones(SHAPE)), b, atol=1e-5)
    assert torch.allclose(spectral_blend(a, b, torch.zeros(SHAPE)), a, atol=1e-6)


@pytest.mark.parametrize("method", ["onestep", "multistep", "wadds"])
def test_every_method_recovers_a_point_mass(method):
    x_star = torch.rand(1, *SHAPE) * 1.6 - 0.8
    gain = torch.rand(SHAPE)
    den = DDSDenoiser(oracle_eps(x_star), sigma=0.5, method=method, steps=5, eta=1.0, gain=gain)
    x01 = (x_star[0] + 1) / 2
    noisy = x01.unsqueeze(0) + 0.5 * torch.randn(4, *SHAPE)
    out = den(noisy, torch.Generator().manual_seed(0))
    assert torch.allclose(out, x01.expand_as(out), atol=1e-3)


def test_wadds_with_zero_gain_is_multistep():
    noisy = torch.rand(3, *SHAPE) + 0.5 * torch.randn(3, *SHAPE)
    kw = dict(sigma=0.5, steps=6, eta=1.0)
    ms = DDSDenoiser(toy_eps, method="multistep", **kw)(noisy, torch.Generator().manual_seed(1))
    wz = DDSDenoiser(toy_eps, method="wadds", gain=torch.zeros(SHAPE), **kw)(noisy, torch.Generator().manual_seed(1))
    wg = DDSDenoiser(toy_eps, method="wadds", gain=torch.ones(SHAPE), gamma=0.0, **kw)(noisy, torch.Generator().manual_seed(1))
    assert torch.allclose(ms, wz, atol=1e-6) and torch.allclose(ms, wg, atol=1e-6)


def test_wadds_with_unit_gain_is_the_posterior_mean():
    noisy = torch.rand(3, *SHAPE) + 0.25 * torch.randn(3, *SHAPE)
    one = DDSDenoiser(toy_eps, sigma=0.25, method="onestep")(noisy)
    w1 = DDSDenoiser(toy_eps, sigma=0.25, method="wadds", steps=8, gain=torch.ones(SHAPE))(
        noisy, torch.Generator().manual_seed(2))
    assert torch.allclose(one, w1, atol=1e-5)


def test_anchor_window_outside_the_trajectory_is_multistep():
    noisy = torch.rand(2, *SHAPE)
    kw = dict(sigma=0.5, steps=4, eta=1.0)
    ms = DDSDenoiser(toy_eps, method="multistep", **kw)(noisy, torch.Generator().manual_seed(3))
    off = DDSDenoiser(toy_eps, method="wadds", gain=torch.ones(SHAPE), anchor_window=(2.0, 3.0), **kw)(
        noisy, torch.Generator().manual_seed(3))
    assert torch.allclose(ms, off, atol=1e-6)


def test_diagnose_variants_share_randomness():
    noisy = torch.rand(4, *SHAPE)
    gain = torch.rand(SHAPE)
    kw = dict(sigma=0.5, steps=5, eta=1.0, gain=gain)
    d = DDSDenoiser(toy_eps, method="wadds", **kw).diagnose(noisy, torch.Generator().manual_seed(4))
    assert set(d) == set(DIAG_VARIANTS)
    assert all(v.shape == noisy.shape and v.min() >= 0 and v.max() <= 1 for v in d.values())
    ms = DDSDenoiser(toy_eps, method="multistep", **kw)(noisy, torch.Generator().manual_seed(4))
    wa = DDSDenoiser(toy_eps, method="wadds", **kw)(noisy, torch.Generator().manual_seed(4))
    one = DDSDenoiser(toy_eps, method="onestep", **kw)(noisy)
    assert torch.allclose(d["multistep"], ms, atol=1e-6)
    assert torch.allclose(d["wadds"], wa, atol=1e-6)
    assert torch.allclose(d["onestep"], one, atol=1e-6)


def test_denoiser_rows_are_batch_independent():
    """A row's output depends only on its own noisy input and noise (no cross-row coupling)."""
    noisy = torch.rand(4, *SHAPE)
    den = DDSDenoiser(toy_eps, sigma=0.5, method="wadds", steps=3, eta=0.0, gain=torch.rand(SHAPE))
    full = den(noisy)                 # eta = 0: deterministic
    part = den(noisy[1:2])
    assert torch.allclose(full[1:2], part, atol=1e-6)


def test_bad_arguments():
    with pytest.raises(ValueError):
        DDSDenoiser(toy_eps, sigma=0.5, method="wadds")          # no gain
    with pytest.raises(ValueError):
        DDSDenoiser(toy_eps, sigma=0.5, method="nope")
    with pytest.raises(ValueError):
        DDSDenoiser(toy_eps, sigma=0.5, method="onestep", gamma=1.5)


# --------------------------------------------------------------------------------------------
# spectrum
# --------------------------------------------------------------------------------------------

def test_power_spectrum_of_a_constant_image_is_dc_only():
    c = 0.75
    p = power_spectrum([torch.full((5, 3, 8, 8), c)])
    expected_dc = (2 * c - 1) ** 2 * 64
    assert p[:, 0, 0] == pytest.approx(torch.full((3,), expected_dc), rel=1e-5)
    p[:, 0, 0] = 0
    assert float(p.abs().max()) < 1e-6


def test_white_noise_power_matches_the_noise_floor():
    torch.manual_seed(0)
    sd = 0.3                                  # pixel std on [-1, 1] -> 0.15 on [0, 1]
    imgs = 0.5 + 0.5 * sd * torch.randn(4000, 1, 8, 8)
    p = power_spectrum([imgs])
    p[:, 0, 0] = sd ** 2                      # DC carries the mean, skip it
    assert float(p.mean()) == pytest.approx(sd ** 2, rel=0.05)


def test_wiener_gain_range_and_modes():
    power = torch.rand(3, 8, 8) * 10
    w = wiener_gain(power, 0.5)
    assert w.shape == (3, 8, 8) and float(w.min()) >= 0 and float(w.max()) < 1
    wp = wiener_gain(power, 0.5, "pooled")
    assert wp.shape == (1, 8, 8)
    assert torch.all(wiener_gain(power, 1.0) <= w + 1e-7)      # more noise -> smaller gain
    rows = snr_table(power, [0.25, 1.0])
    assert rows[0]["pct_bins_snr_gt1"] >= rows[1]["pct_bins_snr_gt1"]


# --------------------------------------------------------------------------------------------
# scripts/certify_dds.py summaries (no models)
# --------------------------------------------------------------------------------------------

def _write(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def test_summaries_from_shard_files(tmp_path):
    from scripts import certify_dds as cd

    args = cd.parse_args(["--mode", "certify", "--sigma", "0.5", "--num_sub", "4"])
    rows = [dict(pos=p, idx=10 + p, label=1, predict=pred, correct=int(pred == 1), p_lower=0.9,
                 radius=rad, count_cA=900, seconds=1.0)
            for p, pred, rad in [(0, 1, 0.6), (1, 1, 0.3), (2, ABSTAIN, 0.0), (3, 2, 0.8)]]
    _write(tmp_path / "certify_shard0of2.tsv", cd.CERT_FIELDS, rows[:2])
    _write(tmp_path / "certify_shard1of2.tsv", cd.CERT_FIELDS, rows[2:])
    s = cd.summarize_certify(str(tmp_path), args)
    assert s["complete"] and s["n_images"] == 4
    assert s["certified_accuracy"][0.0] == 0.5 and s["certified_accuracy"][0.5] == 0.25
    assert s["acr"] == pytest.approx(0.9 / 4)

    dargs = cd.parse_args(["--mode", "diagnose", "--sigma", "0.5", "--num_sub", "2", "--n_boot", "50"])
    n_true = {"onestep": (60, 40), "multistep": (55, 45), "posthoc": (70, 50),
              "posthoc_rev": (50, 30), "wadds": (75, 55)}
    drows = [dict(pos=p, idx=p, label=0, variant=v, n=100, n_true=t[p], top_class=0 if t[p] > 50 else 1,
                  n_top=max(t[p], 100 - t[p])) for v, t in n_true.items() for p in (0, 1)]
    _write(tmp_path / "diag_shard0of1.tsv", cd.DIAG_FIELDS, drows)
    ds = cd.summarize_diagnose(str(tmp_path), dargs)
    assert ds["variants"]["wadds"]["acc_per_draw"] == pytest.approx(0.65)
    assert ds["variants"]["wadds"]["d_acc_per_draw_vs_onestep"] == pytest.approx(0.15)
    assert ds["gate_G0"]["pass"] is True


@pytest.mark.parametrize("mode", ["certify", "diagnose"])
def test_main_end_to_end_with_toy_models(tmp_path, monkeypatch, mode):
    """main(): shard files, --resume, summary.json, with the models replaced by CPU toys."""
    import json

    from scripts import certify_dds as cd

    torch.manual_seed(0)
    x = torch.rand(6, *SHAPE)
    y = torch.tensor([0, 1, 2, 0, 1, 2])
    monkeypatch.setattr(cd, "load_eval_subset",
                        lambda *a, **k: (x.clone().requires_grad_(True), y, np.arange(100, 106)))
    weight = torch.randn(3, int(np.prod(SHAPE)))
    classifier = lambda img: img.flatten(1) @ weight.T

    def fake_build(args, device):
        den = DDSDenoiser(toy_eps, sigma=args.sigma, method=args.method if args.mode == "certify" else "wadds",
                          steps=args.steps, gain=torch.rand(SHAPE))
        return den, classifier, 3

    monkeypatch.setattr(cd, "build_models", fake_build)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)   # CPU toys, even on a GPU box
    common = ["--mode", mode, "--sigma", "0.25", "--num_sub", "6", "--steps", "2", "--n0", "10",
              "--n", "40", "--n_diag", "20", "--batch", "16", "--exp", str(tmp_path), "--run_name", "r",
              "--n_boot", "20"]
    cd.main(common + ["--shard", "0/2"])
    with pytest.raises(FileExistsError):
        cd.main(common + ["--shard", "0/2"])
    cd.main(common + ["--shard", "0/2", "--resume"])          # nothing left: no duplicates
    cd.main(common + ["--shard", "1/2"])
    s = json.loads((tmp_path / "r" / "summary.json").read_text())
    assert s["complete"] and s["n_images"] == 6
    prefix = "certify" if mode == "certify" else "diag"
    rows = cd.read_rows(str(tmp_path / "r"), prefix)
    per_image = 1 if mode == "certify" else len(DIAG_VARIANTS)
    assert len(rows) == 6 * per_image
    assert (tmp_path / "r" / "resolved_config_shard1of2.json").exists()
    (tmp_path / "r" / "summary.json").unlink()
    cd.main(["--summarize_dir", str(tmp_path / "r")])
    assert json.loads((tmp_path / "r" / "summary.json").read_text()) == s


def test_compare_dds_paired_statistics(tmp_path, capsys):
    import json

    from scripts import certify_dds as cd
    from scripts import compare_dds as cmp

    run = tmp_path / "diag"
    run.mkdir()
    (run / "resolved_config_shard0of1.json").write_text(json.dumps({"sigma": 0.5}))
    n_true = {"onestep": [40, 60, 70], "multistep": [50, 60, 80], "posthoc": [50, 61, 80],
              "posthoc_rev": [40, 60, 70], "wadds": [60, 70, 90]}
    rows = [dict(pos=p, idx=p, label=0, variant=v, n=100, n_true=t[p], top_class=0 if t[p] > 50 else 1,
                 n_top=max(t[p], 100 - t[p])) for v, t in n_true.items() for p in range(3)]
    _write(run / "diag_shard0of1.tsv", cd.DIAG_FIELDS, rows)
    v = cmp.load_diag(str(run), 0.5)
    c = cmp.compare_variants(v["wadds"], v["multistep"])
    assert c["n"] == 3 and c["d_acc"] == pytest.approx(0.10)
    assert (c["wins"], c["losses"]) == (1, 0)          # image 0: 60 vs 50 -> only wadds is correct
    assert c["d_acr"] > 0
    cmp.main(["diag", str(run)])
    assert "wadds - multistep" in capsys.readouterr().out

    a, b = tmp_path / "ref", tmp_path / "new"
    for d, preds in ((a, [(1, 0.2), (-1, 0.0), (1, 0.6)]), (b, [(1, 0.4), (1, 0.3), (1, 0.6)])):
        d.mkdir()
        _write(d / "certify_shard0of1.tsv", cd.CERT_FIELDS,
               [dict(pos=p, idx=p, label=1, predict=pr, correct=int(pr == 1), p_lower=0.9, radius=r,
                     count_cA=1, seconds=1.0) for p, (pr, r) in enumerate(preds)])
    cmp.main(["certify", str(a), str(b), "--radii", "0,0.25"])
    out = capsys.readouterr().out
    assert "n=3 common images" in out and "W/L=1/0" in out


def test_default_run_names_are_distinct():
    from scripts import certify_dds as cd

    names = {cd.default_run_name(cd.parse_args(a)) for a in (
        ["--mode", "diagnose", "--sigma", "0.5"],
        ["--mode", "certify", "--method", "wadds", "--sigma", "0.5"],
        ["--mode", "certify", "--method", "onestep", "--sigma", "0.5"],
        ["--mode", "certify", "--method", "multistep", "--sigma", "0.5"],
        ["--mode", "certify", "--method", "wadds", "--sigma", "0.5", "--gamma", "0.5"],
        ["--mode", "certify", "--method", "wadds", "--sigma", "0.25"],
    )}
    assert len(names) == 6


# --------------------------------------------------------------------------------------------
# classifier heads and classifier-consistency guidance
# --------------------------------------------------------------------------------------------

def _linear_classifier(seed=0, n_classes=3):
    w = torch.randn(n_classes, int(np.prod(SHAPE)), generator=torch.Generator().manual_seed(seed))
    return lambda img: (img.flatten(1) - 0.5) @ w.T


def test_heads_average_softmax_and_flip():
    from src.certify.heads import ClassifierHead, build_heads, certify_head_key

    f1, f2 = _linear_classifier(0), _linear_classifier(1)
    x = torch.rand(4, *SHAPE)
    h = ClassifierHead([f1, f2])
    expected = (torch.softmax(f1(x), 1) + torch.softmax(f2(x), 1)) / 2
    assert torch.allclose(h(x).exp(), expected, atol=1e-6)
    hf = ClassifierHead([f1], flip=True)
    assert torch.allclose(hf(x), hf(torch.flip(x, dims=[-1])), atol=1e-6)     # flip-invariant
    assert set(build_heads(f1)) == {""}
    assert set(build_heads(f1, [f2], flip=True)) == {"", "+flip", "+ens", "+ens+flip"}
    assert certify_head_key(True, True) == "+ens+flip" and certify_head_key(False, False) == ""


def test_guidance_zero_lambda_or_closed_gate_is_multistep():
    noisy = torch.rand(3, *SHAPE)
    kw = dict(sigma=0.5, steps=5, eta=0.0)
    ms = DDSDenoiser(toy_eps, method="multistep", **kw)(noisy)
    for cfg in (dict(lam=0.0), dict(lam=2.0, conf_min=1.01)):
        d = DDSDenoiser(toy_eps, method="multistep", **kw)
        d.set_guidance(_linear_classifier(), t_frac=1.0, **cfg)
        assert torch.allclose(d(noisy), ms, atol=1e-6)


def test_guidance_raises_the_log_prob_of_the_predicted_class():
    """One DDIM step: the guided x0 = m + lam * unit gradient must raise log p(y_hat)."""
    f = _linear_classifier()
    noisy = 0.5 + 0.05 * torch.randn(6, *SHAPE)
    plain = DDSDenoiser(toy_eps, sigma=0.25, method="multistep", steps=1, eta=0.0)(noisy)
    d = DDSDenoiser(toy_eps, sigma=0.25, method="multistep", steps=1, eta=0.0)
    d.set_guidance(f, lam=0.05, t_frac=1.0)
    guided = d(noisy)
    lp0, lp1 = torch.log_softmax(f(plain), 1), torch.log_softmax(f(guided), 1)
    y = lp0.argmax(1)
    assert torch.all(lp1.gather(1, y[:, None]) > lp0.gather(1, y[:, None]))
    assert float((guided - plain).flatten(1).norm(dim=1).max()) <= 0.05 / 2 + 1e-5   # [0,1] scale = half


def test_guidance_lowpass_removes_high_frequencies():
    from src.guidance.spectral import radial_frequency

    noisy = 0.5 + 0.05 * torch.randn(4, *SHAPE)
    plain = DDSDenoiser(toy_eps, sigma=0.25, method="multistep", steps=1, eta=0.0)(noisy)
    d = DDSDenoiser(toy_eps, sigma=0.25, method="multistep", steps=1, eta=0.0)
    d.set_guidance(_linear_classifier(), lam=0.05, t_frac=1.0, lowpass=0.5)
    delta = torch.fft.fft2(d(noisy) - plain, norm="ortho").abs() ** 2
    r = radial_frequency(SHAPE[1], SHAPE[2])
    assert float(delta[..., r > 2].sum()) < 1e-3 * float(delta.sum())


def test_guided_variant_and_argument_checks():
    noisy = torch.rand(2, *SHAPE)
    d = DDSDenoiser(toy_eps, sigma=0.5, method="wadds", steps=3, eta=0.0, gain=torch.rand(SHAPE))
    with pytest.raises(ValueError):
        d.diagnose(noisy, variants=DIAG_VARIANTS + ("guided",))       # not configured
    d.set_guidance(_linear_classifier(), lam=0.1, t_frac=0.5)
    out = d.diagnose(noisy, torch.Generator().manual_seed(0), variants=DIAG_VARIANTS + ("guided",))
    assert "guided" in out and out["guided"].shape == noisy.shape
    with pytest.raises(ValueError):
        DDSDenoiser(toy_eps, sigma=0.5, method="onestep").set_guidance(_linear_classifier(), lam=1.0)
    with pytest.raises(ValueError):
        d.set_guidance(_linear_classifier(), lam=1.0, t_frac=0.0)


def test_diagnose_end_to_end_with_heads_and_guidance(tmp_path, monkeypatch):
    import json

    from scripts import certify_dds as cd
    from scripts import compare_dds as cmp
    from src.certify.heads import build_heads

    x = torch.rand(4, *SHAPE)
    y = torch.tensor([0, 1, 2, 0])
    monkeypatch.setattr(cd, "load_eval_subset", lambda *a, **k: (x.clone(), y, np.arange(10, 14)))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    f1, f2 = _linear_classifier(0), _linear_classifier(1)

    def fake_build(args, device):
        den = DDSDenoiser(toy_eps, sigma=args.sigma, method="wadds", steps=args.steps, eta=args.eta,
                          gain=torch.rand(SHAPE))
        if args.guide_lambda > 0:
            den.set_guidance(f1, lam=args.guide_lambda, t_frac=args.guide_t_frac)
        return den, build_heads(f1, [f2] if args.ens_list else [], flip=args.tta_flip), 3

    monkeypatch.setattr(cd, "build_models", fake_build)
    argv = ["--mode", "diagnose", "--sigma", "0.5", "--num_sub", "4", "--steps", "2", "--eta", "0",
            "--n_diag", "10", "--batch", "5", "--exp", str(tmp_path), "--n_boot", "20",
            "--guide_lambda", "0.1", "--ens_classifiers", "toy2", "--tta_flip"]
    cd.main(argv)
    name = cd.default_run_name(cd.parse_args(argv))
    assert "_gl0.1_gt0.3" in name and "_ens1" in name and name.endswith("_flip_dev4_ds0")
    s = json.loads((tmp_path / name / "summary.json").read_text())
    expected = set(DIAG_VARIANTS) | {"guided"} | {f"{v}{k}" for v in ("multistep", "guided")
                                                  for k in ("+flip", "+ens", "+ens+flip")}
    assert set(s["variants"]) == expected and s["complete"]
    pairs = cmp.variant_pairs(set(s["variants"]))
    assert ("guided+ens", "guided") in pairs and ("guided+ens", "multistep") in pairs
    assert ("multistep+flip", "multistep") in pairs and pairs.count(("multistep+flip", "multistep")) == 1


def test_certify_uses_the_selected_head(tmp_path, monkeypatch):
    import json

    from scripts import certify_dds as cd
    from src.certify.heads import build_heads

    x = torch.rand(2, *SHAPE)
    monkeypatch.setattr(cd, "load_eval_subset", lambda *a, **k: (x.clone(), torch.tensor([0, 1]), np.arange(2)))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    used = []
    f1 = _linear_classifier(0)

    def spy(img):
        used.append(img.shape[0])
        return f1(img)

    monkeypatch.setattr(cd, "build_models", lambda args, device: (
        DDSDenoiser(toy_eps, sigma=0.5, method="multistep", steps=2, eta=0.0),
        {"": f1, "+flip": spy}, 3))
    cd.main(["--mode", "certify", "--method", "multistep", "--sigma", "0.5", "--num_sub", "2", "--steps", "2",
             "--eta", "0", "--n0", "4", "--n", "8", "--batch", "4", "--exp", str(tmp_path), "--run_name", "r",
             "--tta_flip"])
    assert sum(used) == 2 * (4 + 8)
    assert json.loads((tmp_path / "r" / "summary.json").read_text())["complete"]
