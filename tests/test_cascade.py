"""CPU tests of the confidence-routed cascade (src/certify/cascade.py, heads.ConfidenceRouter,
src/models/hf_classifier.py, scripts/screen_cascade.py, scripts/analyze_cascade.py and
``certify_dds --method cascade``). No score_sde / robustbench / transformers needed."""

import json

import numpy as np
import pytest
import torch

from src.certify.cascade import (cascade_nfe, cascade_predictions, choose_tau, combine, draw_metrics,
                                 head_specs, make_cascade_base)
from src.certify.denoisers import DDSDenoiser, make_base_classifier
from src.certify.heads import ClassifierHead, ConfidenceRouter

SHAPE = (3, 8, 8)


def toy_eps(x, t_vec):
    return 0.5 * torch.tanh(x) + 0.1 * torch.roll(x, 1, dims=-1)


def linear(seed, k=3, scale=3.0):
    w = torch.randn(k, int(np.prod(SHAPE)), generator=torch.Generator().manual_seed(seed)) * scale
    return lambda img: (img - 0.5).flatten(1) @ w.T


def noisy(seed=0, b=32):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(SHAPE, generator=g).unsqueeze(0) + 0.5 * torch.randn((b, *SHAPE), generator=g)


# --------------------------------------------------------------------------------------------
# torch cascade / router
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("eta", [0.0, 1.0])
def test_cascade_limits_are_onestep_and_multistep(eta):
    f = linear(0)
    x = noisy()
    ms = DDSDenoiser(toy_eps, sigma=0.5, method="multistep", steps=3, eta=eta)
    one = DDSDenoiser(toy_eps, sigma=0.5, method="onestep")
    ref_ms = make_base_classifier(ms, f, [torch.Generator().manual_seed(1)])(x)
    ref_one = make_base_classifier(one, f, [torch.Generator().manual_seed(1)])(x)
    st = {}
    hi = make_cascade_base(ms, f, f, 1.5, [torch.Generator().manual_seed(1)], st)(x)
    assert torch.equal(hi, ref_ms) and st == {"n": 32, "n_continued": 32}
    st = {}
    lo = make_cascade_base(ms, f, f, 0.0, [torch.Generator().manual_seed(1)], st)(x)
    assert torch.equal(lo, ref_one) and st["n_continued"] == 0


def test_cascade_mixes_rows_by_confidence():
    f = linear(0, scale=0.5)
    x = noisy(b=64)
    ms = DDSDenoiser(toy_eps, sigma=0.5, method="multistep", steps=3, eta=0.0)
    m01 = (ms.start(x)[2].clamp(-1, 1) + 1) * 0.5
    conf = torch.softmax(f(m01), 1).max(1).values
    tau = float(conf.median())
    st = {}
    pred = make_cascade_base(ms, f, f, tau, [None], st)(x)
    p_one = f(m01).argmax(1)
    p_ms = f(ms(x)).argmax(1)
    expect = torch.where(conf < tau, p_ms, p_one)
    assert torch.equal(pred, expect)
    assert st["n_continued"] == int((conf < tau).sum())
    assert 0 < st["n_continued"] < 64
    assert cascade_nfe(3, st["n_continued"] / st["n"]) == pytest.approx(1 + 2 * st["n_continued"] / 64)


def test_cascade_rejects_other_denoisers():
    with pytest.raises(ValueError):
        make_cascade_base(DDSDenoiser(toy_eps, sigma=0.5, method="onestep"), linear(0), linear(0), 0.5, [None])


def test_router_matches_numpy_combine_and_ensemble():
    fs = [linear(s) for s in range(3)]
    img = torch.rand(16, *SHAPE)
    P = torch.stack([torch.softmax(f(img), 1) for f in fs], dim=1).numpy()          # (B, C, K)
    route = torch.exp(ConfidenceRouter(fs)(img)).numpy()
    assert np.allclose(route, combine(P, ("route", (0, 1, 2))), atol=1e-6)
    ens = torch.exp(ClassifierHead(fs)(img)).numpy()
    assert np.allclose(ens, combine(P, ("ens", (0, 1, 2))), atol=1e-6)
    assert np.allclose(combine(P, ("single", (1,))), P[:, 1])
    with pytest.raises(ValueError):
        ConfidenceRouter(fs[:1])


# --------------------------------------------------------------------------------------------
# numpy policies
# --------------------------------------------------------------------------------------------

def test_head_specs_count():
    specs = head_specs(4)
    assert len(specs) == 4 + 2 * 11 and ("route", (0, 3)) in specs and ("ens", (0, 1, 2, 3)) in specs


def test_cascade_predictions_and_choose_tau():
    rng = np.random.RandomState(0)
    I, D, K = 20, 30, 3
    labels = rng.randint(0, K, I)
    p2 = np.eye(K)[np.repeat(labels[:, None], D, 1)] * 0.9 + 0.1 / K               # always right
    p1 = rng.dirichlet(np.ones(K), size=(I, D))                                     # mostly wrong
    pred, cont = cascade_predictions(p1, p2, 0.0)
    assert not cont.any() and np.array_equal(pred, p1.argmax(-1))
    pred, cont = cascade_predictions(p1, p2, 1.01)
    assert cont.all() and np.array_equal(pred, p2.argmax(-1))
    idx = np.arange(I)
    taus = np.linspace(0, 1.01, 50)
    assert choose_tau(p1, p2, labels, idx, 1.0, taus) == pytest.approx(taus[-1]) or \
        (cascade_predictions(p1, p2, choose_tau(p1, p2, labels, idx, 1.0, taus))[0] == labels[:, None]).all()
    assert choose_tau(p2, p2, labels, idx, 1.0, taus) == 0.0                         # first stage already perfect


def test_draw_metrics():
    labels = np.array([0, 1])
    pred = np.array([[0, 0, 0, 1], [0, 0, 1, 1]])
    m = draw_metrics(pred, labels, 0.5, 3)
    assert np.allclose(m["p_true"], [0.75, 0.5])
    assert m["top_ok"].tolist() == [True, False]
    assert m["radius"][0] > 0 and m["radius"][1] == 0


# --------------------------------------------------------------------------------------------
# HF wrapper
# --------------------------------------------------------------------------------------------

def test_hf_wrapper_preprocessing_and_label_order():
    from src.models.hf_classifier import CIFAR10_CLASSES, HFImageClassifier, label_permutation

    assert label_permutation({str(i): c for i, c in enumerate(CIFAR10_CLASSES)}, list(CIFAR10_CLASSES)) == list(range(10))
    shuffled = {str(i): c for i, c in enumerate(reversed(CIFAR10_CLASSES))}
    perm = label_permutation(shuffled, list(CIFAR10_CLASSES))
    assert perm == list(range(9, -1, -1))
    assert label_permutation({"0": "LABEL_0", "1": "LABEL_1"}, ["a", "b"]) is None

    seen = {}

    class Out:
        def __init__(self, logits):
            self.logits = logits

    class Fake(torch.nn.Module):
        def forward(self, pixel_values):
            seen["x"] = pixel_values
            return Out(torch.arange(10.0).repeat(pixel_values.shape[0], 1))

    m = HFImageClassifier(Fake(), size=16, perm=perm)
    out = m(torch.ones(2, 3, 8, 8))
    assert seen["x"].shape == (2, 3, 16, 16) and torch.allclose(seen["x"], torch.ones_like(seen["x"]))  # 2*1-1
    assert out[0].tolist() == list(range(9, -1, -1))


# --------------------------------------------------------------------------------------------
# scripts
# --------------------------------------------------------------------------------------------

def test_certify_dds_cascade_end_to_end(tmp_path, monkeypatch):
    from scripts import certify_dds as cd

    torch.manual_seed(0)
    x = torch.rand(4, *SHAPE)
    y = torch.tensor([0, 1, 2, 0])
    f = linear(0, scale=0.5)
    monkeypatch.setattr(cd, "load_eval_subset", lambda *a, **k: (x.clone(), y, np.arange(4)))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(cd, "build_models", lambda args, device: (
        DDSDenoiser(toy_eps, sigma=0.5, method="multistep", steps=3, eta=0.0), f, 3))
    common = ["--mode", "certify", "--sigma", "0.5", "--num_sub", "4", "--steps", "3", "--eta", "0",
              "--n0", "8", "--n", "40", "--batch", "16", "--exp", str(tmp_path)]
    cd.main(common + ["--method", "cascade", "--exit_tau", "2", "--run_name", "hi"])
    cd.main(common + ["--method", "multistep", "--run_name", "ms"])
    cd.main(common + ["--method", "cascade", "--exit_tau", "0.6", "--run_name", "mid"])
    hi = json.loads((tmp_path / "hi" / "summary.json").read_text())
    ms = json.loads((tmp_path / "ms" / "summary.json").read_text())
    mid = json.loads((tmp_path / "mid" / "summary.json").read_text())
    assert hi["certified_accuracy"] == ms["certified_accuracy"] and hi["acr"] == pytest.approx(ms["acr"])
    assert hi["nfe_mean"] == pytest.approx(3.0) and "nfe_mean" not in ms
    assert 1.0 <= mid["nfe_mean"] <= 3.0
    with pytest.raises(SystemExit):
        cd.parse_args(["--sigma", "0.5", "--method", "cascade"])                # tau missing


def test_run_names_with_classifier_tags():
    from scripts import certify_dds as cd

    base = ["--sigma", "0.5", "--method", "multistep", "--steps", "10", "--eta", "0", "--num_sub", "500",
            "--eval_split", "test", "--n", "10000"]
    assert cd.default_run_name(cd.parse_args(base)) == "multistep_s0.5_S10_eta0_N10000_test500_ds0"
    vit = cd.default_run_name(cd.parse_args(base + ["--classifier_name",
                                                    "hf:aaraki/vit-base-patch16-224-in21k-finetuned-cifar10"]))
    rb = cd.default_run_name(cd.parse_args(base + ["--classifier_name", "robustbench:Wang2023Better_WRN-28-10:L2"]))
    tagged = cd.default_run_name(cd.parse_args(base + ["--classifier_name", "hf:x/y", "--clf_tag", "vitb16"]))
    route = cd.default_run_name(cd.parse_args(base + ["--route_classifiers", "a,b"]))
    casc = cd.default_run_name(cd.parse_args(["--sigma", "1.0", "--method", "cascade", "--exit_tau", "0.9"]))
    assert "_clf-vit-base-patch16-224-in21k-fi" in vit and rb.endswith("_clf-Wang2023Better_WRN-28-10_N10000_test500_ds0")
    assert "_clf-vitb16_" in tagged and "_route2_" in route and casc.startswith("cascade_s1_S10_eta1_tau0.9_")
    assert len({vit, rb, tagged, route}) == 4


def test_screen_and_analyze_end_to_end(tmp_path, monkeypatch, capsys):
    from scripts import analyze_cascade as ac
    from scripts import certify_dds as cd
    from scripts import screen_cascade as sc

    torch.manual_seed(0)
    x = torch.rand(6, *SHAPE)
    y = torch.tensor([0, 1, 2, 0, 1, 2])
    fs = [linear(s, scale=1.0) for s in range(3)]
    monkeypatch.setattr(sc.loaders, "load_eval_subset", lambda *a, **k: (x.clone(), y, np.arange(10, 16)))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(cd, "build_models", lambda args, device: (
        DDSDenoiser(toy_eps, sigma=args.sigma, method="multistep", steps=args.steps, eta=args.eta), fs[0], 3))
    names = iter(fs[1:])
    monkeypatch.setattr(sc, "load_classifier", lambda name, device: next(names))
    args = ["--sigma", "0.5", "--num_sub", "6", "--n_draws", "24", "--batch", "10", "--steps_list", "2,3",
            "--classifiers", "a,b,c", "--tags", "A,B,C", "--exp", str(tmp_path), "--run_name", "r"]
    sc.main(args + ["--shard", "0/2"])
    names = iter(fs[1:])
    sc.main(args + ["--shard", "1/2"])
    files = sorted((tmp_path / "r" / "probs").glob("pos_*.npz"))
    assert len(files) == 6
    z = np.load(files[0])
    assert z["probs"].shape == (24, 3, 3, 3) and z["clean"].shape == (3, 3)
    assert np.allclose(z["probs"].astype(np.float32).sum(-1), 1.0, atol=1e-2)
    names = iter(fs[1:])
    sc.main(args + ["--shard", "0/1"])                       # resume: every image exists -> no work
    a = ac.analyze(str(tmp_path / "r"), n_boot=50)
    assert a["complete"] and a["n_images"] == 6 and a["variants"] == ["onestep", "ms2", "ms3"]
    assert len(a["policies"]) == len(head_specs(3)) * 3 and len(a["cascades"]) == len(head_specs(3))
    ref = next(r for r in a["policies"] if r["head"] == "A" and r["variant"] == "ms3")
    assert ref["d_acc"] == 0.0
    assert all(1.0 <= c["nfe"] <= 3.0 for c in a["cascades"])
    assert (tmp_path / "r" / "policies.tsv").exists() and (tmp_path / "r" / "cascade_curve.tsv").exists()
    ac.print_report(a)
    assert "GATE" in capsys.readouterr().out


def test_plain_vit_matches_transformers():
    transformers = pytest.importorskip("transformers")
    from src.models.vit import ViTForImageClassification, to_legacy_keys

    cfg = dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=64,
               image_size=32, patch_size=8, num_channels=3, layer_norm_eps=1e-12, qkv_bias=True,
               hidden_act="gelu", model_type="vit", id2label={str(i): f"c{i}" for i in range(5)})
    try:
        ref = transformers.ViTForImageClassification(transformers.ViTConfig(**cfg)).eval()
    except ImportError as e:                      # transformers without a usable torch backend
        pytest.skip(str(e))
    ours = ViTForImageClassification(cfg).eval()
    state = {k: v for k, v in to_legacy_keys(ref.state_dict()).items() if not k.startswith("vit.pooler.")}
    missing, unexpected = ours.load_state_dict(state, strict=False)
    assert not missing and not unexpected
    x = torch.randn(3, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(ours(pixel_values=x).logits, ref(pixel_values=x).logits, atol=1e-5)
