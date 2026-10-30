"""Certified l2 robustness of diffusion denoised smoothing: WA-DDS and its baselines (docs/wadds.md).

Two modes, both on a fixed image subset (``--eval_split`` / ``--num_sub`` / ``--data_seed``):

``--mode diagnose``  gate G0. For every image, ``--n_diag`` noisy copies are denoised ONCE with
                     shared randomness and classified under five variants (one-shot mean m,
                     unanchored sample s, post-hoc band blends of m and s, WA-DDS). Records the
                     frequency of the true class per variant; no confidence bound.
``--mode certify``   Cohen et al.'s CERTIFY with ``--method`` (onestep | multistep | wadds).

Results go to ``outputs/<exp>/<run_name>/`` as one TSV per shard (``--shard i/n``), appended image
by image, so a killed job is resumed with the same command plus ``--resume``. ``summary.json`` is
recomputed from every shard file at the end of each job (or with ``--summarize_only``).

Example (gate G0, then G1 on 100 dev images, sigma = 0.5):

    python -m scripts.spectral_stats --domain cifar10
    python -m scripts.certify_dds --mode diagnose --sigma 0.5 --num_sub 100 --n_diag 100
    python -m scripts.certify_dds --mode certify --method wadds --sigma 0.5 --num_sub 100 --n 1000
"""

import argparse
import csv
import glob
import json
import logging
import os
import time

import numpy as np
import torch

from src.certify.denoisers import (DIAG_VARIANTS, GUIDE_BASES, GUIDED, METHODS, DDSDenoiser,
                                   make_base_classifier)
from src.certify.heads import build_heads, certify_head_key
from src.certify.smoothing import (DEFAULT_RADII, certify, image_generator, plugin_radius,
                                   summarize)
from src.certify.spectrum import POWER_MODES, load_power, wiener_gain
from src.data.loaders import EVAL_SPLITS, load_eval_subset
from src.data.specs import DATASET_NAMES, get_dataset_spec
from src.paths import output_path, pretrained_path
from src.utils import setup_logging

logger = logging.getLogger(__name__)

CERT_FIELDS = ["pos", "idx", "label", "predict", "correct", "p_lower", "radius", "count_cA", "seconds"]
DIAG_FIELDS = ["pos", "idx", "label", "variant", "n", "n_true", "top_class", "n_top"]


# --------------------------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--mode", choices=("certify", "diagnose"), default="certify")
    p.add_argument("--method", choices=METHODS, default="wadds", help="denoiser for --mode certify")
    p.add_argument("--config", default="cifar10.yml", help="score-SDE config in configs/")
    p.add_argument("--domain", default="cifar10", choices=DATASET_NAMES)
    p.add_argument("--classifier_name", default="cifar10-wideresnet-28-10")
    p.add_argument("--sigma", type=float, required=True, help="smoothing noise, [0, 1] pixel scale")
    # denoiser
    p.add_argument("--steps", type=int, default=10, help="DDIM steps = NFE per noisy sample")
    p.add_argument("--eta", type=float, default=1.0, help="DDIM stochasticity")
    p.add_argument("--gamma", type=float, default=1.0, help="anchor strength in [0, 1]")
    p.add_argument("--anchor_t_min", type=float, default=0.0, help="anchor window start (fraction of t_sigma)")
    p.add_argument("--anchor_t_max", type=float, default=1.0, help="anchor window end (fraction of t_sigma)")
    p.add_argument("--power_path", default=None, help="default: pretrained/spectral/<domain>_power.pt")
    p.add_argument("--power_mode", choices=POWER_MODES, default="channel")
    p.add_argument("--t_end", type=float, default=1e-5)
    # classifier heads (training-free): ensemble of off-the-shelf classifiers, flip TTA
    p.add_argument("--ens_classifiers", default="", help="comma-separated extra classifier names (ensemble)")
    p.add_argument("--tta_flip", action="store_true", help="average the prediction with the mirrored image")
    # classifier-consistency guidance (src/certify/denoisers.py)
    p.add_argument("--guide_lambda", type=float, default=0.0, help="l2 step per guided step; 0 = off")
    p.add_argument("--guide_t_frac", type=float, default=0.3, help="guide steps with t <= frac * t_sigma")
    p.add_argument("--guide_conf_min", type=float, default=0.0, help="guide only rows with max prob >= this")
    p.add_argument("--guide_lowpass", type=float, default=0.0, help="Gaussian low-pass of the gradient (cycles); 0 = off")
    p.add_argument("--guide_base", choices=GUIDE_BASES, default="multistep", help="trajectory of the guided variant")
    # data
    p.add_argument("--eval_split", choices=EVAL_SPLITS, default="dev")
    p.add_argument("--num_sub", type=int, default=100)
    p.add_argument("--data_seed", type=int, default=0)
    # smoothing
    p.add_argument("--n0", type=int, default=100, help="CERTIFY selection samples")
    p.add_argument("--n", type=int, default=1000, help="CERTIFY estimation samples")
    p.add_argument("--alpha", type=float, default=0.001)
    p.add_argument("--n_diag", type=int, default=100, help="noisy copies per image in --mode diagnose")
    p.add_argument("--batch", type=int, default=125, help="noisy copies denoised at once (<= 125 fits a V100-32GB)")
    p.add_argument("--seed", type=int, default=0, help="seed of the smoothing / DDIM noise")
    p.add_argument("--radii", default=",".join(str(r) for r in DEFAULT_RADII))
    # bookkeeping
    p.add_argument("--exp", default="./wadds", help="experiment folder under outputs/")
    p.add_argument("--run_name", default=None, help="default: derived from the settings")
    p.add_argument("--shard", default="0/1", help="i/n: this job takes images with pos % n == i")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--summarize_only", action="store_true")
    p.add_argument("--n_boot", type=int, default=2000, help="bootstrap resamples (diagnose)")
    p.add_argument("--verbose", default="info")
    args = p.parse_args(argv)
    i, n = (int(v) for v in args.shard.split("/"))
    if not 0 <= i < n:
        p.error(f"bad --shard {args.shard}")
    args.shard_i, args.shard_n = i, n
    args.radii_list = [float(r) for r in args.radii.split(",")]
    args.ens_list = [c for c in args.ens_classifiers.split(",") if c]
    if args.mode == "certify" and args.method == "onestep" and args.guide_lambda > 0:
        p.error("--guide_lambda needs --method multistep or wadds")
    return args


def default_run_name(args) -> str:
    base = f"s{args.sigma:g}"
    if args.mode == "diagnose":
        name = f"diag_{base}_S{args.steps}_eta{args.eta:g}_g{args.gamma:g}"
    elif args.method == "onestep":
        name = f"onestep_{base}"
    elif args.method == "multistep":
        name = f"multistep_{base}_S{args.steps}_eta{args.eta:g}"
    else:
        name = f"wadds_{base}_S{args.steps}_eta{args.eta:g}_g{args.gamma:g}"
    if args.mode == "diagnose" or args.method == "wadds":
        if (args.anchor_t_min, args.anchor_t_max) != (0.0, 1.0):
            name += f"_w{args.anchor_t_min:g}-{args.anchor_t_max:g}"
        if args.power_mode != "channel":
            name += f"_{args.power_mode}"
    if getattr(args, "guide_lambda", 0.0) > 0:
        name += f"_gl{args.guide_lambda:g}_gt{args.guide_t_frac:g}"
        if args.guide_conf_min > 0:
            name += f"_gc{args.guide_conf_min:g}"
        if args.guide_lowpass > 0:
            name += f"_glp{args.guide_lowpass:g}"
        if args.guide_base != "multistep":
            name += f"_gb{args.guide_base}"
    if getattr(args, "ens_list", []):
        name += f"_ens{len(args.ens_list)}"
    if getattr(args, "tta_flip", False):
        name += "_flip"
    if args.mode == "certify":
        name += f"_N{args.n}"
    return f"{name}_{args.eval_split}{args.num_sub}_ds{args.data_seed}"


# --------------------------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------------------------

def _frozen(model, device):
    model = model.to(device).eval()
    for q in model.parameters():
        q.requires_grad_(False)
    return model


def build_models(args, device):
    """(denoiser, heads, n_classes); imports score_sde / robustbench lazily.

    ``heads`` maps a variant suffix to a classifier head (src/certify/heads.py); ``heads['']`` is
    the main classifier, which also drives the guidance.
    """
    from src.config import load_config
    from src.models.classifiers import get_image_classifier
    from src.models.diffusion.base import VPDynamics, load_score_model
    from src.models.diffusion.sampler import alpha_bar_cont

    spec = get_dataset_spec(args.domain)
    classifier = _frozen(get_image_classifier(args.classifier_name), device)
    extra = [_frozen(get_image_classifier(name), device) for name in args.ens_list]
    heads = build_heads(classifier, extra, flip=args.tta_flip)
    if extra or args.tta_flip:
        logger.info("classifier heads: %s (ensemble: %s)", sorted(heads), [args.classifier_name, *args.ens_list])

    config = load_config(args.config)
    config.device = device
    score_model = load_score_model(config, device)
    for q in score_model.parameters():
        q.requires_grad_(False)
    dyn = VPDynamics(score_model, img_shape=spec.img_shape).to(device)

    def eps_fn(x, t_vec):
        score = dyn.score(t_vec, x.flatten(1)).view_as(x)
        return -score * torch.sqrt(1 - alpha_bar_cont(t_vec)).to(x).view(-1, 1, 1, 1)

    gain = None
    needs_gain = args.mode == "diagnose" or args.method == "wadds"
    if needs_gain:
        path = args.power_path or pretrained_path("spectral", f"{args.domain}_power.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"power spectrum {path} not found; run scripts/spectral_stats.py first")
        gain = wiener_gain(load_power(path), args.sigma, args.power_mode).to(device)
        logger.info("Wiener gain from %s (%s): mean %.4f, share of bins with W > 0.5: %.2f%%",
                    path, args.power_mode, float(gain.mean()), float((gain > 0.5).float().mean() * 100))

    denoiser = DDSDenoiser(eps_fn, sigma=args.sigma, method=args.method if args.mode == "certify" else "wadds",
                           steps=args.steps, eta=args.eta, gain=gain, gamma=args.gamma,
                           anchor_window=(args.anchor_t_min, args.anchor_t_max), t_end=args.t_end)
    if args.guide_lambda > 0:
        denoiser.set_guidance(classifier, lam=args.guide_lambda, t_frac=args.guide_t_frac,
                              conf_min=args.guide_conf_min, lowpass=args.guide_lowpass, base=args.guide_base)
        logger.info("classifier-consistency guidance: %s",
                    {k: v for k, v in denoiser.guide.items() if k != "classifier"})
    logger.info("sigma=%g -> t_sigma=%.4f (discrete ~%d), method=%s, NFE/sample=%d",
                args.sigma, denoiser.t_sigma, round(denoiser.t_sigma * 1000),
                denoiser.method if args.mode == "certify" else "diagnose", denoiser.nfe)
    return denoiser, heads, spec.num_classes


# --------------------------------------------------------------------------------------------
# TSV helpers
# --------------------------------------------------------------------------------------------

def shard_file(run_dir, prefix, args):
    return os.path.join(run_dir, f"{prefix}_shard{args.shard_i}of{args.shard_n}.tsv")


def read_rows(run_dir, prefix):
    rows = []
    for path in sorted(glob.glob(os.path.join(run_dir, f"{prefix}_shard*.tsv"))):
        with open(path, newline="", encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh, delimiter="\t"))
    return rows


def open_writer(path, fields):
    new = not os.path.exists(path)
    fh = open(path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
    if new:
        w.writeheader()
        fh.flush()
    return fh, w


# --------------------------------------------------------------------------------------------
# summaries
# --------------------------------------------------------------------------------------------

def summarize_certify(run_dir, args):
    rows = {int(r["pos"]): r for r in read_rows(run_dir, "certify")}
    if not rows:
        return None
    rr = [rows[k] for k in sorted(rows)]
    labels = [int(r["label"]) for r in rr]
    predicts = [int(r["predict"]) for r in rr]
    radii = [float(r["radius"]) for r in rr]
    out = summarize(labels, predicts, radii, args.radii_list)
    out["complete"] = len(rr) == args.num_sub
    out["seconds_per_image"] = float(np.mean([float(r["seconds"]) for r in rr]))
    return out


def summarize_diagnose(run_dir, args):
    rows = read_rows(run_dir, "diag")
    if not rows:
        return None
    by = {}
    for r in rows:
        by.setdefault(r["variant"], {})[int(r["pos"])] = r
    common = sorted(set.intersection(*(set(v) for v in by.values())))
    rng = np.random.RandomState(0)
    boot = rng.randint(0, len(common), size=(args.n_boot, len(common))) if common else None
    out = {"n_images": len(common), "complete": len(common) == args.num_sub, "variants": {}}
    p_true = {}
    for v, d in by.items():
        n = np.array([int(d[k]["n"]) for k in common])
        pt = np.array([int(d[k]["n_true"]) for k in common]) / n
        top_ok = np.array([int(d[k]["top_class"]) == int(d[k]["label"]) for k in common])
        p_true[v] = pt
        out["variants"][v] = {
            "acc_per_draw": float(pt.mean()),
            "smoothed_acc": float(top_ok.mean()),
            "plugin_acr": float(np.mean([plugin_radius(p, args.sigma, int(nn)) if ok else 0.0
                                         for p, ok, nn in zip(pt, top_ok, n)])),
            "frac_p_true_gt_0.5": float((pt > 0.5).mean()),
        }
    ref = "onestep"
    if ref in p_true and boot is not None:
        for v, pt in p_true.items():
            if v == ref:
                continue
            d = pt - p_true[ref]
            bs = d[boot].mean(1)
            out["variants"][v]["d_acc_per_draw_vs_onestep"] = float(d.mean())
            out["variants"][v]["ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    v = out["variants"]
    if all(k in v for k in ("onestep", "multistep", "wadds", "posthoc")):
        best_base = max(v["onestep"]["acc_per_draw"], v["multistep"]["acc_per_draw"])
        out["gate_G0"] = {
            "wadds_acc_ge_best_baseline": v["wadds"]["acc_per_draw"] >= best_base,
            "wadds_plugin_acr_ge_onestep": v["wadds"]["plugin_acr"] >= v["onestep"]["plugin_acr"],
            "posthoc_acc_ge_best_baseline": v["posthoc"]["acc_per_draw"] >= best_base,
        }
        out["gate_G0"]["pass"] = bool(out["gate_G0"]["wadds_acc_ge_best_baseline"]
                                      and out["gate_G0"]["wadds_plugin_acr_ge_onestep"])
    return out


def write_summary(run_dir, args):
    s = summarize_certify(run_dir, args) if args.mode == "certify" else summarize_diagnose(run_dir, args)
    if s is None:
        logger.warning("no results in %s yet", run_dir)
        return None
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(s, fh, indent=2)
    print(json.dumps(s, indent=2))
    return s


# --------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------

def summarize_dir(run_dir):
    """Recompute ``summary.json`` of a finished or partial run from its saved config (no models)."""
    cfgs = sorted(glob.glob(os.path.join(run_dir, "resolved_config_shard*.json")))
    if not cfgs:
        raise FileNotFoundError(f"no resolved_config_shard*.json in {run_dir}")
    with open(cfgs[0], encoding="utf-8") as fh:
        args = argparse.Namespace(**json.load(fh))
    return write_summary(run_dir, args)


def main(argv=None):
    if argv is None:
        import sys
        argv = sys.argv[1:]
    if argv[:1] == ["--summarize_dir"]:
        setup_logging("info")
        for d in argv[1:]:
            summarize_dir(d)
        return
    args = parse_args(argv)
    setup_logging(args.verbose)
    run_name = args.run_name or default_run_name(args)
    run_dir = os.path.normpath(output_path(args.exp, run_name))
    os.makedirs(run_dir, exist_ok=True)
    logger.info("run folder: %s", run_dir)
    if args.summarize_only:
        write_summary(run_dir, args)
        return

    cfg = {k: v for k, v in vars(args).items()}
    with open(os.path.join(run_dir, f"resolved_config_shard{args.shard_i}of{args.shard_n}.json"), "w",
              encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_all, y_all, indices = load_eval_subset(args.domain, args.num_sub, args.data_seed,
                                             split=args.eval_split, log_dir=run_dir)
    x_all = x_all.detach()
    denoiser, heads, n_classes = build_models(args, device)
    if not isinstance(heads, dict):          # a bare classifier (tests)
        heads = {"": heads}
    cert_key = certify_head_key(bool(getattr(args, "ens_list", [])), getattr(args, "tta_flip", False))
    variants = DIAG_VARIANTS + ((GUIDED,) if getattr(denoiser, "guide", None) is not None else ())
    head_targets = {"multistep", GUIDED}   # extra heads only here (cost; wadds ~ multistep at eta 0)
    diag_keys = [v for v in variants] + [v + k for v in variants if v in head_targets for k in heads if k]

    prefix, fields = ("certify", CERT_FIELDS) if args.mode == "certify" else ("diag", DIAG_FIELDS)
    path = shard_file(run_dir, prefix, args)
    done = set()
    if os.path.exists(path):
        if not args.resume:
            raise FileExistsError(f"{path} exists; pass --resume to continue it")
        done = {int(r["pos"]) for r in read_rows(run_dir, prefix)}
    todo = [p for p in range(args.num_sub) if p % args.shard_n == args.shard_i and p not in done]
    logger.info("%d images to do in shard %s (%d already done)", len(todo), args.shard, len(done))

    gen_ref = [None]
    base = make_base_classifier(denoiser, heads[cert_key], gen_ref)
    fh, writer = open_writer(path, fields)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_run = time.time()
    try:
        for j, pos in enumerate(todo):
            x = x_all[pos].to(device)
            label, idx = int(y_all[pos]), int(indices[pos])
            gen = image_generator(args.seed, idx, device)
            gen_ref[0] = gen
            t0 = time.time()
            if args.mode == "certify":
                r = certify(base, x, sigma=args.sigma, n0=args.n0, n=args.n, alpha=args.alpha,
                            n_classes=n_classes, batch=args.batch, generator=gen)
                writer.writerow({"pos": pos, "idx": idx, "label": label, "predict": r.predict,
                                 "correct": int(r.predict == label), "p_lower": f"{r.p_lower:.6f}",
                                 "radius": f"{r.radius:.6f}",
                                 "count_cA": int(r.counts_estimate[r.predict]) if r.predict >= 0 else 0,
                                 "seconds": f"{time.time() - t0:.2f}"})
            else:
                counts = {v: np.zeros(n_classes, dtype=np.int64) for v in diag_keys}
                remaining = args.n_diag
                while remaining > 0:
                    b = min(args.batch, remaining)
                    xi = torch.randn((b, *x.shape), generator=gen, device=device, dtype=x.dtype)
                    outs = denoiser.diagnose(x.unsqueeze(0) + args.sigma * xi, gen, variants=variants)
                    with torch.no_grad():
                        for v, img in outs.items():
                            for k, head in heads.items():
                                if k and v not in head_targets:
                                    continue
                                pred = head(img).argmax(1).cpu().numpy()
                                counts[v + k] += np.bincount(pred, minlength=n_classes)[:n_classes]
                    remaining -= b
                for v, c in counts.items():
                    writer.writerow({"pos": pos, "idx": idx, "label": label, "variant": v,
                                     "n": args.n_diag, "n_true": int(c[label]),
                                     "top_class": int(c.argmax()), "n_top": int(c.max())})
            fh.flush()
            if j == 0 or (j + 1) % 10 == 0 or j + 1 == len(todo):
                mem = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else 0.0
                logger.info("image %d/%d (pos %d): %.1f s; elapsed %.1f min; peak GPU mem %.1f GiB",
                            j + 1, len(todo), pos, time.time() - t0, (time.time() - t_run) / 60, mem)
    finally:
        fh.close()
    write_summary(run_dir, args)


if __name__ == "__main__":
    main()
