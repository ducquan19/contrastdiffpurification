"""Screen of the confidence-routed cascade (docs/cascade.md): save class probabilities, no CERTIFY.

For every image of a fixed subset, ``--n_draws`` noisy copies ``x + sigma xi`` are denoised ONCE
into the one-shot output ``m`` and the multi-step DDIM outputs (``--steps_list``, eta ``--eta``),
and EVERY classifier of ``--classifiers`` classifies EVERY output (plus the clean image, a sanity
check of the weights). Probabilities go to ``outputs/<exp>/<run>/probs/pos_XXXX.npz``:

    probs (D, V, C, K) float16   draws x variants (onestep, ms<S>...) x classifiers x classes
    clean (C, K) float16, label, idx, pos

so that scripts/analyze_cascade.py evaluates every head (single / ensemble / router) and every
early-exit threshold offline with common random numbers. A killed job resumes (existing image
files are skipped).

    python -m scripts.screen_cascade --sigma 0.5 --num_sub 200 --n_draws 200
    python -m scripts.analyze_cascade outputs/cascade/screen_s0.5_*/
"""

import argparse
import json
import logging
import os
import time

import numpy as np
import torch

from scripts import certify_dds
from scripts.certify_dds import _frozen
from src.certify.denoisers import DDSDenoiser
from src.certify.smoothing import image_generator
from src.data import loaders
from src.data.loaders import EVAL_SPLITS
from src.paths import output_path
from src.utils import setup_logging

logger = logging.getLogger(__name__)

DEFAULT_CLASSIFIERS = ",".join([
    "cifar10-wideresnet-28-10",                                   # standard WRN (current results)
    "robustbench:Wang2023Better_WRN-28-10:L2",                    # l2-AT WRN-28-10 (+ EDM-generated data)
    "robustbench:Rebuffi2021Fixing_28_10_cutmix_ddpm:L2",         # l2-AT WRN-28-10 (+ DDPM-generated data)
    "hf:aaraki/vit-base-patch16-224-in21k-finetuned-cifar10",     # Carlini et al.'s ViT-B/16 (IN-21k)
])
DEFAULT_TAGS = "wrn,wang23l2,rebuffi21l2,vitb16"


def load_classifier(name, device):
    """Frozen classifier by name (lazy import: robustbench is only needed on the GPU side)."""
    from src.models.classifiers import get_image_classifier
    return _frozen(get_image_classifier(name), device)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sigma", type=float, required=True)
    p.add_argument("--classifiers", default=DEFAULT_CLASSIFIERS, help="comma-separated; the first is the reference")
    p.add_argument("--tags", default=DEFAULT_TAGS, help="short names of --classifiers (same order)")
    p.add_argument("--steps_list", default="10", help="multi-step variants, e.g. '5,10'")
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--config", default="cifar10.yml")
    p.add_argument("--domain", default="cifar10")
    p.add_argument("--eval_split", choices=EVAL_SPLITS, default="dev")
    p.add_argument("--num_sub", type=int, default=200)
    p.add_argument("--data_seed", type=int, default=0)
    p.add_argument("--n_draws", type=int, default=200)
    p.add_argument("--batch", type=int, default=100, help="noisy copies at once (ViT at 224 px: <= 100 on 32 GB)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exp", default="./cascade")
    p.add_argument("--run_name", default=None)
    p.add_argument("--shard", default="0/1")
    p.add_argument("--verbose", default="info")
    args = p.parse_args(argv)
    args.clf_list = [c for c in args.classifiers.split(",") if c]
    args.tag_list = [t for t in args.tags.split(",") if t]
    if len(args.tag_list) != len(args.clf_list):
        if args.tags != DEFAULT_TAGS:
            p.error("--tags must name every classifier")
        args.tag_list = [f"c{i}" for i in range(len(args.clf_list))]
    args.steps_values = sorted({int(v) for v in args.steps_list.split(",") if v})
    if not args.steps_values or min(args.steps_values) < 2:
        p.error("--steps_list needs values >= 2")
    i, n = (int(v) for v in args.shard.split("/"))
    args.shard_i, args.shard_n = i, n
    return args


def default_run_name(args):
    steps = "-".join(str(s) for s in args.steps_values)
    return (f"screen_s{args.sigma:g}_S{steps}_eta{args.eta:g}_D{args.n_draws}_"
            f"{args.eval_split}{args.num_sub}_ds{args.data_seed}")


def main(argv=None):
    args = parse_args(argv)
    setup_logging(args.verbose)
    run_dir = os.path.normpath(output_path(args.exp, args.run_name or default_run_name(args)))
    prob_dir = os.path.join(run_dir, "probs")
    os.makedirs(prob_dir, exist_ok=True)
    variants = ["onestep"] + [f"ms{s}" for s in args.steps_values]
    meta = {**{k: v for k, v in vars(args).items()}, "variants": variants}
    with open(os.path.join(run_dir, f"meta_shard{args.shard_i}of{args.shard_n}.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # denoiser (+ the first classifier) exactly as certify_dds builds them
    cargs = certify_dds.parse_args(["--mode", "certify", "--method", "multistep", "--sigma", str(args.sigma),
                                "--steps", str(args.steps_values[0]), "--eta", str(args.eta),
                                "--config", args.config, "--domain", args.domain,
                                "--classifier_name", args.clf_list[0]])
    den0, heads, n_classes = certify_dds.build_models(cargs, device)
    if not isinstance(heads, dict):          # a bare classifier (tests)
        heads = {"": heads}
    classifiers = [heads[""]] + [load_classifier(c, device) for c in args.clf_list[1:]]
    dens = [DDSDenoiser(den0.eps_fn, sigma=args.sigma, method="multistep", steps=s, eta=args.eta,
                        t_end=den0.t_end) for s in args.steps_values]

    x_all, y_all, indices = loaders.load_eval_subset(args.domain, args.num_sub, args.data_seed,
                                             split=args.eval_split, log_dir=run_dir)
    todo = [p for p in range(args.num_sub) if p % args.shard_n == args.shard_i
            and not os.path.exists(os.path.join(prob_dir, f"pos_{p:04d}.npz"))]
    logger.info("%d images to do; variants %s; classifiers %s", len(todo), variants, args.tag_list)

    def probs_of(img01):
        return torch.stack([torch.softmax(f(img01).float(), dim=1) for f in classifiers], dim=1)  # (B, C, K)

    t_run = time.time()
    with torch.no_grad():
        for j, pos in enumerate(todo):
            t0 = time.time()
            x = x_all[pos].to(device)
            label, idx = int(y_all[pos]), int(indices[pos])
            gen = image_generator(args.seed, idx, device)
            out = np.zeros((args.n_draws, len(variants), len(classifiers), n_classes), dtype=np.float16)
            done = 0
            while done < args.n_draws:
                b = min(args.batch, args.n_draws - done)
                xi = torch.randn((b, *x.shape), generator=gen, device=device, dtype=x.dtype)
                x_noisy = x.unsqueeze(0) + args.sigma * xi
                x_t, eps0, m = dens[0].start(x_noisy)
                imgs = [(m.clamp(-1, 1) + 1) * 0.5]
                for d in dens:
                    zs = d.draw_step_noise(x_noisy.shape, gen, device, x_noisy.dtype)
                    s = d.trajectory(x_t, eps0, m, zs, anchored=False)
                    imgs.append((s.clamp(-1, 1) + 1) * 0.5)
                for v, img in enumerate(imgs):
                    out[done:done + b, v] = probs_of(img).cpu().numpy().astype(np.float16)
                done += b
            clean = probs_of(x.unsqueeze(0))[0].cpu().numpy().astype(np.float16)
            tmp = os.path.join(prob_dir, f"pos_{pos:04d}.tmp.npz")
            np.savez_compressed(tmp, probs=out, clean=clean, label=label, idx=idx, pos=pos)
            os.replace(tmp, os.path.join(prob_dir, f"pos_{pos:04d}.npz"))
            if j == 0 or (j + 1) % 10 == 0 or j + 1 == len(todo):
                acc = (out.astype(np.float32).argmax(-1) == label).mean(axis=0)   # (V, C)
                logger.info("image %d/%d (pos %d): %.1f s, elapsed %.1f min; acc/draw [variant x clf] %s",
                            j + 1, len(todo), pos, time.time() - t0, (time.time() - t_run) / 60,
                            np.round(acc, 2).tolist())


if __name__ == "__main__":
    main()
