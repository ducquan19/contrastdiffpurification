"""Paired comparisons of WA-DDS runs on the same images (no GPU; docs/wadds.md section 4).

    # G0: within each diagnose run, variant vs variant (same noisy samples), and across runs
    python -m scripts.compare_dds diag outputs/wadds/diag_s0.5_*/

    # certify: every run vs the first (reference) run, matched by image position
    python -m scripts.compare_dds certify outputs/wadds/multistep_s0.5_*dev100*/ outputs/wadds/wadds_s0.5_*dev100*/

Paired statistics, per image:
* ``acc/draw`` = frequency of the true class (p_true); the difference gets a bootstrap 95% CI
  over images;
* ``smoothed`` = majority class correct; wins / losses are the discordant images, with an exact
  McNemar p-value;
* ``plugin ACR`` (diagnose) / ``ACR`` (certify): mean radius, bootstrap CI of the difference;
* certified accuracy at each radius (certify), McNemar on "correct and certified at r".
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy.stats import binomtest

from scripts.certify_dds import read_rows
from src.certify.smoothing import DEFAULT_RADII, plugin_radius

PAIRS = (("wadds", "multistep"), ("posthoc", "multistep"), ("wadds", "posthoc"),
         ("multistep", "onestep"), ("wadds", "onestep"))


BASE_VARIANTS = {"onestep", "multistep", "posthoc", "posthoc_rev", "wadds"}


def variant_pairs(variants) -> list:
    """Fixed pairs plus, for every extra variant (heads ``v+ens``..., ``guided``), v - multistep and v - its base."""
    pairs = [(a, b) for a, b in PAIRS if a in variants and b in variants]
    for v in sorted(set(variants) - BASE_VARIANTS):
        for ref in ("multistep", v.split("+")[0]):
            if ref != v and ref in variants and (v, ref) not in pairs:
                pairs.append((v, ref))
    return pairs


def boot_ci(d: np.ndarray, n_boot: int = 5000, seed: int = 0):
    if len(d) == 0:
        return (float("nan"), float("nan"))
    idx = np.random.RandomState(seed).randint(0, len(d), size=(n_boot, len(d)))
    bs = d[idx].mean(1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def mcnemar(a_ok: np.ndarray, b_ok: np.ndarray):
    """(wins of a, losses of a, exact two-sided p) on the discordant pairs."""
    wins = int(np.sum(a_ok & ~b_ok))
    losses = int(np.sum(~a_ok & b_ok))
    p = binomtest(wins, wins + losses, 0.5).pvalue if wins + losses else 1.0
    return wins, losses, float(p)


def load_diag(run_dir: str, sigma: float):
    """{variant: {pos: (p_true, smoothed_correct, plugin_radius)}} of a diagnose run."""
    out = {}
    for r in read_rows(run_dir, "diag"):
        n, pt = int(r["n"]), int(r["n_true"]) / int(r["n"])
        ok = int(r["top_class"]) == int(r["label"])
        out.setdefault(r["variant"], {})[int(r["pos"])] = (pt, ok, plugin_radius(pt, sigma, n) if ok else 0.0)
    return out


def compare_variants(a: dict, b: dict) -> dict:
    keys = sorted(set(a) & set(b))
    pa, pb = (np.array([x[k][0] for k in keys]) for x in (a, b))
    oa, ob = (np.array([x[k][1] for k in keys]) for x in (a, b))
    ra, rb = (np.array([x[k][2] for k in keys]) for x in (a, b))
    w, l, p = mcnemar(oa, ob)
    return {"n": len(keys), "d_acc": float((pa - pb).mean()), "ci_acc": boot_ci(pa - pb),
            "d_acr": float((ra - rb).mean()), "ci_acr": boot_ci(ra - rb), "wins": w, "losses": l, "p": p}


def _fmt(name, c):
    return (f"  {name:<24} n={c['n']:<4} d_acc/draw={100 * c['d_acc']:+6.2f} pts "
            f"[{100 * c['ci_acc'][0]:+.2f}, {100 * c['ci_acc'][1]:+.2f}]  "
            f"d_ACR={c['d_acr']:+.4f} [{c['ci_acr'][0]:+.4f}, {c['ci_acr'][1]:+.4f}]  "
            f"smoothed W/L={c['wins']}/{c['losses']} p={c['p']:.3g}")


def sigma_of(run_dir: str) -> float:
    import glob
    import json
    if not os.path.isdir(run_dir):
        raise SystemExit(f"run folder not found: {run_dir}\n(an unmatched shell glob is passed literally: "
                         f"the run was never started or has another name; check `ls outputs/wadds/`)")
    cfgs = sorted(glob.glob(os.path.join(run_dir, "resolved_config_shard*.json")))
    if not cfgs:
        raise FileNotFoundError(f"no resolved config in {run_dir}")
    with open(cfgs[0], encoding="utf-8") as fh:
        return float(json.load(fh)["sigma"])


def run_diag(dirs):
    runs = []
    for d in dirs:
        s = sigma_of(d)
        v = load_diag(d, s)
        runs.append((os.path.basename(os.path.normpath(d)), s, v))
        print(f"\n## {runs[-1][0]}  (sigma={s:g})")
        for a, b in variant_pairs(v):
            print(_fmt(f"{a} - {b}", compare_variants(v[a], v[b])))
    if len(runs) > 1:
        print("\n## across runs: wadds variant vs the wadds variant of the first run with the same sigma")
        for name, s, v in runs[1:]:
            ref = next((r for r in runs if r[1] == s), None)
            if ref is None or ref[0] == name or "wadds" not in v:
                continue
            print(_fmt(f"{name} - {ref[0]}"[:60], compare_variants(v["wadds"], ref[2]["wadds"])))


def load_certify(run_dir: str):
    return {int(r["pos"]): (int(r["label"]), int(r["predict"]), float(r["radius"]))
            for r in read_rows(run_dir, "certify")}


def run_certify(dirs, radii):
    ref_name = os.path.basename(os.path.normpath(dirs[0]))
    ref = load_certify(dirs[0])
    print(f"reference: {ref_name} ({len(ref)} images)")
    for d in dirs[1:]:
        cur = load_certify(d)
        keys = sorted(set(ref) & set(cur))
        lab = np.array([ref[k][0] for k in keys])
        rp, rr = np.array([ref[k][1] for k in keys]), np.array([ref[k][2] for k in keys])
        cp, cr = np.array([cur[k][1] for k in keys]), np.array([cur[k][2] for k in keys])
        acr_r, acr_c = np.where(rp == lab, rr, 0.0), np.where(cp == lab, cr, 0.0)
        lo, hi = boot_ci(acr_c - acr_r)
        print(f"\n## {os.path.basename(os.path.normpath(d))} - reference   (n={len(keys)} common images)")
        print(f"  ACR {acr_c.mean():.4f} vs {acr_r.mean():.4f}: d={acr_c.mean() - acr_r.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]")
        for r in radii:
            ok_c, ok_r = (cp == lab) & (cr >= r), (rp == lab) & (rr >= r)
            w, l, p = mcnemar(ok_c, ok_r)
            print(f"  r={r:<5g} cert.acc {100 * ok_c.mean():5.1f} vs {100 * ok_r.mean():5.1f}  "
                  f"d={100 * (ok_c.mean() - ok_r.mean()):+5.1f}  W/L={w}/{l} p={p:.3g}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("mode", choices=("diag", "certify"))
    p.add_argument("runs", nargs="+", help="run folders under outputs/wadds/")
    p.add_argument("--radii", default=",".join(str(r) for r in DEFAULT_RADII))
    args = p.parse_args(argv)
    if args.mode == "diag":
        run_diag(args.runs)
    else:
        if len(args.runs) < 2:
            sys.exit("certify mode needs a reference run and at least one other run")
        run_certify(args.runs, [float(r) for r in args.radii.split(",")])


if __name__ == "__main__":
    main()
