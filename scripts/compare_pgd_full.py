"""Paired comparison of pgd_full runs on the images they have in common.

    python -m scripts.compare_pgd_full outputs/eot_sweep/sp_eot4 outputs/eot_sweep/sp_eot8 ...

Each argument is one run (the ``-i`` folder or any folder below it that holds a single run); the
batch files ``pgd_full_batch*.pt`` are found recursively, so unfinished or sharded runs work.
Images are matched by their index in the evaluated subset (``lo`` + position in the batch), so runs
with different batch sizes or shards line up. On the common images it prints, per run:
clean (mean over score_reps), clean N-eval, robust (mean), robust N-eval and worst-case; and for
every run against the first one (the reference): the paired difference in robust accuracy with a
bootstrap 95% CI, and the N-eval discordant pairs with an exact McNemar p-value.
"""

import argparse
import glob
import math
import os

import torch

COLS = ("clean", "clean_neval", "robust", "neval", "worst")


def load_run(root):
    files = sorted(glob.glob(os.path.join(root, "**", "pgd_full_batch*.pt"), recursive=True))
    if not files:
        raise SystemExit(f"no pgd_full_batch*.pt under {root}")
    dirs = {os.path.dirname(f) for f in files}
    if len(dirs) > 1:
        raise SystemExit(f"{root} holds several runs ({sorted(dirs)}); pass a single run folder")
    per = {}
    for f in files:
        r = torch.load(f, map_location="cpu", weights_only=True)
        if r.get("def_adv") is None:
            continue
        clean, adv = r["def_clean"].float(), r["def_adv"].float()
        broken = r["broken_during_attack"].bool()
        for i in range(len(r["y"])):
            c, a = float(clean[i]), float(adv[i])
            per[int(r["lo"]) + i] = {"clean": c, "clean_neval": float(c >= 1 - 1e-6), "robust": a,
                                     "neval": float(a >= 1 - 1e-6),
                                     "worst": float(not bool(broken[i])) * a}
    return per


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def bootstrap_ci(d, reps=5000, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(len(d), (reps, len(d)), generator=g)
    means = d[idx].mean(dim=1).sort().values
    return float(means[int(0.025 * reps)]), float(means[int(0.975 * reps) - 1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="run folders; the first is the reference")
    a = p.parse_args()

    names = [os.path.basename(os.path.normpath(r)) for r in a.runs]
    runs = [load_run(r) for r in a.runs]
    common = sorted(set.intersection(*(set(r) for r in runs)))
    if not common:
        raise SystemExit("the runs have no image in common")
    print(f"common images: {len(common)} (indices {common[0]}..{common[-1]}); "
          f"per run: " + ", ".join(f"{n} {len(r)}" for n, r in zip(names, runs)))

    vals = [{k: torch.tensor([r[i][k] for i in common]) for k in COLS} for r in runs]
    print("\n| run | clean | clean N-eval | robust | robust N-eval | worst-case |")
    print("|---|---|---|---|---|---|")
    for n, v in zip(names, vals):
        print(f"| {n} | " + " | ".join(f"{100 * float(v[k].mean()):.2f}" for k in COLS) + " |")

    ref = vals[0]
    print(f"\npaired against {names[0]} (difference = run - reference, in points)")
    print("| run | d robust | 95% CI (bootstrap) | d N-eval | N-eval ref-only / run-only | McNemar p |")
    print("|---|---|---|---|---|---|")
    for n, v in zip(names[1:], vals[1:]):
        d = v["robust"] - ref["robust"]
        lo, hi = bootstrap_ci(d)
        b = int(((ref["neval"] == 1) & (v["neval"] == 0)).sum())
        c = int(((ref["neval"] == 0) & (v["neval"] == 1)).sum())
        dn = 100 * float((v["neval"] - ref["neval"]).mean())
        print(f"| {n} | {100 * float(d.mean()):+.2f} | [{100 * lo:+.2f}, {100 * hi:+.2f}] | {dn:+.2f} | "
              f"{b} / {c} | {mcnemar_exact(b, c):.3f} |")


if __name__ == "__main__":
    main()
