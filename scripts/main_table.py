"""Main certification table from CERTIFY run folders (no GPU; docs/cascade.md).

    python -m scripts.main_table                                   # outputs/wadds/*stride500*
    python -m scripts.main_table outputs/wadds/*stride500*/ --csv outputs/main_table_stride500.csv

For every run: certified accuracy at each radius, ACR, mean NFE (cascade) and seconds per image,
plus, per sigma, the paired difference against the reference run (default: multi-step, standard
WRN), matched by test index: ACR difference with a bootstrap 95% CI and McNemar p-values of
"correct and certified at r". The same table is printed without the test images that are also
dev images used for the screen (they chose the classifier and tau), and as the best-over-sigma
envelope that published tables report.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np

from scripts.certify_dds import classifier_tag, read_rows
from scripts.compare_dds import boot_ci, mcnemar
from src.data.loaders import dev_pool_indices, select_subset_indices

RADII = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)


def dev_indices(n_test: int = 10000, num_sub: int = 200, data_seed: int = 0) -> set:
    """Test indices of the dev images of the screen (``--eval_split dev --num_sub 200 --data_seed 0``)."""
    pool = dev_pool_indices(n_test)
    return {int(i) for i in pool[select_subset_indices(len(pool), num_sub, data_seed)]}


def load_run(run_dir: str):
    cfgs = sorted(glob.glob(os.path.join(run_dir, "resolved_config_shard*.json")))
    if not cfgs:
        return None
    with open(cfgs[0], encoding="utf-8") as fh:
        cfg = argparse.Namespace(**json.load(fh))
    if getattr(cfg, "mode", "certify") != "certify":
        return None
    rows = {}
    for r in read_rows(run_dir, "certify"):
        rows[int(r["idx"])] = r
    method = cfg.method
    if method in ("multistep", "wadds", "cascade"):
        method += f" S{cfg.steps}"
    if cfg.method == "cascade":
        method += f" tau{cfg.exit_tau:g}"
    clf = classifier_tag(cfg) or "wrn"
    if getattr(cfg, "route_classifiers", ""):
        clf += f"+route{len(cfg.route_classifiers.split(','))}"
    return {"dir": run_dir, "name": os.path.basename(os.path.normpath(run_dir)), "sigma": float(cfg.sigma),
            "method": method, "clf": clf, "label": f"{clf} | {method}", "num_sub": int(cfg.num_sub), "rows": rows}


def arrays(run, keep):
    keys = sorted(k for k in run["rows"] if k in keep)
    lab = np.array([int(run["rows"][k]["label"]) for k in keys])
    pred = np.array([int(run["rows"][k]["predict"]) for k in keys])
    rad = np.array([float(run["rows"][k]["radius"]) for k in keys])
    return keys, lab, pred, rad


def stats(run, keep):
    keys, lab, pred, rad = arrays(run, keep)
    ok = pred == lab
    out = {"n": len(keys), "acr": float(np.where(ok, rad, 0).mean()) if keys else float("nan")}
    for r in RADII:
        out[r] = float(100 * (ok & (rad >= r)).mean()) if keys else float("nan")
    rr = list(run["rows"].values())
    out["nfe"] = float(np.mean([float(x["nfe_mean"]) for x in rr])) if rr and "nfe_mean" in rr[0] else None
    out["sec"] = float(np.mean([float(x["seconds"]) for x in rr])) if rr else float("nan")
    return out


def paired(run, ref, keep):
    keys = sorted(set(run["rows"]) & set(ref["rows"]) & keep)
    if not keys:
        return None
    def get(x):
        lab = np.array([int(x["rows"][k]["label"]) for k in keys])
        pred = np.array([int(x["rows"][k]["predict"]) for k in keys])
        rad = np.array([float(x["rows"][k]["radius"]) for k in keys])
        return lab, pred, rad
    lab, pa, ra = get(run)
    _, pb, rb = get(ref)
    acr_a, acr_b = np.where(pa == lab, ra, 0.0), np.where(pb == lab, rb, 0.0)
    out = {"n": len(keys), "d_acr": float((acr_a - acr_b).mean()), "ci": boot_ci(acr_a - acr_b)}
    for r in (0.0, 0.5, 1.0):
        w, l, p = mcnemar((pa == lab) & (ra >= r), (pb == lab) & (rb >= r))
        out[r] = (float(100 * (((pa == lab) & (ra >= r)).mean() - ((pb == lab) & (rb >= r)).mean())), p)
    return out


def print_table(runs, keep, title, ref_match):
    print(f"\n=== {title}")
    hdr = "  {:<34s} {:>4s} " + " ".join(["{:>5s}"] * len(RADII)) + " {:>7s} {:>5s} {:>6s}"
    print(hdr.format("classifier | method", "n", *[f"r{r:g}" for r in RADII], "ACR", "NFE", "s/img"))
    table = []
    for s in sorted({r["sigma"] for r in runs}):
        group = sorted((r for r in runs if r["sigma"] == s), key=lambda r: r["label"])
        ref = next((r for r in group if ref_match(r)), None)
        print(f"  sigma = {s:g}" + (f"   (reference: {ref['label']})" if ref else "   (no reference run)"))
        for r in group:
            st = stats(r, keep)
            nfe = st["nfe"] if st["nfe"] is not None else (1.0 if r["method"].startswith("onestep") else
                                                           float(r["method"].split("S")[1].split()[0]))
            flag = "" if len(r["rows"]) == r["num_sub"] else f"  INCOMPLETE {len(r['rows'])}/{r['num_sub']}"
            print(("  {:<34s} {:>4d} " + " ".join(["{:5.1f}"] * len(RADII)) + " {:7.4f} {:5.2f} {:6.1f}{}").format(
                r["label"][:34], st["n"], *[st[x] for x in RADII], st["acr"], nfe, st["sec"], flag))
            row = {"sigma": s, "classifier": r["clf"], "method": r["method"], "n": st["n"], "ACR": st["acr"],
                   "NFE": nfe, "complete": not flag, **{f"r={x:g}": st[x] for x in RADII}}
            if ref is not None and r is not ref:
                pr = paired(r, ref, keep)
                if pr:
                    print("      vs ref: d_ACR {:+.4f} [{:+.4f}, {:+.4f}]  d_cert r0 {:+.1f} (p={:.2g}), "
                          "r0.5 {:+.1f} (p={:.2g}), r1 {:+.1f} (p={:.2g})".format(
                              pr["d_acr"], *pr["ci"], pr[0.0][0], pr[0.0][1], pr[0.5][0], pr[0.5][1], pr[1.0][0], pr[1.0][1]))
                    row.update({"d_ACR_vs_ref": pr["d_acr"], "d_ACR_ci_lo": pr["ci"][0], "d_ACR_ci_hi": pr["ci"][1]})
            table.append(row)
    return table


def print_envelope(table, title):
    print(f"\n=== {title}: best over sigma per radius (as published tables report)")
    for key in sorted({(r["classifier"], r["method"].split(" tau")[0]) for r in table}):
        rows = [r for r in table if (r["classifier"], r["method"].split(" tau")[0]) == key]
        env = [max(r[f"r={x:g}"] for r in rows) for x in RADII]
        sig = sorted({r["sigma"] for r in rows})
        print(("  {:<34s} " + " ".join(["{:5.1f}"] * len(RADII)) + "   sigmas {}").format(
            f"{key[0]} | {key[1]}"[:34], *env, ",".join(f"{s:g}" for s in sig)))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("runs", nargs="*", help="run folders (default outputs/wadds/*stride500*)")
    p.add_argument("--ref_clf", default="wrn", help="classifier tag of the reference run")
    p.add_argument("--ref_method", default="multistep S10", help="method of the reference run")
    p.add_argument("--csv", default=None, help="write the table (all images) to this CSV")
    args = p.parse_args(argv)
    dirs = args.runs or sorted(glob.glob(os.path.join("outputs", "wadds", "*stride500*")))
    runs = [r for r in (load_run(d) for d in dirs if os.path.isdir(d)) if r and r["rows"]]
    if not runs:
        raise SystemExit(f"no CERTIFY results in {dirs or 'outputs/wadds/*stride500*'}")
    match = lambda r: r["clf"] == args.ref_clf and r["method"] == args.ref_method
    every = set().union(*(set(r["rows"]) for r in runs))
    dev = dev_indices()
    table = print_table(runs, every, "all test images", match)
    print_envelope(table, "all test images")
    overlap = every & dev
    print(f"\n{len(overlap)} of the test images are also screen dev images: {sorted(overlap)}")
    t2 = print_table(runs, every - dev, f"without the {len(overlap)} dev images", match)
    print_envelope(t2, f"without the {len(overlap)} dev images")
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            fields = sorted({k for r in table for k in r}, key=lambda k: (k[:2] != "si", k))
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(table)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
