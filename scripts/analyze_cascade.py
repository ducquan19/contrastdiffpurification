"""Offline analysis of a cascade screen (scripts/screen_cascade.py); CPU only, no models.

For each run folder: every head (single classifier, ensemble / router of every subset) on every
denoiser variant, and the early-exit cascade one-shot -> multi-step of every head, with

* acc/draw (frequency of the true class), smoothed accuracy (majority class) and the plug-in ACR
  (optimistic, ignores the confidence bound -- for ranking only; CERTIFY gives the real numbers);
* paired bootstrap 95% CIs against the reference = multi-step (largest S) + the first classifier;
* the cascade threshold chosen by 2-fold cross-fitting over the images (fit tau on even
  positions, evaluate on odd ones, and vice versa) as the fewest-NFE tau that keeps
  ``--retain`` of the same head's multi-step acc/draw -- so its reported accuracy is not tuned on
  the images it is evaluated on;
* the go / no-go gate of docs/cascade.md.

Writes ``analysis.json``, ``policies.tsv`` and ``cascade_curve.tsv`` into each run folder.

    python -m scripts.analyze_cascade outputs/cascade/screen_s0.5_*/ outputs/cascade/screen_s1_*/
"""

import argparse
import csv
import glob
import json
import os

import numpy as np

from src.certify.cascade import (cascade_nfe, cascade_predictions, choose_tau, combine, draw_metrics,
                                 head_name, head_specs, paired_bootstrap, summarize_metrics, tau_grid)


def load_run(run_dir):
    metas = sorted(glob.glob(os.path.join(run_dir, "meta_shard*.json")))
    if not metas:
        raise FileNotFoundError(f"no meta_shard*.json in {run_dir} (not a screen_cascade run folder?)")
    with open(metas[0], encoding="utf-8") as fh:
        meta = json.load(fh)
    files = sorted(glob.glob(os.path.join(run_dir, "probs", "pos_*.npz")))
    files = [f for f in files if not f.endswith(".tmp.npz")]
    if not files:
        raise FileNotFoundError(f"no probs/pos_*.npz in {run_dir} yet")
    P, clean, labels, pos = [], [], [], []
    for f in files:
        z = np.load(f)
        P.append(z["probs"]); clean.append(z["clean"]); labels.append(int(z["label"])); pos.append(int(z["pos"]))
    return meta, np.stack(P), np.stack(clean), np.array(labels), np.array(pos)


def analyze(run_dir, retain=0.99, n_boot=2000, gate_dacc=0.03, gate_retain=0.98, gate_nfe_frac=0.5):
    meta, P, clean, labels, pos = load_run(run_dir)
    sigma, variants, tags = float(meta["sigma"]), list(meta["variants"]), list(meta["tag_list"])
    I, D, V, C, K = P.shape
    ms = len(variants) - 1                          # largest S (variants are sorted)
    S = int(variants[ms][2:])
    specs = head_specs(C)
    names = [head_name(sp, tags) for sp in specs]
    out = {"run_dir": run_dir, "sigma": sigma, "n_images": I, "n_draws": D, "variants": variants,
           "classifiers": dict(zip(tags, meta["clf_list"])), "complete": I == int(meta["num_sub"]),
           "clean_acc": {t: float((clean[:, c].argmax(-1) == labels).mean()) for c, t in enumerate(tags)}}

    hp = {n: combine(P, sp) for n, sp in zip(names, specs)}          # (I, D, V, K)
    met = {(n, v): draw_metrics(hp[n][:, :, vi].argmax(-1), labels, sigma, K)
           for n in names for vi, v in enumerate(variants)}
    ref = (tags[0], variants[ms])
    rows = []
    for n in names:
        for vi, v in enumerate(variants):
            m = met[(n, v)]
            r = {"head": n, "variant": v, "nfe": 1 if v == "onestep" else int(v[2:]), **summarize_metrics(m)}
            r["d_acc"], r["d_acc_ci"] = paired_bootstrap(m["p_true"], met[ref]["p_true"], n_boot)
            r["d_acr"], r["d_acr_ci"] = paired_bootstrap(m["radius"], met[ref]["radius"], n_boot)
            rows.append(r)

    # cascade one-shot -> multi-step (largest S), same head in both stages, 2-fold cross-fitted tau
    folds = [np.where(pos % 2 == 0)[0], np.where(pos % 2 == 1)[0]]
    casc, curve = [], []
    for n in names:
        p1, p2 = hp[n][:, :, 0], hp[n][:, :, ms]
        taus = tau_grid(p1.max(-1))
        for tau in taus:
            pred, cont = cascade_predictions(p1, p2, tau)
            curve.append({"head": n, "tau": float(tau), "nfe": cascade_nfe(S, cont.mean()),
                          "acc_per_draw": float((pred == labels[:, None]).mean())})
        pred = np.zeros((I, D), dtype=np.int64)
        cont = np.zeros((I, D), dtype=bool)
        tau_f = []
        for fit, ev in ((folds[0], folds[1]), (folds[1], folds[0])):
            if len(fit) == 0 or len(ev) == 0:
                continue
            tau = choose_tau(p1, p2, labels, fit, retain, taus)
            tau_f.append(tau)
            pred[ev], cont[ev] = cascade_predictions(p1[ev], p2[ev], tau)
        m = draw_metrics(pred, labels, sigma, K)
        r = {"head": n, "variant": f"cascade1->{S}", "tau_folds": tau_f,
             "nfe": cascade_nfe(S, cont.mean()), **summarize_metrics(m),
             "retain_vs_own_ms": float(m["p_true"].mean() / max(met[(n, variants[ms])]["p_true"].mean(), 1e-12))}
        r["d_acc"], r["d_acc_ci"] = paired_bootstrap(m["p_true"], met[ref]["p_true"], n_boot)
        r["d_acr"], r["d_acr_ci"] = paired_bootstrap(m["radius"], met[ref]["radius"], n_boot)
        casc.append(r)

    # gate (docs/cascade.md)
    best_ms = max((r for r in rows if r["variant"] == variants[ms]), key=lambda r: r["acc_per_draw"])
    ref_row = next(r for r in rows if r["head"] == ref[0] and r["variant"] == ref[1])
    best_c = max(casc, key=lambda r: r["acc_per_draw"])
    out["reference"] = {"head": ref[0], "variant": ref[1], **{k: ref_row[k] for k in ("acc_per_draw", "plugin_acr")}}
    out["best_multistep_head"] = best_ms
    out["best_cascade"] = best_c
    out["gate"] = {
        "accuracy: best head (multi-step) - reference >= %.0f pts, CI > 0" % (100 * gate_dacc):
            bool(best_ms["d_acc"] >= gate_dacc and best_ms["d_acc_ci"][0] > 0),
        "cost: best cascade keeps >= %.0f%% of its multi-step acc at <= %.0f%% NFE" % (100 * gate_retain, 100 * gate_nfe_frac):
            bool(best_c["retain_vs_own_ms"] >= gate_retain and best_c["nfe"] <= gate_nfe_frac * S),
    }
    out["policies"] = rows
    out["cascades"] = casc

    with open(os.path.join(run_dir, "analysis.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    fields = ["head", "variant", "nfe", "acc_per_draw", "smoothed_acc", "plugin_acr", "d_acc", "d_acc_ci",
              "d_acr", "d_acr_ci", "tau_folds", "retain_vs_own_ms"]
    with open(os.path.join(run_dir, "policies.tsv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in rows + casc:
            w.writerow({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()})
    with open(os.path.join(run_dir, "cascade_curve.tsv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["head", "tau", "nfe", "acc_per_draw"], delimiter="\t")
        w.writeheader()
        w.writerows(curve)
    return out


def print_report(a, top=8):
    print(f"\n=== {os.path.basename(os.path.normpath(a['run_dir']))}: sigma {a['sigma']:g}, "
          f"{a['n_images']} images x {a['n_draws']} draws{'' if a['complete'] else ' (INCOMPLETE)'}")
    print("clean accuracy (sanity check of the weights):",
          ", ".join(f"{t} {v:.3f}" for t, v in a["clean_acc"].items()))
    ref = a["reference"]
    print(f"reference {ref['head']} {ref['variant']}: acc/draw {ref['acc_per_draw']:.4f}, plug-in ACR {ref['plugin_acr']:.4f}")
    fmt = "  {:<44s} {:>9s} {:>6.2f} {:>8.4f} {:>8.4f} {:>+8.4f} [{:+.4f},{:+.4f}] {:>+8.4f}"
    print("  {:<44s} {:>9s} {:>6s} {:>8s} {:>8s} {:>8s} {:>17s} {:>8s}".format(
        "head", "variant", "NFE", "acc/drw", "ACR", "d_acc", "95% CI", "d_ACR"))
    for title, rows in (("single classifiers", [r for r in a["policies"] if "(" not in r["head"]]),
                        ("best combined heads (multi-step)", sorted(
                            [r for r in a["policies"] if "(" in r["head"] and r["variant"] != "onestep"],
                            key=lambda r: -r["acc_per_draw"])[:top]),
                        ("best cascades (tau cross-fitted)", sorted(a["cascades"], key=lambda r: -r["acc_per_draw"])[:top])):
        print(" ", title)
        for r in rows:
            print(fmt.format(r["head"][:44], r["variant"], r["nfe"], r["acc_per_draw"], r["plugin_acr"],
                             r["d_acc"], *r["d_acc_ci"], r["d_acr"]))
    for k, v in a["gate"].items():
        print(f"  GATE {'PASS' if v else 'FAIL'}: {k}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--retain", type=float, default=0.99, help="cascade: share of multi-step acc/draw to keep")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--top", type=int, default=8)
    args = p.parse_args(argv)
    for d in args.run_dirs:
        try:
            a = analyze(d, retain=args.retain, n_boot=args.n_boot)
        except FileNotFoundError as e:
            print(f"skip {d}: {e}")
            continue
        print_report(a, args.top)


if __name__ == "__main__":
    main()
