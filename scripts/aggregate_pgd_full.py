"""Table of clean / robust accuracy (mean +- std over seeds) for every ``pgd_full`` experiment.

    python -m scripts.aggregate_pgd_full outputs/exp_results_pcp [--csv results_pcp.csv]

Reads every ``pgd_full_summary.json`` written by ``scripts.summarize_pgd_full`` under the given
folder, groups runs by experiment name (the ``-i`` folder) and prints a Markdown table. Run
``summarize_pgd_full`` on every run folder first (it merges the shards of one run).
"""

import argparse
import csv
import glob
import json
import os
import statistics


def _fmt(vals):
    if not vals:
        return "-"
    m = 100 * statistics.mean(vals)
    s = 100 * statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{m:.2f} ± {s:.2f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", help="experiment root, e.g. outputs/exp_results_pcp")
    p.add_argument("--csv", default=None)
    a = p.parse_args()

    rows = {}
    for f in sorted(glob.glob(os.path.join(a.root, "**", "pgd_full_summary.json"), recursive=True)):
        rel = os.path.relpath(f, a.root).split(os.sep)
        # <name>/<classifier>/<diffusion>_<guidance>_pgd_full/seed<S>/data<D>/pgd_full_summary.json
        name, seed = rel[0], rel[-3]
        s = json.load(open(f))
        r = rows.setdefault(name, {"seeds": [], "n": [], "reps": [], "def_clean": [], "def_adv": [],
                                   "def_adv_neval": [], "def_adv_worst": [], "clf_adv": [],
                                   "error_corr_adv": []})
        r["seeds"].append(seed)
        r["n"].append(s.get("n"))
        r["reps"].append((s.get("attack") or {}).get("score_reps"))
        for k in ("def_clean", "def_adv", "def_adv_neval", "def_adv_worst", "clf_adv"):
            if s.get(k):
                r[k].append(s[k]["acc"])
        agr = s.get("agreement_adv") or {}
        if agr.get("error_corr") == agr.get("error_corr") and "error_corr" in agr:
            r["error_corr_adv"].append(agr["error_corr"])

    if not rows:
        raise SystemExit(f"no pgd_full_summary.json under {a.root}; run scripts.summarize_pgd_full first")

    header = ["experiment", "seeds", "images/seed", "score reps", "clean acc (%)", "robust acc (%)",
              "robust N-eval (%)", "robust worst-case (%)", "classifier robust (%)",
              "particle error corr. (adv)"]
    table = []
    for name, r in rows.items():
        corr = f"{statistics.mean(r['error_corr_adv']):.3f}" if r["error_corr_adv"] else "-"
        table.append([name, str(len(r["seeds"])), "/".join(sorted({str(n) for n in r["n"]})),
                      "/".join(sorted({str(v) for v in r["reps"]})),
                      _fmt(r["def_clean"]), _fmt(r["def_adv"]), _fmt(r["def_adv_neval"]),
                      _fmt(r["def_adv_worst"]), _fmt(r["clf_adv"]), corr])

    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row in table:
        print("| " + " | ".join(row) + " |")
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(table)
        print(f"written {a.csv}")


if __name__ == "__main__":
    main()
