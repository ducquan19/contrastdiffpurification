"""Merge the per-batch results of a (possibly sharded) ``pgd_full`` run into one summary.

    python -m scripts.summarize_pgd_full <run folder containing pgd_full_batch*.pt>

Writes ``pgd_full_summary.json`` next to the batch files and prints clean / robust accuracy.
Reads ``resolved_config.json`` of the run for the attack settings.
"""

import argparse
import glob
import json
import os
from types import SimpleNamespace

import torch

from src.evaluation.attacks_full import default_step_size, summarize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    a = p.parse_args()
    files = sorted(glob.glob(os.path.join(a.run_dir, "pgd_full_batch*.pt")))
    if not files:
        raise SystemExit(f"no pgd_full_batch*.pt in {a.run_dir}")
    records = [torch.load(f, map_location="cpu", weights_only=True) for f in files]
    cfg = json.load(open(os.path.join(a.run_dir, "resolved_config.json")))["args"]
    args = SimpleNamespace(adv_eps=cfg["adv_eps"], pgd_steps=cfg["pgd_steps"], pgd_loss=cfg["pgd_loss"],
                           pgd_random_start=cfg["pgd_random_start"], score_reps=cfg["score_reps"],
                           pgd_lowpass_sigma=cfg.get("pgd_lowpass_sigma"))
    norm = "Linf" if cfg["lp_norm"] == "Linf" else "L2"
    eot = cfg["pgd_eot_iter"] if cfg.get("pgd_eot_iter") is not None else cfg["eot_iter"]
    step = cfg.get("pgd_step_size")
    if step is None:
        step = default_step_size(norm, cfg["adv_eps"], cfg["pgd_steps"])
    summary = summarize(records, args, norm, step, eot)
    summary["batches"] = [r["batch"] for r in records]
    out = os.path.join(a.run_dir, "pgd_full_summary.json")
    json.dump(summary, open(out, "w"), indent=2)
    for key in ("def_clean", "def_adv", "clf_clean", "clf_adv"):
        v = summary.get(key)
        if v:
            print(f"{key:10s} {100 * v['acc']:6.2f}%  (n={v['n']}, 95% CI {100 * v['wilson95'][0]:.1f}-"
                  f"{100 * v['wilson95'][1]:.1f})")
    print(f"written {out}")


if __name__ == "__main__":
    main()
