"""Full-gradient PGD+EOT (``--attack_version pgd_full``).

The attack back-propagates through the whole defense: use it with ``--diffusion_type particle``
(explicit reverse loop) or with ``--grad_mode full`` on the SDE/ODE purifiers (adjoint). With the
default ``--grad_mode straight_through`` of ``PurifiedClassifier`` it would silently be a
BPDA-identity attack, so ``eval_pgd_full`` refuses that combination.

Protocol (following Lee & Kim, ICCV 2023, with per-sample bookkeeping as in AutoAttack):

* ``steps`` iterations; each averages the input gradient over ``eot_iter`` fresh runs of the
  randomized defense (new diffusion noise every run). The loss is CE (or DLR) on the
  differentiable soft vote of the particles (``output_vote='soft'``).
* An image counts as broken as soon as the EOT-averaged prediction at some iterate is wrong;
  that iterate is kept as its adversarial example (``x_best``), else the last iterate.
* The kept examples are then scored ``score_reps`` times with fresh randomness and the
  configured vote (``hard`` majority by default); robust accuracy is the mean over reps.

Every attack batch is saved to ``pgd_full_batch<NNN>.pt`` (adversarial examples, flags,
per-particle predictions) and skipped on ``--resume``; ``pgd_full_summary.json`` collects the
results with Wilson 95% intervals. ``--shard i/n`` evaluates every n-th batch starting at i, so
several single-GPU processes can split one evaluation.
"""

from __future__ import annotations

import json
import logging
import os
import time

import torch
import torch.nn.functional as F

from src.evaluation.metrics import wilson_interval
from src.evaluation.voting import particle_agreement

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------------
# losses and projections
# --------------------------------------------------------------------------------------------

def dlr_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-sample DLR loss (Croce & Hein, 2020), to be maximised."""
    srt, ind = logits.sort(dim=1)
    top_is_y = (ind[:, -1] == y).float()
    z_y = logits.gather(1, y[:, None]).squeeze(1)
    z_other = srt[:, -2] * top_is_y + srt[:, -1] * (1 - top_is_y)
    return -(z_y - z_other) / (srt[:, -1] - srt[:, -3] + 1e-12)


def attack_loss(logits, y, kind: str) -> torch.Tensor:
    if kind == "ce":
        return F.cross_entropy(logits, y, reduction="none")
    if kind == "dlr":
        return dlr_loss(logits, y)
    raise ValueError(f"unknown loss '{kind}'; available: ce, dlr")


def project(x_adv, x, eps: float, norm: str):
    if norm == "Linf":
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)
    elif norm == "L2":
        d = x_adv - x
        n = d.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1, 1)
        x_adv = x + d * torch.clamp(eps / n, max=1.0)
    else:
        raise ValueError(f"unknown norm '{norm}'")
    return x_adv.clamp(0, 1)


def random_start(x, eps: float, norm: str):
    if norm == "Linf":
        return (x + torch.empty_like(x).uniform_(-eps, eps)).clamp(0, 1)
    d = torch.randn_like(x)
    d = d / d.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
    r = torch.rand(x.shape[0], 1, 1, 1, device=x.device)
    return project(x + eps * r * d, x, eps, norm)


def default_step_size(norm: str, eps: float, steps: int) -> float:
    """0.007 for Linf (the PGD setting of Lee & Kim; check against their code), 2.5 eps / steps for L2."""
    return 0.007 if norm == "Linf" else 2.5 * eps / steps


# --------------------------------------------------------------------------------------------
# the attack
# --------------------------------------------------------------------------------------------

def pgd_eot_full(model, x, y, *, norm="Linf", eps=8 / 255, steps=200, step_size=None,
                 eot_iter=20, loss="ce", rand_init=True, log_every=20):
    """Full-gradient PGD+EOT on one batch; returns ``(x_last, x_best, broken)``.

    ``model(x)`` must return logits (the soft vote for particle models) and be differentiable.
    """
    step_size = default_step_size(norm, eps, steps) if step_size is None else step_size
    x = x.detach().clone()  # SAFE: the clean input is a constant of the attack
    x_adv = random_start(x, eps, norm) if rand_init else x.clone()
    x_best = x_adv.clone()
    broken = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)

    for it in range(steps):
        x_adv.requires_grad_(True)
        grad = torch.zeros_like(x_adv)
        prob = torch.zeros(x.shape[0], 0, device=x.device)
        for _ in range(eot_iter):
            logits = model(x_adv)
            g, = torch.autograd.grad(attack_loss(logits, y, loss).sum(), x_adv)
            grad += g
            p = F.softmax(logits.detach(), dim=1)  # SAFE: bookkeeping
            prob = p if prob.numel() == 0 else prob + p
        with torch.no_grad():  # SAFE: the PGD update itself
            wrong = prob.argmax(dim=1) != y
            newly = wrong & ~broken
            x_best[newly] = x_adv.detach()[newly]  # SAFE: bookkeeping
            broken |= wrong
            if norm == "Linf":
                x_new = x_adv + step_size * grad.sign()
            else:
                gn = grad.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1, 1)
                x_new = x_adv + step_size * grad / gn
            x_adv = project(x_new, x, eps, norm).detach()  # SAFE: next iterate
        if log_every and (it % log_every == 0 or it == steps - 1):
            logger.info("pgd_full it %d/%d: broken %d/%d", it + 1, steps, int(broken.sum()), len(broken))
    x_best[~broken] = x_adv[~broken]
    return x_adv, x_best, broken


@torch.no_grad()  # SAFE: scoring only
def score(model, x, y, reps: int = 1, vote_mode: str = "hard"):
    """Accuracy of the defense on ``x`` over ``reps`` fresh runs; also per-particle predictions."""
    has_vote = hasattr(model, "output_vote")
    if has_vote:
        old, model.output_vote = model.output_vote, vote_mode
    try:
        correct = torch.zeros(x.shape[0], device=x.device)
        particle_pred = None
        for _ in range(reps):
            correct += (model(x).argmax(dim=1) == y).float()
            pl = getattr(model, "last_particle_logits", None)
            if pl is not None:
                particle_pred = pl.argmax(dim=-1)
    finally:
        if has_vote:
            model.output_vote = old
    return correct / reps, particle_pred


# --------------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------------

def _shard(spec: str):
    i, n = (int(v) for v in spec.split("/"))
    if not 0 <= i < n:
        raise ValueError(f"bad --shard {spec}: need 0 <= i < n")
    return i, n


def eval_pgd_full(args, config, model, x_val, y_val, adv_batch_size, log_dir):
    """Clean + full-gradient PGD+EOT robust accuracy of the bare classifier and of the defense."""
    from src.models.classifiers import get_image_classifier  # needs robustbench

    model_ = model.module if isinstance(model, torch.nn.DataParallel) else model
    if hasattr(model_, "args") and getattr(model_.args, "diffusion_type", "particle") in ("sde", "ode") \
            and getattr(model_.args, "grad_mode", "straight_through") != "full":
        raise ValueError("--attack_version pgd_full on the sde/ode purifier needs --grad_mode full; "
                         "otherwise the gradient of the purifier is the identity")

    device = config.device
    norm = "Linf" if args.lp_norm == "Linf" else "L2"
    eot = args.pgd_eot_iter if args.pgd_eot_iter is not None else args.eot_iter
    step_size = args.pgd_step_size if args.pgd_step_size is not None else \
        default_step_size(norm, args.adv_eps, args.pgd_steps)
    vote_mode = getattr(getattr(args, "pcp_config", None), "vote", "hard")
    shard_i, shard_n = _shard(args.shard)
    x_val, y_val = x_val.detach().to(device), y_val.to(device)  # SAFE: inputs
    n_batches = (x_val.shape[0] + adv_batch_size - 1) // adv_batch_size
    logger.info("pgd_full: %s eps=%.5f steps=%d step=%.5f eot=%d loss=%s, %d batches, shard %d/%d",
                norm, args.adv_eps, args.pgd_steps, step_size, eot, args.pgd_loss, n_batches,
                shard_i, shard_n)

    clf = get_image_classifier(args.classifier_name).to(device).eval() if args.attack_classifier else None
    attack_defense = getattr(args, "attack_defense", True)

    results = []
    for b in range(shard_i, n_batches, shard_n):
        path = os.path.join(log_dir, f"pgd_full_batch{b:03d}.pt")
        if args.resume and os.path.exists(path):
            logger.info("batch %d: found %s, skipping", b, path)
            results.append(torch.load(path, map_location="cpu"))
            continue
        lo, hi = b * adv_batch_size, min((b + 1) * adv_batch_size, x_val.shape[0])
        x, y = x_val[lo:hi], y_val[lo:hi]
        t0 = time.time()
        rec = {"batch": b, "lo": lo, "hi": hi, "y": y.cpu()}

        if clf is not None:
            with torch.no_grad():  # SAFE: scoring
                rec["clf_clean"] = (clf(x).argmax(1) == y).cpu()
            _, xb, _ = pgd_eot_full(clf, x, y, norm=norm, eps=args.adv_eps, steps=args.pgd_steps,
                                    step_size=step_size, eot_iter=1, loss=args.pgd_loss,
                                    rand_init=args.pgd_random_start, log_every=0)
            with torch.no_grad():  # SAFE: scoring
                rec["clf_adv"] = (clf(xb).argmax(1) == y).cpu()

        if attack_defense:
            model_.reset_counter()
            clean, pp_clean = score(model, x, y, args.score_reps, vote_mode)
            rec["def_clean"] = clean.cpu()
            rec["particle_pred_clean"] = None if pp_clean is None else pp_clean.cpu()

            x_last, x_best, broken = pgd_eot_full(
                model, x, y, norm=norm, eps=args.adv_eps, steps=args.pgd_steps, step_size=step_size,
                eot_iter=eot, loss=args.pgd_loss, rand_init=args.pgd_random_start)
            adv, pp_adv = score(model, x_best, y, args.score_reps, vote_mode)
            rec.update({"def_adv": adv.cpu(), "broken_during_attack": broken.cpu(),
                        "particle_pred_adv": None if pp_adv is None else pp_adv.cpu(),
                        "x_adv": x_best.cpu(), "x_last": x_last.cpu()})
        rec["seconds"] = time.time() - t0
        torch.save(rec, path)
        logger.info("batch %d: %s, %.0fs", b, ", ".join(
            f"{k} {float(rec[k].float().mean()):.3f}" for k in ("clf_clean", "clf_adv", "def_clean", "def_adv")
            if rec.get(k) is not None), rec["seconds"])
        results.append(rec)

    summary = summarize(results, args, norm, step_size, eot)
    # an unsharded run is complete here; shards are merged later by scripts.summarize_pgd_full
    name = "pgd_full_summary.json" if shard_n == 1 else f"pgd_full_summary_shard{shard_i}of{shard_n}.json"
    out = os.path.join(log_dir, name)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("pgd_full summary: %s", json.dumps({k: summary[k] for k in ("n", "def_clean", "def_adv")}))
    return summary


def _acc(records, key):
    vals = [r[key].float() for r in records if r.get(key) is not None]
    if not vals:
        return None
    v = torch.cat(vals)
    k, n = float(v.sum()), len(v)
    lo, hi = wilson_interval(round(k), n)
    return {"acc": k / n, "n": n, "wilson95": [lo, hi]}


def summarize(records, args, norm, step_size, eot):
    out = {
        "n": sum(len(r["y"]) for r in records),
        "attack": {"norm": norm, "eps": args.adv_eps, "steps": args.pgd_steps, "step_size": step_size,
                   "eot_iter": eot, "loss": args.pgd_loss, "random_start": args.pgd_random_start,
                   "score_reps": args.score_reps},
        "clf_clean": _acc(records, "clf_clean"), "clf_adv": _acc(records, "clf_adv"),
        "def_clean": _acc(records, "def_clean"), "def_adv": _acc(records, "def_adv"),
        "broken_during_attack": _acc(records, "broken_during_attack"),
    }
    for key in ("particle_pred_clean", "particle_pred_adv"):
        preds = [r[key] for r in records if r.get(key) is not None]
        if preds:
            ys = torch.cat([r["y"] for r in records if r.get(key) is not None])
            out[key.replace("particle_pred", "agreement")] = particle_agreement(torch.cat(preds), ys)
    return out
