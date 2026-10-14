"""Measure the score network's throughput and estimate the GPU-days of full-gradient attacks.

    python -m scripts.calibrate_throughput --config cifar10.yml --batch_sizes 64,128,256

Prints r = images x NFE / second for a forward pass and for forward + backward (with the
recomputation of ``--pcp_checkpoint eps``), then the GPU-days of a PGD+EOT evaluation,

    NFE = N_img * steps_attack * EOT * K * S * c_bwd * c_g,

for the configurations of the plan. ``c_g`` (extra cost of the PCP guidance under the attack's
double backward) is not measured here; pass an estimate with --c_g.
"""

import argparse
import time

import torch

from src.config import load_config
from src.models.diffusion.sampler import alpha_bar_cont


def bench(fn, reps):
    torch.cuda.synchronize()
    fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / reps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="cifar10.yml")
    p.add_argument("--batch_sizes", default="64,128,256")
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--c_g", type=float, default=2.0)
    args = p.parse_args()

    from src.models.diffusion.base import VPDynamics, load_score_model

    device = torch.device("cuda")
    config = load_config(args.config)
    config.device = device
    model = load_score_model(config, device)
    for q in model.parameters():
        q.requires_grad_(False)
    shape = (config.data.num_channels, config.data.image_size, config.data.image_size)
    dyn = VPDynamics(model, img_shape=shape).to(device)

    def eps(x, t):
        return -dyn.score(t, x.flatten(1)).view_as(x) * torch.sqrt(1 - alpha_bar_cont(t)).to(x).view(-1, 1, 1, 1)

    print(f"GPU: {torch.cuda.get_device_name()}")
    best_fwd = best_bwd = 0.0
    for bs in (int(v) for v in args.batch_sizes.split(",")):
        x = torch.randn(bs, *shape, device=device)
        t = torch.full((bs,), 0.05, device=device)
        with torch.no_grad():
            s_f = bench(lambda: eps(x, t), args.reps)

        def fb():
            xr = x.clone().requires_grad_(True)
            eps(xr, t).sum().backward()

        s_b = bench(fb, args.reps)
        r_f, r_b = bs / s_f, bs / s_b
        best_fwd, best_bwd = max(best_fwd, r_f), max(best_bwd, r_b)
        print(f"batch {bs:4d}: forward {r_f:8.0f} img*NFE/s | forward+backward {r_b:8.0f} img*NFE/s | "
              f"peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
        torch.cuda.reset_peak_memory_stats()

    # forward + recompute + backward ~ one forward plus one forward+backward
    r_attack = 1.0 / (1.0 / best_fwd + 1.0 / best_bwd)
    print(f"\nattack throughput (fwd + recompute + bwd): {r_attack:.0f} img*NFE/s")
    print(f"{'config':40s} {'GPU-days':>9s}")
    for name, n, it, eot, k, s, cg in [
        ("Lee&Kim DiffPure-SP (S=100, EOT 20)", 512, 200, 20, 1, 100, 1.0),
        ("DiffPure-MV K=8 S=20 EOT 4", 512, 200, 4, 8, 20, 1.0),
        ("PCP K=8 S=20 EOT 4", 512, 200, 4, 8, 20, args.c_g),
        ("PCP K=8 S=100 EOT 20", 512, 200, 20, 8, 100, args.c_g),
        ("pilot PCP K=8 S=20, 256 img, 50 it", 256, 50, 4, 8, 20, args.c_g),
    ]:
        nfe = n * it * eot * k * s * cg
        print(f"{name:40s} {nfe / r_attack / 86400:9.2f}")


if __name__ == "__main__":
    main()
