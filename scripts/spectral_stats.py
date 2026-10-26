"""Power spectrum P(f) of the training images, used by the Wiener gain of WA-DDS (docs/wadds.md).

CPU only, about a minute for CIFAR-10. Run from the repository root (login node is fine):

    python -m scripts.spectral_stats --domain cifar10

Uses the training images outside the held-out validation pool (``bank_pool_size``), never the
test split, and writes ``pretrained/spectral/<domain>_power.pt`` plus a printed SNR table.
"""

import argparse
import json
import logging
import os

import numpy as np
from torch.utils.data import DataLoader, Subset

from src.certify.spectrum import power_spectrum, save_power, snr_table
from src.data.loaders import bank_pool_size
from src.data.specs import DATASET_NAMES, get_dataset_spec
from src.paths import pretrained_path
from src.utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--domain", default="cifar10", choices=DATASET_NAMES)
    p.add_argument("--n_images", type=int, default=0, help="0 = every image of the pool")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=1000)
    p.add_argument("--out", default=None, help="default: pretrained/spectral/<domain>_power.pt")
    p.add_argument("--verbose", default="info")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)
    spec = get_dataset_spec(args.domain)
    train = spec.build_split(train=True)
    pool = bank_pool_size(len(train))
    if args.n_images and args.n_images < pool:
        idx = np.random.RandomState(args.seed).choice(pool, args.n_images, replace=False)
    else:
        idx = np.arange(pool)
    loader = DataLoader(Subset(train, idx.tolist()), batch_size=args.batch_size, shuffle=False, num_workers=4)
    power = power_spectrum(xb for xb, _ in loader)

    out = args.out or pretrained_path("spectral", f"{args.domain}_power.pt")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    meta = {"domain": args.domain, "n_images": int(len(idx)), "split": "train", "pool": int(pool),
            "scale": "[-1,1]", "fft": "fft2 ortho", "seed": int(args.seed)}
    save_power(out, power, meta)
    table = snr_table(power, [0.12, 0.25, 0.5, 1.0])
    logger.info("saved %s: shape %s from %d training images", out, tuple(power.shape), len(idx))
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
