"""Build the PCP prototype bank: spherical k-means of encoder features of clean *training* images.

For every time bucket t > 0 the images are noised to t and mapped back with the score network's
Tweedie estimate, so that the prototypes live where the guidance reads features during the
reverse process (see src/guidance/memory_bank.py). Run from the repository root on a GPU node:

    python -m scripts.build_prototype_bank --config cifar10.yml --domain cifar10 \
        --encoder robustbench:Rebuffi2021Fixing_R18_ddpm --n_prototypes 500 \
        --t_values 0,0.025,0.05,0.075,0.1 --n_images 20000

The output path defaults to the ``bank`` entry of configs/method/pcp.yaml (under pretrained/).
Training images only, and never the held-out validation pool (src/data/loaders.py, VAL_SIZE).
"""

import argparse
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.config import load_config
from src.data.loaders import bank_pool_size
from src.data.specs import DATASET_NAMES, get_dataset_spec
from src.guidance.encoders import build_feature_encoder
from src.guidance.memory_bank import build_prototype_bank
from src.method_pcp import load_pcp_config
from src.models.diffusion.sampler import alpha_bar_cont
from src.utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", default="cifar10.yml", help="score-SDE config in configs/")
    p.add_argument("--domain", default="cifar10", choices=DATASET_NAMES)
    p.add_argument("--pcp_method", default="method/pcp.yaml", help="defaults for encoder / bank path")
    p.add_argument("--encoder", default=None, help="default: encoder of --pcp_method")
    p.add_argument("--encoder_feature", default=None, choices=["penultimate", "logits"])
    p.add_argument("--n_prototypes", type=int, default=500)
    p.add_argument("--t_values", default="0,0.025,0.05,0.075,0.1",
                   help="comma-separated continuous times of the buckets (t* = 100 -> 0.1)")
    p.add_argument("--n_images", type=int, default=20000, help="random training images used")
    p.add_argument("--batch_size", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="default: bank path of --pcp_method")
    p.add_argument("--verbose", default="info")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)
    pcp = load_pcp_config(args.pcp_method)
    encoder_name = args.encoder or pcp.encoder
    feature = args.encoder_feature or pcp.encoder_feature
    out = args.out or pcp.bank_path()
    t_values = [float(v) for v in args.t_values.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    spec = get_dataset_spec(args.domain)
    train = spec.build_split(train=True)
    # never the validation pool (--eval_split val), never the test split
    pool = bank_pool_size(len(train))
    idx = np.random.RandomState(args.seed).choice(pool, min(args.n_images, pool), replace=False)
    loader = DataLoader(Subset(train, idx.tolist()), batch_size=args.batch_size, shuffle=False, num_workers=4)
    batches = (xb for xb, _ in loader)

    encoder = build_feature_encoder(encoder_name, feature, dataset=args.domain, device=device)

    eps_fn = None
    if any(t > 0 for t in t_values):
        from src.models.diffusion.base import VPDynamics, load_score_model

        config = load_config(args.config)
        config.device = device
        model = load_score_model(config, device)
        dyn = VPDynamics(model, img_shape=spec.img_shape).to(device)

        def eps_fn(x, t_vec):
            score = dyn.score(t_vec, x.flatten(1)).view_as(x)
            return -score * torch.sqrt(1 - alpha_bar_cont(t_vec)).to(x).view(-1, 1, 1, 1)

    bank = build_prototype_bank(
        encoder, batches, n_prototypes=args.n_prototypes, t_values=t_values, eps_fn=eps_fn,
        device=device, seed=args.seed,
        meta={"encoder": encoder_name, "encoder_feature": feature, "domain": args.domain,
              "config": args.config, "train_indices_seed": args.seed})
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    bank.save(out)
    logger.info("saved bank %s: %s, fingerprint %s", out, tuple(bank.prototypes.shape), bank.fingerprint())


if __name__ == "__main__":
    main()
