# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DiffPure. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
"""Evaluation drivers and command-line arguments for ``scripts.eval_robustness`` and
``scripts.eval_bpda``.

``setup`` loads the configs and seeds everything; ``robustness_eval`` / ``bpda_robustness_eval``
build the purified model, load the evaluation subset and run the selected attack. Results go to
``<exp>/<image_folder>/<classifier>/<diffusion>_<attack>/seed<seed>/data<data_seed>/``.
"""

import argparse
import json
import logging
import os
import random

import numpy as np
import torch
import torch.distributed as dist

from src.config import load_config
from src.data.loaders import load_data
from src.data.specs import DATASET_NAMES
from src.evaluation.attacks import eval_apgd, eval_autoattack, eval_bpda, eval_eotpgd, eval_stadv
from src.evaluation.purified_model import PurifiedClassifier
from src.method import CLI_FIELDS, DEFAULT_METHOD_CONFIG, ESTIMATORS, cli_dest
from src.method import resolve_contrastive
from src.paths import output_path
from src.utils import Logger, str2bool

GUIDANCE_CHOICES = ['none', 'contrastive']

ATTACKS = {
    'standard': eval_autoattack,
    'rand': eval_autoattack,
    'stadv': eval_stadv,
    'eotpgd': eval_eotpgd,
    'apgd': eval_apgd,
}


def record_run_config(args, log_dir):
    """Save the resolved configuration for reproducibility."""
    record = {
        "args": {
            key: value
            for key, value in sorted(vars(args).items())
            if isinstance(value, (int, float, str, bool, type(None)))
        },
        "diffusion_config": args.config,
        "contrastive": (
            args.method_config.as_dict()
            if args.method_config is not None
            else None
        ),
    }

    with open(os.path.join(log_dir, "resolved_config.json"), "w") as f:
        json.dump(record, f, indent=2)


def _start_run(args, middle_name):
    """Create the run's log folder, start mirroring stdout to ``log.txt`` and record the config."""
    log_dir = os.path.join(
        args.image_folder,
        args.classifier_name,
        middle_name,
        f"seed{args.seed}",
        f"data{args.data_seed}"
    )
    os.makedirs(log_dir, exist_ok=True)
    args.log_dir = log_dir

    logger = Logger(file_name=f'{log_dir}/log.txt', file_mode="w+", should_flush=True)
    record_run_config(args, log_dir)
    return log_dir, logger


def _build_model(args, config, ngpus):
    """The purified classifier in eval mode, wrapped in ``DataParallel`` on several GPUs."""
    print("starting the model and loader...")
    model = PurifiedClassifier(args, config)
    if ngpus > 1:
        model = torch.nn.DataParallel(model)
    return model.eval().to(config.device)


def robustness_eval(args, config):
    """Run AutoAttack / StAdv / EOT-PGD / APGD (``args.attack_version``) against the purifier."""
    contrastive = args.guidance == 'contrastive'

    parts = [args.diffusion_type, args.attack_version]
    if args.attack_version not in ["stadv", "standard", "rand"]:
        parts.append(args.attack_type)
    log_dir, logger = _start_run(args, "_".join(parts))

    ngpus = torch.cuda.device_count()

    if contrastive and ngpus > 1 and (args.contrastive_data_drift != -1):
        raise ValueError("contrastive_data_drift is not supported for multi-gpu")

    adv_batch_size = args.adv_batch_size * ngpus
    print(f"ngpus: {ngpus}, adv_batch_size: {adv_batch_size}")

    # The data/model order differs between the two modes and is kept that way on purpose:
    # the DataLoader and the model construction both consume the global torch RNG, so swapping
    # them would change the diffusion noise and therefore the published numbers.
    if contrastive:
        x_val, y_val = load_data(args, log_dir=log_dir)

        # --contrastive_data_drift k: evaluate only the k-th attack batch of the subset
        if args.contrastive_data_drift >= 0:
            lo = args.contrastive_data_drift * args.adv_batch_size
            hi = lo + args.adv_batch_size
            print(f"use data drift mode; pick x_val[{lo}:{hi}]")
            x_val, y_val = x_val[lo:hi], y_val[lo:hi]

        model = _build_model(args, config, ngpus)
    else:
        model = _build_model(args, config, ngpus)
        x_val, y_val = load_data(args, log_dir=log_dir)

    if args.attack_version not in ATTACKS:
        raise NotImplementedError(f"unknown attack_version: {args.attack_version}")
    ATTACKS[args.attack_version](args, config, model, x_val, y_val, adv_batch_size, log_dir)

    logger.close()

    if not contrastive and ngpus > 1 and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def bpda_robustness_eval(args, config):
    """Run BPDA+EOT against the purifier."""
    log_dir, logger = _start_run(args, '_'.join([args.diffusion_type, 'bpda']))

    ngpus = torch.cuda.device_count()
    adv_batch_size = args.adv_batch_size * ngpus
    print(f"ngpus: {ngpus}, adv_batch_size: {adv_batch_size}")

    model = _build_model(args, config, ngpus)
    x_val, y_val = load_data(args, log_dir=log_dir)

    eval_bpda(args, config, model, x_val, y_val, adv_batch_size, log_dir)

    logger.close()


def _add_common_args(parser, adv_batch_size, num_sub):
    """Arguments shared by both evaluation entry points (defaults differ per entry point)."""
    parser.add_argument(
        "--guidance",
        type=str,
        required=True,
        choices=GUIDANCE_CHOICES,
        help="none = DiffPure baseline, contrastive = contrastive guidance"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Config file name in configs/ (or a path)")
    parser.add_argument("--data_seed", type=int, default=0,
                        help="Seed that selects the evaluation subset of the test set")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Seed of torch / numpy / random (diffusion noise, attacks)")
    parser.add_argument("--exp", type=str, default='exp',
                        help="Path for saving running related data (relative paths go under outputs/).")
    parser.add_argument("--verbose", type=str, default='info',
                        help="Verbose level: info | debug | warning | critical")
    parser.add_argument("-i", "--image_folder", type=str, default='images',
                        help="The folder name of samples")

    # data and classifier
    parser.add_argument('--domain', type=str, default='cifar10', choices=DATASET_NAMES,
                        help='which dataset to evaluate on')
    parser.add_argument('--classifier_name', type=str, default='cifar10-wideresnet-28-10',
                        help='which classifier to use')
    parser.add_argument('--num_sub', type=int, default=num_sub,
                        help='number of test images (per data_seed)')
    parser.add_argument('--adv_batch_size', type=int, default=adv_batch_size)
    parser.add_argument('--adv_eps', type=float, default=0.031373)

    # purification
    parser.add_argument("--sample_step", type=int, default=1, help="Total sampling steps")
    parser.add_argument("--t", type=int, default=100, help="Sampling noise scale")
    parser.add_argument("--t_delta", type=int, default=15,
                        help="Perturbation range of sampling noise scale")
    parser.add_argument("--rand_t", type=str2bool, default=False,
                        help="Decide if randomize sampling noise scale")
    parser.add_argument("--fix_rand", type=str2bool, default=False,
                        help="use one fixed noise sample for the forward diffusion")
    parser.add_argument("--diffusion_type", type=str, default='sde', choices=['sde', 'ode'],
                        help="reverse SDE (DiffPure) or probability-flow ODE")
    parser.add_argument("--score_type", type=str, default='score_sde', choices=['score_sde'],
                        help="score network type (only score_sde / NCSN++ is supported)")
    parser.add_argument('--step_size', type=float, default=1e-3,
                        help='step size for ODE Euler method')
    parser.add_argument("--use_bm", action='store_true', help="whether to use brownian motion")
    parser.add_argument("--eot_iter", type=int, default=20,
                        help="only for rand version of autoattack")

    _add_contrastive_args(parser)


_CONTRASTIVE_HELP = {
    "classifier": ("NAME", "backbone whose logits the feature extractor F(.) embeds"),
    "n_classes": ("N", "F(.) input width (default: the dataset number of classes)"),
    "embed_lr": ("LR", "Adam learning rate of F(.)"),
    "drift_min": ("STEP", "first reverse step that receives guidance"),
    "drift_max": ("STEP", "one past the last reverse step that receives guidance"),
    "tau_minus": ("TAU_MINUS", "tau^- in Eq. (10)"),
    "beta": ("BETA", "beta in Eq. (10): hardness of the negative mining"),
    "temperature": ("TAU", "tau: temperature of the contrastive loss"),
    "guidance_strength": ("LAMBDA", "lambda in Eq. (8): strength of the guidance"),
    "estimator": ("{hard,soft}", "'hard' = HNM loss Eq. (10), 'soft' = InfoNCE Eq. (9)"),
}


def _add_contrastive_args(parser):
    """Add ``--contrastive_<field>`` for every field of ``ContrastiveConfig``.

    The default ``None`` keeps the value from the method config file (``--method``).
    """
    group = parser.add_argument_group(
        "contrastive guidance (--guidance contrastive)",
        "Hyperparameters of the method; unset values come from --method."
    )
    group.add_argument("--method", type=str, default=DEFAULT_METHOD_CONFIG,
                       help="method config under configs/ (default: %(default)s)")
    for name, type_ in CLI_FIELDS.items():
        metavar, help_ = _CONTRASTIVE_HELP[name]
        group.add_argument(
            f"--{cli_dest(name)}", type=type_, default=None, metavar=metavar, help=help_,
            choices=ESTIMATORS if name == "estimator" else None)

    parser.add_argument('--contrastive_data_drift', type=int, default=-1,
                        help='developer option: evaluate only the k-th batch of size '
                             '--adv_batch_size (-1 = all)')


def build_robustness_parser():
    """CLI of ``scripts.eval_robustness``."""
    parser = argparse.ArgumentParser(
        description='AutoAttack / StAdv / EOT-PGD / APGD evaluation of diffusion purification'
    )
    _add_common_args(parser, adv_batch_size=64, num_sub=64)
    parser.add_argument('--attack_type', type=str, default='square',
                        help='only used in the output folder name of eotpgd / apgd runs')
    parser.add_argument('--lp_norm', type=str, default='Linf', choices=['Linf', 'L2'])
    parser.add_argument('--attack_version', type=str, default='rand', choices=list(ATTACKS))
    return parser


def build_bpda_parser():
    """CLI of ``scripts.eval_bpda``."""
    parser = argparse.ArgumentParser(
        description='BPDA+EOT evaluation of diffusion purification'
    )
    _add_common_args(parser, adv_batch_size=10, num_sub=200)
    parser.add_argument('--eot_defense_reps', type=int, default=150)
    parser.add_argument('--eot_attack_reps', type=int, default=15)
    return parser


def setup(args):
    """Load the configs, configure logging, create the output folder and seed everything."""
    new_config = load_config(args.config)
    args.method_config = (
        resolve_contrastive(args)
        if args.guidance == 'contrastive'
        else None
    )

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            '%(levelname)s - %(filename)s - %(asctime)s - %(message)s'
        )
    )
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(level)

    if not os.path.isabs(args.exp):
        args.exp = os.path.normpath(output_path(args.exp))

    args.image_folder = os.path.join(args.exp, args.image_folder)
    os.makedirs(args.image_folder, exist_ok=True)

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    new_config.device = device

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config
