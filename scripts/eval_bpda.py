"""Evaluate diffusion purification (``--guidance none`` or ``contrastive``) against BPDA+EOT.

Run from the repository root:

    python -m scripts.eval_bpda --guidance none --config cifar10.yml ...
"""
from src.evaluation.robustness import bpda_robustness_eval, build_bpda_parser, setup


def parse_args_and_config():
    return setup(build_bpda_parser().parse_args())


if __name__ == '__main__':
    args, config = parse_args_and_config()
    bpda_robustness_eval(args, config)
