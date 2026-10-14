"""Evaluate diffusion purification (``--guidance none`` or ``contrastive``) against
AutoAttack / StAdv / EOT-PGD / APGD.

Run from the repository root:

    python -m scripts.eval_robustness --guidance contrastive --config cifar10.yml ...
"""

from src.evaluation.robustness import build_robustness_parser, robustness_eval, setup


def parse_args_and_config():
    return setup(build_robustness_parser().parse_args())


if __name__ == '__main__':
    args, config = parse_args_and_config()
    robustness_eval(args, config)
