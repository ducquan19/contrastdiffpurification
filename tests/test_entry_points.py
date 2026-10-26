"""Import and argument-parsing smoke tests for the entry points.

Needs the full environment (torchvision, torchsde, torchattacks, autoattack, robustbench, score_sde, ...);
skipped otherwise. No GPU, data or checkpoints are needed.
"""
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for mod in ("torchvision", "torchsde", "torchdiffeq", "torchattacks", "autoattack", "robustbench",
            "timm", "score_sde"):
    pytest.importorskip(mod)


def test_evaluation_modules_import():
    import src.evaluation.attacks  # noqa: F401
    import src.evaluation.robustness  # noqa: F401
    import scripts.eval_bpda  # noqa: F401
    import scripts.eval_robustness  # noqa: F401


def _python_command(script):
    """Return the CLI tokens of the `python -m ...` call in an experiment script (loops expanded once)."""
    text = script.read_text()
    text = re.sub(r"CUDA_VISIBLE_DEVICES=\$cuda_index python -m (\S+)", r"echo \1", text)
    text = re.sub(r"^cd .*$", "", text, flags=re.M)
    # bytes, not text: on Windows a text-mode pipe would turn "\n" into "\r\n" and break bash
    r = subprocess.run(["bash", "-s", "0", "121", "0", "100"], input=text.encode(), capture_output=True,
                       check=True)
    return r.stdout.decode().splitlines()[0].split()


@pytest.mark.parametrize("script", sorted((ROOT / "experiments").glob("*/run_*.sh"))
                         + sorted((ROOT / "experiments" / "ablation_t").glob("*.sh")), ids=lambda p: p.name)
def test_experiment_arguments_parse(script):
    from src.evaluation.robustness import build_bpda_parser, build_robustness_parser
    module, *argv = _python_command(script)
    parser = build_bpda_parser() if module == "scripts.eval_bpda" else build_robustness_parser()
    args = parser.parse_args(argv)
    assert args.guidance in ("none", "contrastive", "pcp", "srsg", "freqpure")
    assert args.diffusion_type in ("sde", "ode", "particle")
    assert args.domain == "cifar10"
    if args.guidance == "pcp":
        assert args.diffusion_type == "particle"


@pytest.mark.parametrize("module", ["scripts.eval_robustness", "scripts.eval_bpda", "scripts.train_score_sde",
                                    "scripts.train_classifier", "scripts.convert_lightning_checkpoint"])
def test_cli_help(module):
    pytest.importorskip("pytorch_lightning")
    r = subprocess.run([sys.executable, "-m", module, "--help"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
