"""T7: no unmarked ``detach`` / ``no_grad`` on the data path of the full-gradient code.

A full-gradient evaluation is only as strong as the weakest link in the graph: one stray
``.detach()`` in the purifier or the guidance turns it into a BPDA-like attack without any visible
error. Every such call in the files below must carry a ``# SAFE: <reason>`` comment on its line.
Only code tokens are checked (docstrings and comments may mention the words).
"""
import io
import tokenize
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKED = [
    "src/models/diffusion/sampler.py",
    "src/guidance/particle.py",
    "src/guidance/memory_bank.py",
    "src/evaluation/particle_model.py",
    "src/evaluation/attacks_full.py",
    "src/evaluation/voting.py",
    "src/guidance/spectral.py",
]
SUSPICIOUS = {"detach", "detach_", "no_grad", "inference_mode"}


@pytest.mark.parametrize("rel", CHECKED)
def test_every_gradient_cut_is_marked(rel):
    src = (ROOT / rel).read_text(encoding="utf-8")
    lines = src.splitlines()
    offenders = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.NAME and tok.string in SUSPICIOUS:
            line = lines[tok.start[0] - 1]
            if "# SAFE" not in line:
                offenders.append(f"{rel}:{tok.start[0]}: {line.strip()}")
    assert not offenders, "unmarked gradient cuts:\n" + "\n".join(offenders)
