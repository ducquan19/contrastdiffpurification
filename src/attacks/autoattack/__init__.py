"""AutoAttack (Croce & Hein, ICML 2020), from https://github.com/fra31/auto-attack.

Reduced to what this project runs: the ``standard`` and ``rand`` versions, Linf / L2, PyTorch
models. ``run_standard_evaluation`` accepts ``state_path`` so an interrupted evaluation can resume.
License: LICENSE_AUTOATTACK.
"""
from .autoattack import AutoAttack

__all__ = ["AutoAttack"]
