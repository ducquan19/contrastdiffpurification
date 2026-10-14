"""Loading the images a robustness evaluation runs on.

An evaluation scores a fixed subset of ``num_sub`` test images selected by ``data_seed``, returned
as a single tensor because the attacks need the whole subset at once.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.data.specs import get_dataset_spec

EVAL_INDICES_FILE = "eval_indices.json"


def select_subset_indices(
    n_total: int,
    num_sub: int,
    data_seed: int
) -> np.ndarray:
    """The ``num_sub`` test indices evaluated for ``data_seed``.

    Depends only on its arguments, so the subset is reproducible across machines and independent
    of the global RNG state. The order is the one ``choice`` returns and is **not** sorted: it
    decides which images land in which attack batch.
    """

    if num_sub <= 0:
        raise ValueError(
            f"num_sub must be positive, got {num_sub}"
        )

    if num_sub > n_total:
        raise ValueError(
            f"num_sub={num_sub} exceeds the test split size ({n_total})"
        )

    return np.random.RandomState(data_seed).choice(
        n_total,
        num_sub,
        replace=False
    )


def load_eval_subset(
    dataset: str,
    num_sub: int,
    data_seed: int,
    *,
    log_dir: str | Path | None = None,
    num_workers: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Load one deterministic evaluation subset.

    Returns:
        A tuple ``(x, y, indices)`` where:

        - ``x`` contains float tensors in ``[0, 1]`` with shape ``(num_sub, C, H, W)``.
        - ``y`` contains the corresponding labels.
        - ``indices`` contains the selected test-set indices in evaluation order.

    If ``log_dir`` is given, the indices and the parameters that selected them are written to
    ``eval_indices.json`` in that directory.

    The DataLoader consumes the global torch RNG, so changing ``num_workers`` (or calling this
    at a different point of the run) changes the diffusion noise drawn afterwards.
    """
    spec = get_dataset_spec(dataset)
    test_set = spec.build_split(train=False)

    indices = select_subset_indices(
        n_total=len(test_set),
        num_sub=num_sub,
        data_seed=data_seed
    )

    subset = Subset(test_set, indices.tolist())

    loader = DataLoader(
        subset,
        batch_size=len(subset),
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers
    )

    x_val, y_val = next(iter(loader))

    x_val = x_val.contiguous().requires_grad_(True)
    y_val = y_val.contiguous()

    if log_dir is not None:
        record = {
            "dataset": spec.name,
            "num_sub": int(num_sub),
            "data_seed": int(data_seed),
            "test_split_size": len(test_set),
            "indices": [int(i) for i in indices],
        }

        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        (log_path / EVAL_INDICES_FILE).write_text(json.dumps(record, indent=2), encoding="utf-8")

    return x_val, y_val, indices


def load_data(
    args: Namespace,
    log_dir: str | Path | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``load_eval_subset`` driven by the parsed command line. Returns ``(x, y)``."""

    x_val, y_val, _ = load_eval_subset(args.domain, args.num_sub, args.data_seed, log_dir=log_dir)
    return x_val, y_val
