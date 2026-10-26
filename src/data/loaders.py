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
EVAL_SPLITS = ("test", "val", "dev")

# ``--eval_split dev`` draws from the *test* images that are NOT in the reported subset (the first
# REPORT_SIZE images of ``data_seed`` REPORT_SEED), so hyperparameters are chosen on test-distribution
# images the pretrained models never saw, without touching the images that are reported.
REPORT_SIZE = 512
REPORT_SEED = 0

# The last VAL_SIZE images of the *training* split are held out as a validation pool:
# ``--eval_split val`` draws the evaluated subset from it (hyperparameter tuning, never on the test
# set), and the PCP prototype bank never uses it (scripts/build_prototype_bank.py).
VAL_SIZE = 5000


def val_pool_indices(n_train: int) -> np.ndarray:
    """Training-split indices of the held-out validation pool (the last ``VAL_SIZE`` images)."""
    if n_train <= VAL_SIZE:
        raise ValueError(f"training split of {n_train} images is too small for a {VAL_SIZE}-image pool")
    return np.arange(n_train - VAL_SIZE, n_train)


def dev_pool_indices(n_test: int) -> np.ndarray:
    """Test indices outside the reported subset (``select_subset_indices(n, REPORT_SIZE, REPORT_SEED)``)."""
    report = select_subset_indices(n_test, min(REPORT_SIZE, n_test), REPORT_SEED)
    return np.setdiff1d(np.arange(n_test), report)


def bank_pool_size(n_train: int) -> int:
    """Training images (indices ``0 .. n - 1``) that may be used to build the prototype bank."""
    return n_train - VAL_SIZE


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
    split: str = "test",
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

    ``split='test'`` (default) is the evaluation subset of the published protocol. ``split='val'``
    draws the subset, with the same rule, from the held-out validation pool of the *training*
    split (``val_pool_indices``); ``indices`` are then training-split indices. Use it to tune.
    """
    if split not in EVAL_SPLITS:
        raise ValueError(f"unknown split '{split}'; available: {', '.join(EVAL_SPLITS)}")
    spec = get_dataset_spec(dataset)

    if split == "test":
        test_set = spec.build_split(train=False)
        n_pool = len(test_set)
        indices = select_subset_indices(
            n_total=len(test_set),
            num_sub=num_sub,
            data_seed=data_seed
        )
    elif split == "dev":
        test_set = spec.build_split(train=False)
        pool = dev_pool_indices(len(test_set))
        n_pool = len(pool)
        indices = pool[select_subset_indices(n_total=n_pool, num_sub=num_sub, data_seed=data_seed)]
    else:
        test_set = spec.build_split(train=True)    # the training split; name kept for the code below
        pool = val_pool_indices(len(test_set))
        n_pool = len(pool)
        indices = pool[select_subset_indices(n_total=n_pool, num_sub=num_sub, data_seed=data_seed)]

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
            "split": split,
            {"test": "test_split_size", "val": "val_pool_size", "dev": "dev_pool_size"}[split]: n_pool,
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

    x_val, y_val, _ = load_eval_subset(args.domain, args.num_sub, args.data_seed,
                                       split=getattr(args, "eval_split", "test"), log_dir=log_dir)
    return x_val, y_val
