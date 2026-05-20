"""
Centralized subject-level train/val/test split logic.

All notebooks and scripts should use ``make_splits`` rather than inlining
``train_test_split`` calls so partitions are consistent and reproducible.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from sklearn.model_selection import train_test_split


def make_splits(
    subject_ids: Sequence,
    labels: Optional[Sequence] = None,
    *,
    seed: int,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    stratify: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Subject-level train/val/test split with a fixed seed.

    Splits indices into the input ``subject_ids`` sequence (caller is
    responsible for keeping subject_ids and labels aligned). Each subject
    appears in exactly one partition, so no leakage occurs at the subject
    level.

    Returns
    -------
    dict with keys ``train``, ``val``, ``test``, each an ``np.ndarray`` of
    integer indices into ``subject_ids``.
    """
    if not 0.0 < val_frac < 1.0 or not 0.0 < test_frac < 1.0:
        raise ValueError("val_frac and test_frac must each be in (0, 1)")
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must be < 1.0")

    n = len(subject_ids)
    indices = np.arange(n)
    strat = np.asarray(labels) if (stratify and labels is not None) else None

    trainval_idx, test_idx = train_test_split(
        indices,
        test_size=test_frac,
        random_state=seed,
        stratify=strat,
    )

    val_size_relative = val_frac / (1.0 - test_frac)
    strat_trainval = strat[trainval_idx] if strat is not None else None
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=val_size_relative,
        random_state=seed,
        stratify=strat_trainval,
    )

    return {
        "train": np.asarray(train_idx),
        "val": np.asarray(val_idx),
        "test": np.asarray(test_idx),
    }
