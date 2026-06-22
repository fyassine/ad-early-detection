"""
CLASSIFIER/model/utils/metrics.py

Subject-level aggregation of scan-level predictions.

A subject that contributes N scans appears N times in the scan-level evaluation,
which inflates AUC when one subject's scans cluster together (easy or hard).
This helper reduces scan-level (probs, labels) to one row per subject so that
the reported AUC reflects subject-wise generalisation.
"""
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_VALID_REDUCERS = {"mean", "median", "max", "last"}


def aggregate_scan_to_subject(
    probs: Iterable[float],
    scan_subject_ids: Iterable[str],
    scan_labels: Iterable[int],
    reduce: str = "mean",
    scan_order: "Iterable[float] | None" = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reduce per-scan predictions to per-subject predictions.

    Parameters
    ----------
    probs : array-like of float
        Scan-level predicted probabilities.
    scan_subject_ids : array-like
        Subject ID for each scan (used to group).
    scan_labels : array-like of int
        Scan-level label. Must be constant within a subject (asserted).
    reduce : {"mean", "median", "max", "last"}
        Aggregation across a subject's scans.
        "last" requires scan_order so the latest scan can be picked.
    scan_order : array-like of float, optional
        Sortable value (e.g. visit month) used only when reduce="last".

    Returns
    -------
    subject_ids : np.ndarray  (S,) — unique subject IDs in stable input order
    subject_probs : np.ndarray (S,) — aggregated probability per subject
    subject_labels : np.ndarray (S,) — label per subject
    """
    if reduce not in _VALID_REDUCERS:
        raise ValueError(f"reduce must be one of {_VALID_REDUCERS}, got {reduce!r}")

    df = pd.DataFrame({
        "sid":   list(scan_subject_ids),
        "prob":  np.asarray(probs, dtype=float),
        "label": np.asarray(scan_labels, dtype=int),
    })
    if reduce == "last":
        if scan_order is None:
            raise ValueError("reduce='last' requires scan_order")
        df["order"] = np.asarray(scan_order, dtype=float)

    label_var = df.groupby("sid")["label"].nunique()
    bad = label_var[label_var > 1]
    if len(bad) > 0:
        raise ValueError(
            f"{len(bad)} subjects have inconsistent labels across scans: "
            f"{bad.index.tolist()[:5]}"
        )

    if reduce == "mean":
        agg_prob = df.groupby("sid", sort=False)["prob"].mean()
    elif reduce == "median":
        agg_prob = df.groupby("sid", sort=False)["prob"].median()
    elif reduce == "max":
        agg_prob = df.groupby("sid", sort=False)["prob"].max()
    else:
        agg_prob = (
            df.sort_values("order")
              .groupby("sid", sort=False)["prob"]
              .last()
        )

    label_per_sid = df.groupby("sid", sort=False)["label"].first()

    sids   = agg_prob.index.to_numpy()
    return sids, agg_prob.to_numpy(), label_per_sid.loc[sids].to_numpy()


def subject_level_auc(
    probs, scan_subject_ids, scan_labels, reduce: str = "mean",
    scan_order=None,
) -> float:
    """One-shot helper: aggregate then compute AUC."""
    _, p, y = aggregate_scan_to_subject(
        probs, scan_subject_ids, scan_labels,
        reduce=reduce, scan_order=scan_order,
    )
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))
