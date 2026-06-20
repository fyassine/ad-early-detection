"""
common/early_detection.py — post-test longitudinal analyses over the test bundle.

Two shared routines that re-run a trained model across the held-out test set using
the model's hooks (so they stay model-agnostic):

    * ``early_detection_table`` — "if we only had the first N visits, how well could
      we detect conversion?" For each N it truncates every eligible test subject to
      their first N visits and re-scores.
    * ``trajectory_frame``      — per-visit P(converter) for each multi-visit subject,
      as a tidy DataFrame ready for ``common.plots.plot_conversion_trajectories``.

Both reuse the trained model and its validation-derived threshold; neither derives a
new threshold, so there is no test-set leakage.
"""
from __future__ import annotations

from typing import Any, Callable, List

import numpy as np
import pandas as pd


def early_detection_table(
    test_bundle,
    eval_split: Callable[..., dict],
    truncate_to_n_visits: Callable[[Any, int], Any],
    state_dict: Any,
    threshold: float,
    *,
    device: Any,
    min_subjects: int = 4,
) -> List[dict]:
    """AUC / sensitivity / specificity as a function of the number of visits used.

    For ``N = 1 .. max_visits`` (max visit count in the bundle), restrict every
    subject with at least ``N`` visits to their first ``N`` (via the
    ``truncate_to_n_visits`` hook) and re-score with ``eval_split`` at the fixed
    ``threshold``. Rows with fewer than ``min_subjects`` subjects or a single class
    are skipped (same guard as the source notebooks). Prints the table and returns
    the rows as dicts.
    """
    if not test_bundle.items:
        return []
    max_scans = max(int(item["n_scans"]) for item in test_bundle.items)

    print(f'\n{"Visits":>6} {"N":>4} {"AUC":>8} {"Sens":>8} {"Spec":>8}')
    print("-" * 40)

    rows: List[dict] = []
    for n_vis in range(1, max_scans + 1):
        sub_bundle = truncate_to_n_visits(test_bundle, n_vis)
        if len(sub_bundle.items) < min_subjects or len(np.unique(sub_bundle.labels)) < 2:
            continue
        m = eval_split(state_dict, sub_bundle, threshold, device=device)
        row = {
            "n_visits": n_vis,
            "n_subjects": len(sub_bundle.items),
            "auc": float(m["auc"]),
            "sensitivity": float(m["sensitivity"]),
            "specificity": float(m["specificity"]),
        }
        rows.append(row)
        print(
            f"{n_vis:>6} {row['n_subjects']:>4} {row['auc']:>8.4f} "
            f"{row['sensitivity']:>8.3f} {row['specificity']:>8.3f}"
        )
    return rows


def trajectory_frame(
    test_bundle,
    per_visit_probs: Callable[..., list],
    state_dict: Any,
    *,
    device: Any,
    min_visits: int = 2,
) -> pd.DataFrame:
    """Build the per-visit conversion-probability frame for multi-visit subjects.

    For every subject with at least ``min_visits`` visits, calls the
    ``per_visit_probs`` hook (returning ``[(month, prob), ...]`` for prefix
    sequences of length 1..T) and stacks the results. Columns: ``pid``, ``label``,
    ``month``, ``prob``.
    """
    records: List[dict] = []
    for item in test_bundle.items:
        if int(item["n_scans"]) < min_visits:
            continue
        for month, prob in per_visit_probs(state_dict, item, device=device):
            records.append(
                {
                    "pid": item["subject_id"],
                    "label": item["label"],
                    "month": month,
                    "prob": prob,
                }
            )
    return pd.DataFrame(records, columns=["pid", "label", "month", "prob"])
