"""
common/thresholds.py — out-of-fold classification-threshold selection.

Shared replacement for the threshold-selection cell duplicated across the GEC and
GELSTM notebooks. Two strategies are offered:

    * Youden's J  — maximises sensitivity + specificity on the ROC curve.
    * Best-F1     — maximises F1 over candidate thresholds.

**Leakage contract (`.claude/rules/evaluation.md`).** Every function here derives a
threshold *from the data passed in*. Pass it the pooled out-of-fold (validation)
predictions only — NEVER the test set. Test-set evaluation must reuse the
validation-derived threshold. ``select_oof_threshold`` defaults to Best-F1, which is
the notebook default per ``.claude/rules/notebooks.md`` (option 1 / Enter).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_curve


def youden_threshold(targets, probs) -> float:
    """Youden's J threshold (argmax of TPR − FPR) from the ROC curve.

    Returns 0.5 when only one class is present (ROC is undefined).
    """
    targets = np.asarray(targets, dtype=int)
    probs = np.asarray(probs, dtype=float)
    if len(np.unique(targets)) < 2:
        return 0.5
    fpr, tpr, thrs = roc_curve(targets, probs)
    return float(thrs[int(np.argmax(tpr - fpr))])


def best_f1_threshold(targets, probs) -> float:
    """Threshold maximising F1 over the ROC-curve candidate thresholds.

    Returns 0.5 when only one class is present.
    """
    targets = np.asarray(targets, dtype=int)
    probs = np.asarray(probs, dtype=float)
    if len(np.unique(targets)) < 2:
        return 0.5
    _, _, thrs = roc_curve(targets, probs)
    f1s = [f1_score(targets, (probs >= t).astype(int), zero_division=0) for t in thrs]
    return float(thrs[int(np.argmax(f1s))])


def oof_threshold_metrics(targets, probs, thr) -> Tuple[float, float, float]:
    """(sensitivity, specificity, f1) of ``probs >= thr`` against ``targets``."""
    targets = np.asarray(targets, dtype=int)
    probs = np.asarray(probs, dtype=float)
    pred = (probs >= thr).astype(int)
    if len(np.unique(targets)) > 1:
        tn, fp, fn, tp = confusion_matrix(targets, pred).ravel()
    else:
        tn = fp = fn = tp = 0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(targets, pred, zero_division=0)
    return float(sens), float(spec), float(f1)


def select_oof_threshold(
    targets,
    probs,
    *,
    threshold_mode: str | None = None,
    fixed_threshold: float | None = None,
    runner_active: bool = False,
    interactive: bool = True,
) -> Tuple[float, str]:
    """Resolve ``(active_threshold, method)`` from out-of-fold predictions.

    Prints both options with their OOF sens/spec/F1, then selects:

      threshold_mode == 'best-f1' -> Best-F1            (method 'oof_f1')
      threshold_mode == 'youden'  -> Youden's J         (method 'oof_youden')
      threshold_mode == 'fixed'   -> fixed_threshold    (method 'fixed'; required)
      threshold_mode is None      -> prompt (Best-F1 is the default / Enter),
                                     unless ``runner_active`` then raise (the
                                     experiment runner must set threshold_mode).

    Best-F1 is the default per ``.claude/rules/notebooks.md``.
    """
    f1_thr = best_f1_threshold(targets, probs)
    youden_thr = youden_threshold(targets, probs)

    f_s, f_sp, f_f1 = oof_threshold_metrics(targets, probs, f1_thr)
    y_s, y_sp, y_f1 = oof_threshold_metrics(targets, probs, youden_thr)
    print("OOF threshold options:")
    print(
        f"  [1] Best-F1 (default) thr={f1_thr:.4f}  "
        f"sens={f_s:.3f}  spec={f_sp:.3f}  F1={f_f1:.3f}"
    )
    print(
        f"  [2] Youden            thr={youden_thr:.4f}  "
        f"sens={y_s:.3f}  spec={y_sp:.3f}  F1={y_f1:.3f}"
    )

    if threshold_mode == "best-f1":
        active, method = f1_thr, "oof_f1"
    elif threshold_mode == "youden":
        active, method = youden_thr, "oof_youden"
    elif threshold_mode == "fixed":
        if fixed_threshold is None:
            raise ValueError("threshold_mode='fixed' requires fixed_threshold=")
        active, method = float(fixed_threshold), "fixed"
    elif threshold_mode is None:
        if runner_active:
            raise ValueError(
                "threshold_mode is required under the experiment runner "
                "(youden | best-f1 | fixed). Set 'threshold_mode:' in experiments.yaml."
            )
        if not interactive:
            # Non-interactive standalone use falls back to the documented default.
            active, method = f1_thr, "oof_f1"
        else:
            choice = input("Select threshold [1=Best-F1 (default), 2=Youden]: ").strip()
            if choice == "2":
                active, method = youden_thr, "oof_youden"
            else:
                active, method = f1_thr, "oof_f1"
    else:
        raise ValueError(
            f"Unknown threshold_mode={threshold_mode!r} "
            "(expected one of: best-f1, youden, fixed, None)."
        )

    print(f"Using {method} threshold: {active:.4f}")
    return float(active), method
