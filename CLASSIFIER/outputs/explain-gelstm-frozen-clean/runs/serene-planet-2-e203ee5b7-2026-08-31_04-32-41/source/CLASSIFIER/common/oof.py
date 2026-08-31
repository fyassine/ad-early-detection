"""
common/oof.py — the out-of-fold (OOF) evaluation frame and Tier 1-3 metrics.

``DOCS/temporal-first-ablation.md``'s 2026-08-24 "Evaluation & Comparison
Protocol" addendum moves the ladder's stopping rule, floor gates, and
robustness vetoes onto pooled CV out-of-fold predictions — never the
in-domain test set or OASIS-3, both of which are read exactly once, after the
ladder is frozen (see ``common.frozen_read`` / the comparison notebook).

This module is pure — no I/O, no torch. ``build_oof_frame`` turns a
``common.crossval.CVResult`` plus the CV-pool ``Bundle`` into a tidy per-subject
frame; ``oof_metrics`` computes everything the addendum's Tier 1-3 tables need
from that frame alone. ``common.run_artifacts.record_oof_artifacts`` persists
both as run artifacts.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)

_FRAME_COLUMNS = ["subject_id", "fold", "cohort", "label", "prob", "n_scans", "age", "sex"]


def build_oof_frame(
    bundle: Any,
    oof_sids: Sequence[Any],
    oof_probs: np.ndarray,
    oof_targets: np.ndarray,
    oof_folds: np.ndarray,
    oof_extras: Optional[Dict[str, np.ndarray]] = None,
    *,
    default_cohort: Optional[str] = None,
) -> pd.DataFrame:
    """Tidy per-subject OOF frame from a ``CVResult`` and its source ``Bundle``.

    ``bundle`` is the CV-pool ``Bundle`` (post-``prepare_data``); its items
    supply ``n_scans`` / ``age`` / ``sex`` and, for a pooled multi-cohort run,
    ``cohort`` (``common.pooled_data.build_multicohort_bundle`` tags every
    item with one). ``oof_sids`` / ``oof_probs`` / ``oof_targets`` /
    ``oof_folds`` / ``oof_extras`` come straight off ``CVResult`` and share one
    row order. ``default_cohort`` is required for a single-cohort bundle whose
    items carry no ``cohort`` key — per-cohort OOF metrics need one label or
    the other, never a silent fallback (``.claude/rules/errors.md``).
    """
    n = len(oof_sids)
    if not (len(oof_probs) == len(oof_targets) == len(oof_folds) == n):
        raise ValueError(
            "build_oof_frame: oof_sids/oof_probs/oof_targets/oof_folds must have "
            f"matching length, got {n}, {len(oof_probs)}, {len(oof_targets)}, "
            f"{len(oof_folds)}."
        )

    by_sid: Dict[Any, Any] = {}
    for it in bundle.items:
        sid = it["subject_id"]
        if sid in by_sid:
            raise ValueError(f"build_oof_frame: duplicate subject_id {sid!r} in bundle.items.")
        by_sid[sid] = it

    extras = oof_extras or {}
    for name, arr in extras.items():
        if len(arr) != n:
            raise ValueError(
                f"build_oof_frame: oof_extras[{name!r}] has length {len(arr)}, expected {n}."
            )

    rows = []
    for i, sid in enumerate(oof_sids):
        if sid not in by_sid:
            raise ValueError(f"build_oof_frame: OOF subject_id {sid!r} not found in bundle.items.")
        it = by_sid[sid]
        cohort = it.get("cohort", default_cohort)
        if cohort is None:
            raise ValueError(
                f"build_oof_frame: item {sid!r} carries no 'cohort' key and no "
                "default_cohort was given — per-cohort OOF metrics need one or the other."
            )
        row = {
            "subject_id": sid,
            "fold": int(oof_folds[i]),
            "cohort": str(cohort),
            "label": int(oof_targets[i]),
            "prob": float(oof_probs[i]),
            "n_scans": int(it["n_scans"]),
            "age": float(it["age"]) if "age" in it else float("nan"),
            "sex": it.get("sex"),
        }
        for name, arr in extras.items():
            row[name] = float(arr[i])
        rows.append(row)

    columns = _FRAME_COLUMNS + sorted(extras)
    return pd.DataFrame(rows, columns=columns)


def _spearman(sub: pd.DataFrame) -> Dict[str, float]:
    """r/p of ``prob`` vs ``n_scans``; ``NaN`` when the input is degenerate."""
    if len(sub) < 2 or sub["n_scans"].nunique() < 2 or sub["prob"].nunique() < 2:
        return {"r": float("nan"), "p": float("nan"), "n": int(len(sub))}
    r, p = spearmanr(sub["n_scans"], sub["prob"])
    return {"r": float(r), "p": float(p), "n": int(len(sub))}


def oof_metrics(frame: pd.DataFrame, *, threshold: float) -> Dict[str, Any]:
    """Tier 1-3 metrics computed purely from ``frame`` — never test/external.

    ``threshold`` is the already-OOF-derived active threshold
    (``common.thresholds.select_oof_threshold``), reused here only to
    binarize predictions for balanced accuracy — it is not re-fit on this
    frame (see ``.claude/rules/evaluation.md``).

    Returns (see ``DOCS/temporal-first-ablation.md``'s 2026-08-24 addendum):

    * ``oof_auc`` / ``oof_pr_auc`` / ``oof_balanced_accuracy`` — pooled OOF.
    * ``oof_auc_<cohort>`` — one column per distinct ``cohort`` value present
      (the Tier-3 per-cohort-collapse veto).
    * ``oof_static_n1_auc`` — only when the frame carries a ``prob_n1`` column
      (the Tier-1 static-baseline floor; absent, not NaN, if no fold_probe
      supplied it — an arm run without the probe simply has no such column).
    * ``oof_prob_nscans_spearman_{overall,converter,non_converter}`` — the
      Tier-3 scan-count-shortcut veto's OOF-side statistic (paired with
      ``common.visit_confound.within_subject_prob_slopes`` on a kept arm's
      saved checkpoint for the mechanism, per the addendum's veto table).
    """
    if frame.empty:
        raise ValueError("oof_metrics: empty OOF frame.")

    y = frame["label"].to_numpy(dtype=int)
    p = frame["prob"].to_numpy(dtype=float)
    pred = (p >= threshold).astype(int)
    both_classes = len(np.unique(y)) > 1

    out: Dict[str, Any] = {
        "oof_n": int(len(frame)),
        "oof_auc": float(roc_auc_score(y, p)) if both_classes else float("nan"),
        "oof_pr_auc": float(average_precision_score(y, p)) if both_classes else float("nan"),
        "oof_balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "oof_threshold": float(threshold),
    }

    for cohort, sub in frame.groupby("cohort"):
        ys = sub["label"].to_numpy(dtype=int)
        ps = sub["prob"].to_numpy(dtype=float)
        out[f"oof_auc_{cohort}"] = (
            float(roc_auc_score(ys, ps)) if len(ys) > 1 and len(np.unique(ys)) > 1 else float("nan")
        )

    if "prob_n1" in frame.columns:
        pn = frame["prob_n1"].to_numpy(dtype=float)
        out["oof_static_n1_auc"] = (
            float(roc_auc_score(y, pn)) if both_classes else float("nan")
        )

    out["oof_prob_nscans_spearman_overall"] = _spearman(frame)["r"]
    out["oof_prob_nscans_spearman_converter"] = _spearman(frame[frame["label"] == 1])["r"]
    out["oof_prob_nscans_spearman_non_converter"] = _spearman(frame[frame["label"] == 0])["r"]

    return out


__all__ = ["build_oof_frame", "oof_metrics"]
