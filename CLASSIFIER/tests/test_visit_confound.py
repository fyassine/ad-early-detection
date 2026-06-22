"""Tests for CLASSIFIER.common.visit_confound — the visit-count confound diagnostics.

Pure, model-free: a synthetic ``Bundle`` plus fake adapter hooks
(``truncate_to_n_visits``, ``eval_split``, ``per_visit_probs``) exercise the
diagnostic logic without touching torch, a GAAE checkpoint, or the DELCODE matrices.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from CLASSIFIER.common.crossval import Bundle
from CLASSIFIER.common.visit_confound import (
    cohort_composition_table,
    early_detection_fixed_cohort,
    prob_spread_summary,
    prob_vs_visit_count,
    summarize_visit_counts,
    visit_counts_by_label,
    within_subject_prob_slopes,
)


def _bundle(specs):
    """specs: list of (subject_id, label, n_scans)."""
    items = [
        {"subject_id": sid, "label": lab, "n_scans": ns,
         "visit_months": [12 * t for t in range(ns)]}
        for sid, lab, ns in specs
    ]
    return Bundle([it["label"] for it in items], [it["subject_id"] for it in items], items)


# Fake hooks ------------------------------------------------------------------
def _truncate(bundle, n):
    items = [{**it, "n_scans": n, "visit_months": it["visit_months"][:n]}
             for it in bundle.items if it["n_scans"] >= n]
    return Bundle([it["label"] for it in items], [it["subject_id"] for it in items], items)


def _eval_split(state, bundle, threshold, *, device):
    # Deterministic separable probs so AUC is well-defined when both classes present.
    labels = np.asarray(bundle.labels, dtype=int)
    probs = np.where(labels == 1, 0.8, 0.2)
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
    pred = (probs >= threshold).astype(int)
    sens = (pred[labels == 1] == 1).mean() if (labels == 1).any() else 0.0
    spec = (pred[labels == 0] == 0).mean() if (labels == 0).any() else 0.0
    return {"auc": float(auc), "sensitivity": float(sens), "specificity": float(spec)}


def _per_visit_probs(state, item, *, device):
    # Final prob grows with n_scans -> a positive prob/n_scans Spearman by construction.
    return [(item["visit_months"][t - 1], 0.1 * item["n_scans"] + 0.01 * t)
            for t in range(1, item["n_scans"] + 1)]


# Tests -----------------------------------------------------------------------
def test_visit_counts_by_label_shape():
    b = _bundle([("a", 1, 2), ("b", 0, 4)])
    df = visit_counts_by_label(b)
    assert list(df.columns) == ["subject_id", "label", "group", "n_scans"]
    assert df.set_index("subject_id").loc["a", "group"] == "converter"
    assert df.set_index("subject_id").loc["b", "group"] == "non_converter"


def test_summarize_visit_counts_gap_and_pvalue():
    # converters have fewer visits than non-converters
    b = _bundle([("c1", 1, 1), ("c2", 1, 2), ("n1", 0, 4), ("n2", 0, 5), ("n3", 0, 6)])
    s = summarize_visit_counts(b).set_index("group")
    assert s.loc["converter", "mean"] < s.loc["non_converter", "mean"]
    assert s.loc["converter", "n"] == 2 and s.loc["non_converter", "n"] == 3
    assert np.isfinite(s.loc["converter", "mwu_pvalue"])


def test_summarize_visit_counts_single_group_pvalue_nan():
    b = _bundle([("c1", 1, 1), ("c2", 1, 3)])  # converters only
    s = summarize_visit_counts(b)
    assert np.isnan(s["mwu_pvalue"].iloc[0])


def test_cohort_composition_shrinks_with_n():
    b = _bundle([("a", 1, 1), ("b", 1, 2), ("c", 0, 3), ("d", 0, 4)])
    rows = cohort_composition_table(b, _truncate)
    by_n = {r["n_visits"]: r for r in rows}
    assert by_n[1]["n_subjects"] == 4
    assert by_n[2]["n_subjects"] == 3   # 'a' (1 visit) dropped
    assert by_n[3]["n_subjects"] == 2
    assert by_n[4]["n_subjects"] == 1
    assert by_n[1]["n_converters"] == 2 and by_n[1]["n_nonconverters"] == 2
    assert abs(by_n[1]["frac_converter"] - 0.5) < 1e-9


def test_early_detection_fixed_cohort_holds_cohort_constant():
    # Default min_n_scans = global max (4) -> fixed cohort = the deepest-followed
    # subjects (a, b); c, d (3 visits) are excluded, so the cohort never shrinks.
    b = _bundle([("a", 1, 4), ("b", 0, 4), ("c", 1, 3), ("d", 0, 3)])
    rows = early_detection_fixed_cohort(b, _eval_split, _truncate, state_dict=None,
                                        threshold=0.5, device="cpu", min_subjects=2)
    assert {r["n_subjects"] for r in rows} == {2}        # never shrinks
    assert {r["n_visits"] for r in rows} == {1, 2, 3, 4}  # all N on the same 2 subjects


def test_early_detection_fixed_cohort_skips_singleclass_tail():
    # e, f are a converter-only deep tail (5 visits). Anchoring to the global max (5)
    # would give a single-class cohort -> empty; the deepest *viable* depth is 3.
    b = _bundle([("a", 1, 3), ("b", 0, 3), ("c", 1, 2), ("d", 0, 2),
                 ("e", 1, 5), ("f", 1, 5)])
    rows = early_detection_fixed_cohort(b, _eval_split, _truncate, state_dict=None,
                                        threshold=0.5, device="cpu", min_subjects=2)
    assert rows, "fixed cohort should anchor to the deepest two-class depth, not be empty"
    assert {r["n_subjects"] for r in rows} == {4}        # subjects with >= 3 visits
    assert {r["n_visits"] for r in rows} == {1, 2, 3}     # N=4,5 are converter-only -> skipped


def test_prob_vs_visit_count_positive_spearman():
    b = _bundle([("a", 1, 1), ("b", 1, 3), ("c", 0, 2), ("d", 0, 5)])
    df, stats = prob_vs_visit_count(b, _per_visit_probs, state_dict=None, device="cpu")
    assert set(df.columns) == {"subject_id", "label", "group", "n_scans", "prob"}
    assert stats["overall"]["r"] > 0.9       # prob built to rise with n_scans
    assert stats["overall"]["n"] == 4


def _per_visit_probs_decreasing(state, item, *, device):
    # Within each subject P(converter) decreases as visits accumulate (slope < 0).
    return [(item["visit_months"][t - 1], 0.9 - 0.1 * (t - 1))
            for t in range(1, item["n_scans"] + 1)]


def test_within_subject_prob_slopes_negative():
    b = _bundle([("a", 0, 4), ("b", 0, 3), ("c", 1, 2)])
    df, stats = within_subject_prob_slopes(b, _per_visit_probs_decreasing,
                                           state_dict=None, device="cpu")
    assert set(df.columns) == {"subject_id", "label", "group", "n_scans", "slope"}
    assert (df["slope"] < 0).all()
    assert stats["non_converter"]["frac_negative"] == 1.0
    assert stats["non_converter"]["median_slope"] < 0
    assert stats["non_converter"]["n"] == 2


def test_within_subject_prob_slopes_skips_single_visit():
    b = _bundle([("a", 0, 1), ("b", 1, 3)])  # 'a' has 1 visit -> excluded
    df, _ = within_subject_prob_slopes(b, _per_visit_probs_decreasing,
                                       state_dict=None, device="cpu")
    assert df["subject_id"].tolist() == ["b"]


def test_prob_spread_summary_separation():
    import pandas as pd
    df = pd.DataFrame({
        "label": [1, 1, 0, 0],
        "prob": [0.8, 0.9, 0.2, 0.1],
    })
    s = prob_spread_summary(df)
    assert abs(s["separation"] - (0.85 - 0.15)) < 1e-9
    assert s["converter"]["n"] == 2 and s["non_converter"]["n"] == 2
