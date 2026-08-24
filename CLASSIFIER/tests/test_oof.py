"""Tests for CLASSIFIER.common.oof."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from CLASSIFIER.common.crossval import Bundle
from CLASSIFIER.common.oof import build_oof_frame, oof_metrics


def _bundle(cohorts=None):
    n = 8
    labels = [i % 2 for i in range(n)]
    groups = [f"sub{i}" for i in range(n)]
    items = []
    for i, (g, lab) in enumerate(zip(groups, labels, strict=False)):
        it = {"subject_id": g, "label": lab, "n_scans": 1 + (i % 3), "age": 60.0 + i, "sex": i % 2}
        if cohorts is not None:
            it["cohort"] = cohorts[i]
        items.append(it)
    return Bundle(labels, groups, items)


def _oof_arrays(bundle, seed=0):
    rng = np.random.default_rng(seed)
    n = len(bundle)
    probs = rng.uniform(0, 1, size=n)
    targets = np.array(bundle.labels)
    folds = np.array([1 + (i % 5) for i in range(n)])
    return bundle.groups, probs, targets, folds


def test_build_oof_frame_schema_and_values():
    bundle = _bundle()
    sids, probs, targets, folds = _oof_arrays(bundle)
    frame = build_oof_frame(bundle, sids, probs, targets, folds, default_cohort="delcode")
    assert list(frame.columns) == ["subject_id", "fold", "cohort", "label", "prob", "n_scans", "age", "sex"]
    assert len(frame) == len(bundle)
    assert set(frame["cohort"]) == {"delcode"}
    assert frame["n_scans"].tolist() == [it["n_scans"] for it in bundle.items]


def test_build_oof_frame_uses_item_cohort_when_present():
    cohorts = ["adni", "delcode"] * 4
    bundle = _bundle(cohorts=cohorts)
    sids, probs, targets, folds = _oof_arrays(bundle)
    frame = build_oof_frame(bundle, sids, probs, targets, folds)
    assert set(frame["cohort"]) == {"adni", "delcode"}


def test_build_oof_frame_raises_without_cohort_or_default():
    bundle = _bundle()  # no 'cohort' key on any item
    sids, probs, targets, folds = _oof_arrays(bundle)
    with pytest.raises(ValueError, match="cohort"):
        build_oof_frame(bundle, sids, probs, targets, folds)


def test_build_oof_frame_length_mismatch_raises():
    bundle = _bundle()
    sids, probs, targets, folds = _oof_arrays(bundle)
    with pytest.raises(ValueError, match="length"):
        build_oof_frame(bundle, sids, probs[:-1], targets, folds, default_cohort="delcode")


def test_build_oof_frame_unknown_subject_raises():
    bundle = _bundle()
    sids, probs, targets, folds = _oof_arrays(bundle)
    sids = list(sids)
    sids[0] = "not-a-real-subject"
    with pytest.raises(ValueError, match="not found"):
        build_oof_frame(bundle, sids, probs, targets, folds, default_cohort="delcode")


def test_build_oof_frame_carries_extras():
    bundle = _bundle()
    sids, probs, targets, folds = _oof_arrays(bundle)
    extras = {"prob_n1": np.full(len(bundle), 0.42)}
    frame = build_oof_frame(bundle, sids, probs, targets, folds, extras, default_cohort="delcode")
    assert "prob_n1" in frame.columns
    assert np.all(frame["prob_n1"] == 0.42)


def test_oof_metrics_basic_and_per_cohort():
    cohorts = ["adni"] * 4 + ["delcode"] * 4
    bundle = _bundle(cohorts=cohorts)
    sids, probs, targets, folds = _oof_arrays(bundle, seed=1)
    frame = build_oof_frame(bundle, sids, probs, targets, folds)
    metrics = oof_metrics(frame, threshold=0.5)

    assert metrics["oof_n"] == len(bundle)
    assert metrics["oof_auc"] == pytest.approx(roc_auc_score(targets, probs))
    assert 0.0 <= metrics["oof_pr_auc"] <= 1.0
    assert 0.0 <= metrics["oof_balanced_accuracy"] <= 1.0
    assert "oof_auc_adni" in metrics and "oof_auc_delcode" in metrics
    assert "oof_static_n1_auc" not in metrics  # no prob_n1 column supplied


def test_oof_metrics_static_n1_auc_present_when_extras_given():
    bundle = _bundle()
    sids, probs, targets, folds = _oof_arrays(bundle, seed=2)
    extras = {"prob_n1": np.random.default_rng(3).uniform(0, 1, size=len(bundle))}
    frame = build_oof_frame(bundle, sids, probs, targets, folds, extras, default_cohort="delcode")
    metrics = oof_metrics(frame, threshold=0.5)
    assert "oof_static_n1_auc" in metrics


def test_oof_metrics_spearman_keys_present():
    bundle = _bundle()
    sids, probs, targets, folds = _oof_arrays(bundle, seed=4)
    frame = build_oof_frame(bundle, sids, probs, targets, folds, default_cohort="delcode")
    metrics = oof_metrics(frame, threshold=0.5)
    for key in (
        "oof_prob_nscans_spearman_overall",
        "oof_prob_nscans_spearman_converter",
        "oof_prob_nscans_spearman_non_converter",
    ):
        assert key in metrics


def test_oof_metrics_empty_frame_raises():
    bundle = _bundle()
    sids, probs, targets, folds = _oof_arrays(bundle)
    frame = build_oof_frame(bundle, sids, probs, targets, folds, default_cohort="delcode")
    with pytest.raises(ValueError, match="empty"):
        oof_metrics(frame.iloc[0:0], threshold=0.5)
