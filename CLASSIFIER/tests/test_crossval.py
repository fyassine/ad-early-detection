"""Tests for CLASSIFIER.common.crossval."""
from __future__ import annotations

import numpy as np

from CLASSIFIER.common.crossval import Bundle, CVResult, run_kfold_cv, summarize_cv


def _make_bundle(n=40):
    # Two subjects' worth of items per group so StratifiedGroupKFold can split,
    # balanced classes, deterministic.
    labels = [i % 2 for i in range(n)]
    groups = [f"sub{i}" for i in range(n)]  # one item per subject (subject-level)
    items = [{"subject_id": g, "label": l, "n_scans": 1} for g, l in zip(groups, labels)]
    return Bundle(labels, groups, items)


def _fold_stub(auc_by_call):
    """Return a train_fold that yields preset AUCs in call order."""
    state = {"calls": 0}

    def train_fold(bundle_tr, bundle_va, cfg, *, rng, device):
        i = state["calls"]
        state["calls"] += 1
        auc = auc_by_call[i]
        n = len(bundle_va.items)
        # canned OOF: probs equal to the fold's auc, targets from the val labels
        return {
            "state_dict": {"fold_auc": auc},
            "val_metrics": {"auc": auc, "sensitivity": auc, "specificity": auc, "f1": auc},
            "best_threshold": 0.3 + 0.1 * i,
            "oof_probs": np.full(n, auc),
            "oof_targets": np.array(bundle_va.labels),
            "oof_sids": list(bundle_va.groups),
        }

    return train_fold


def test_run_kfold_cv_selects_best_fold_and_concatenates_oof():
    bundle = _make_bundle(40)
    aucs = [0.6, 0.9, 0.7, 0.5, 0.8]  # fold 2 (index 1) is best
    res = run_kfold_cv(
        bundle, _fold_stub(aucs), cfg={}, n_folds=5, rng=None, device="cpu"
    )

    assert isinstance(res, CVResult)
    assert res.best_fold == 2
    assert res.best_val_auc == 0.9
    assert res.best_model_state == {"fold_auc": 0.9}
    # best_threshold is the winning fold's threshold (0.3 + 0.1*1)
    assert abs(res.best_threshold - 0.4) < 1e-9
    # OOF arrays cover every subject exactly once
    assert len(res.oof_probs) == len(bundle)
    assert len(res.oof_targets) == len(bundle)
    assert len(res.oof_sids) == len(bundle)
    assert set(res.oof_sids) == set(bundle.groups)
    # cv_results has one entry per fold
    assert res.cv_results["val_auc"] == aucs
    assert res.cv_results["fold"] == [1, 2, 3, 4, 5]


def test_run_kfold_cv_log_fn_called_per_fold():
    bundle = _make_bundle(40)
    logged = []
    run_kfold_cv(
        bundle, _fold_stub([0.5] * 5), cfg={}, n_folds=5,
        rng=None, device="cpu", log_fn=logged.append,
    )
    assert len(logged) == 5
    assert all("val_auc" in d and "fold" in d for d in logged)


def test_best_f1_threshold_computed_from_oof():
    bundle = _make_bundle(40)
    res = run_kfold_cv(bundle, _fold_stub([0.7] * 5), cfg={}, n_folds=5, rng=None, device="cpu")
    assert isinstance(res.best_f1_threshold, float)


def test_bundle_subset_reindexes():
    b = _make_bundle(10)
    sub = b.subset([0, 2, 4])
    assert len(sub) == 3
    assert sub.groups == ["sub0", "sub2", "sub4"]
    assert sub.labels == [0, 0, 0]


def test_summarize_cv_smoke(capsys):
    cv = {
        "val_auc": [0.7, 0.8],
        "val_sensitivity": [0.6, 0.7],
        "val_specificity": [0.5, 0.6],
        "val_f1": [0.65, 0.75],
    }
    summarize_cv(cv)
    out = capsys.readouterr().out
    assert "val_auc" in out and "Mean" in out
