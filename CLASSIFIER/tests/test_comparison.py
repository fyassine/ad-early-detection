"""
Tests for CLASSIFIER.common.comparison — paired DeLong, bootstrap CI, Holm,
McNemar. Validates correctness on synthetic data with known answers and
cross-checks against scipy / sklearn where possible.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from CLASSIFIER.common.comparison import (
    holm_correction,
    mcnemar_paired,
    paired_bootstrap_ci,
    paired_delong_test,
)


def _make_paired_data(n=200, auc_a=0.85, auc_b=0.75, seed=0):
    """Generate paired probabilities with target AUCs on the same labels."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=n)
    # Construct scores so positives are shifted relative to negatives by
    # different magnitudes for A vs B → different AUCs.
    base = rng.normal(size=n)
    probs_a = base + labels * (2.0 * auc_a - 1.0) * 2.0
    probs_b = base + labels * (2.0 * auc_b - 1.0) * 2.0
    return probs_a, probs_b, labels


def test_delong_auc_matches_sklearn():
    probs_a, probs_b, labels = _make_paired_data(seed=1)
    delta, _, _, _ = paired_delong_test(probs_a, probs_b, labels)
    expected = roc_auc_score(labels, probs_a) - roc_auc_score(labels, probs_b)
    assert delta == pytest.approx(expected, abs=1e-9)


def test_delong_identical_predictions_gives_p_one():
    rng = np.random.default_rng(2)
    labels = rng.integers(0, 2, size=100)
    probs = rng.normal(size=100)
    delta, lo, hi, p = paired_delong_test(probs, probs, labels)
    assert delta == 0.0
    assert lo == 0.0 and hi == 0.0
    assert p == 1.0


def test_delong_significant_when_aucs_differ():
    probs_a, probs_b, labels = _make_paired_data(n=500, auc_a=0.9, auc_b=0.6, seed=3)
    _, _, _, p = paired_delong_test(probs_a, probs_b, labels)
    assert p < 0.001


def test_delong_ci_contains_zero_when_equal_in_distribution():
    rng = np.random.default_rng(4)
    n = 400
    labels = rng.integers(0, 2, size=n)
    probs_a = rng.normal(loc=labels.astype(float), scale=1.0)
    probs_b = rng.normal(loc=labels.astype(float), scale=1.0)
    _, lo, hi, p = paired_delong_test(probs_a, probs_b, labels)
    assert lo < 0 < hi
    assert p > 0.05


def test_delong_rejects_degenerate_label_counts():
    labels = np.array([1, 1, 1, 1, 0])  # only 1 negative
    probs = np.array([0.9, 0.8, 0.7, 0.6, 0.1])
    with pytest.raises(ValueError, match="at least 2 positives and 2 negatives"):
        paired_delong_test(probs, probs, labels)


def test_delong_rejects_shape_mismatch():
    probs_a = np.array([0.1, 0.2, 0.3, 0.4])
    probs_b = np.array([0.1, 0.2, 0.3])
    labels = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="identical shape"):
        paired_delong_test(probs_a, probs_b, labels)


def test_bootstrap_ci_recovers_known_delta_auc():
    probs_a, probs_b, labels = _make_paired_data(n=400, auc_a=0.85, auc_b=0.65, seed=5)
    rng = np.random.default_rng(99)
    point, lo, hi = paired_bootstrap_ci(
        probs_a, probs_b, labels, roc_auc_score, n_boot=500, rng=rng,
    )
    truth = roc_auc_score(labels, probs_a) - roc_auc_score(labels, probs_b)
    assert point == pytest.approx(truth, abs=1e-9)
    assert lo < truth < hi
    assert hi - lo < 0.2  # CI should be reasonably tight at n=400


def test_bootstrap_ci_deterministic_with_same_rng():
    probs_a, probs_b, labels = _make_paired_data(seed=6)
    r1 = paired_bootstrap_ci(probs_a, probs_b, labels, roc_auc_score,
                              n_boot=200, rng=np.random.default_rng(7))
    r2 = paired_bootstrap_ci(probs_a, probs_b, labels, roc_auc_score,
                              n_boot=200, rng=np.random.default_rng(7))
    assert r1 == r2


def test_bootstrap_rejects_too_few_boot():
    probs_a, probs_b, labels = _make_paired_data(seed=8)
    with pytest.raises(ValueError, match="n_boot must be"):
        paired_bootstrap_ci(probs_a, probs_b, labels, roc_auc_score,
                            n_boot=50, rng=np.random.default_rng(0))


def test_holm_known_example():
    # Classic textbook example
    p = np.array([0.01, 0.04, 0.03, 0.005])
    rejected, adj = holm_correction(p, alpha=0.05)
    # Sorted: [0.005, 0.01, 0.03, 0.04] with multipliers [4, 3, 2, 1]
    # raw_adj sorted: [0.02, 0.03, 0.06, 0.04] -> cum-max -> [0.02, 0.03, 0.06, 0.06]
    expected_sorted = np.array([0.02, 0.03, 0.06, 0.06])
    order = np.argsort(p)
    np.testing.assert_allclose(adj[order], expected_sorted, atol=1e-12)
    # At alpha=0.05: first two rejected, last two not
    assert rejected[3]  # p=0.005, adj=0.02
    assert rejected[0]  # p=0.01,  adj=0.03
    assert not rejected[2]  # p=0.03, adj=0.06
    assert not rejected[1]  # p=0.04, adj=0.06


def test_holm_all_rejected_when_all_small():
    p = np.array([1e-6, 2e-6, 3e-6, 4e-6])
    rejected, _ = holm_correction(p, alpha=0.05)
    assert rejected.all()


def test_holm_none_rejected_when_all_large():
    p = np.array([0.5, 0.6, 0.7, 0.8])
    rejected, _ = holm_correction(p, alpha=0.05)
    assert not rejected.any()


def test_holm_rejects_out_of_range_p():
    with pytest.raises(ValueError, match="pvalues must be"):
        holm_correction(np.array([0.5, 1.5]))


def test_mcnemar_identical_predictions_returns_one():
    preds = np.array([1, 0, 1, 0, 1])
    labels = np.array([1, 0, 1, 0, 0])
    assert mcnemar_paired(preds, preds, labels) == 1.0


def test_mcnemar_detects_systematic_disagreement():
    # A is always right, B is always wrong → maximum discordance
    rng = np.random.default_rng(10)
    n = 80
    labels = rng.integers(0, 2, size=n)
    preds_a = labels.copy()
    preds_b = 1 - labels
    p = mcnemar_paired(preds_a, preds_b, labels)
    assert p < 0.001


def test_mcnemar_shape_mismatch():
    preds_a = np.array([1, 0])
    preds_b = np.array([1, 0, 1])
    labels = np.array([1, 0])
    with pytest.raises(ValueError, match="identical shape"):
        mcnemar_paired(preds_a, preds_b, labels)
