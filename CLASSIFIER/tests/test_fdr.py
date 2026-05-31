"""
Tests for CLASSIFIER.common.fdr — Fisher's Discriminant Ratio dimension
selection. Validates that the helper picks known-separable dimensions, respects
the top_k bound, and never reads beyond the rows passed in (leakage guard).
"""
from __future__ import annotations

import numpy as np
import pytest

from CLASSIFIER.common.fdr import compute_fdr_filter, compute_fdr_scores


def _make_embeddings(seed=0):
    """
    Build (N, 5) embeddings where dims 1 and 3 are class-separable and dims
    0, 2, 4 are pure noise with identical class distributions.
    """
    rng = np.random.default_rng(seed)
    n = 200
    labels = np.array([0] * (n // 2) + [1] * (n // 2))
    embs = rng.normal(0, 1, size=(n, 5))
    # Inject separation only on dims 1 (strong) and 3 (moderate).
    embs[labels == 1, 1] += 5.0
    embs[labels == 1, 3] += 2.0
    return embs, labels


def test_picks_separable_dims():
    embs, labels = _make_embeddings()
    top_dims, scores = compute_fdr_filter(embs, labels, top_k=2)
    assert set(top_dims.tolist()) == {1, 3}
    # Strongest separation (dim 1) ranks first.
    assert top_dims[0] == 1
    assert scores[1] > scores[3] > scores[0]


def test_top_k_bound():
    embs, labels = _make_embeddings()
    with pytest.raises(ValueError):
        compute_fdr_filter(embs, labels, top_k=0)
    with pytest.raises(ValueError):
        compute_fdr_filter(embs, labels, top_k=6)  # only 5 dims
    # Full selection is allowed.
    top_dims, _ = compute_fdr_filter(embs, labels, top_k=5)
    assert sorted(top_dims.tolist()) == [0, 1, 2, 3, 4]


def test_returns_contiguous_positive_stride():
    embs, labels = _make_embeddings()
    top_dims, _ = compute_fdr_filter(embs, labels, top_k=3)
    # torch rejects negative-stride arrays; ensure the descending sort was copied.
    assert top_dims.flags["C_CONTIGUOUS"]
    assert all(s > 0 for s in top_dims.strides)


def test_requires_both_classes():
    embs, labels = _make_embeddings()
    labels[:] = 1  # single class
    with pytest.raises(ValueError):
        compute_fdr_scores(embs, labels)


def test_leakage_guard_only_sees_passed_rows():
    """
    Selection on a training slice must not change when unrelated rows are
    appended elsewhere — i.e. the function only uses the rows it is given.
    """
    embs, labels = _make_embeddings()
    train_idx = np.arange(0, 150)
    top_train, _ = compute_fdr_filter(embs[train_idx], labels[train_idx], top_k=2)

    # Corrupt the held-out rows wildly; recompute on the SAME train slice.
    embs_corrupted = embs.copy()
    embs_corrupted[150:] += 1000.0
    top_train_again, _ = compute_fdr_filter(
        embs_corrupted[train_idx], labels[train_idx], top_k=2
    )
    assert top_train.tolist() == top_train_again.tolist()


def test_label_length_mismatch():
    embs, labels = _make_embeddings()
    with pytest.raises(ValueError):
        compute_fdr_scores(embs, labels[:-1])
