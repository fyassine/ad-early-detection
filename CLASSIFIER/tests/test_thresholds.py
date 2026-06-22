"""Tests for CLASSIFIER.common.thresholds."""
from __future__ import annotations

import numpy as np
import pytest

from CLASSIFIER.common.thresholds import (
    best_f1_threshold,
    oof_threshold_metrics,
    select_oof_threshold,
    youden_threshold,
)

# A perfectly separable set: positives score high, negatives score low.
TARGETS = np.array([0, 0, 0, 1, 1, 1])
PROBS = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])


def test_youden_threshold_separates_classes():
    thr = youden_threshold(TARGETS, PROBS)
    pred = (PROBS >= thr).astype(int)
    assert (pred == TARGETS).all()


def test_best_f1_threshold_separates_classes():
    thr = best_f1_threshold(TARGETS, PROBS)
    pred = (PROBS >= thr).astype(int)
    assert (pred == TARGETS).all()


def test_thresholds_single_class_default():
    assert youden_threshold(np.zeros(5), np.linspace(0, 1, 5)) == 0.5
    assert best_f1_threshold(np.ones(5), np.linspace(0, 1, 5)) == 0.5


def test_oof_threshold_metrics_perfect():
    thr = youden_threshold(TARGETS, PROBS)
    sens, spec, f1 = oof_threshold_metrics(TARGETS, PROBS, thr)
    assert sens == 1.0 and spec == 1.0 and f1 == 1.0


def test_select_default_is_best_f1():
    thr, method = select_oof_threshold(TARGETS, PROBS, threshold_mode="best-f1")
    assert method == "oof_f1"
    assert thr == best_f1_threshold(TARGETS, PROBS)


def test_select_youden_mode():
    thr, method = select_oof_threshold(TARGETS, PROBS, threshold_mode="youden")
    assert method == "oof_youden"
    assert thr == youden_threshold(TARGETS, PROBS)


def test_select_fixed_requires_value():
    with pytest.raises(ValueError):
        select_oof_threshold(TARGETS, PROBS, threshold_mode="fixed")
    thr, method = select_oof_threshold(
        TARGETS, PROBS, threshold_mode="fixed", fixed_threshold=0.42
    )
    assert method == "fixed" and thr == 0.42


def test_select_runner_active_requires_mode():
    with pytest.raises(ValueError):
        select_oof_threshold(TARGETS, PROBS, threshold_mode=None, runner_active=True)


def test_select_non_interactive_falls_back_to_f1():
    thr, method = select_oof_threshold(
        TARGETS, PROBS, threshold_mode=None, runner_active=False, interactive=False
    )
    assert method == "oof_f1"


def test_select_unknown_mode_raises():
    with pytest.raises(ValueError):
        select_oof_threshold(TARGETS, PROBS, threshold_mode="bogus")
