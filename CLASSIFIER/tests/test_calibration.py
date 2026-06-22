"""Tests for CLASSIFIER.common.calibration — temperature scaling + ECE."""
from __future__ import annotations

import numpy as np

from CLASSIFIER.common.calibration import (
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
)


def _underconfident(n=400, seed=0):
    """Well-ranked but squished-toward-0.5 probabilities (T<1 should widen them)."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n).astype(float)
    # separable logits, then shrunk so sigmoid clusters near 0.5
    logits = np.where(y == 1, 1.0, -1.0) + rng.normal(0, 0.5, size=n)
    probs = 1.0 / (1.0 + np.exp(-(logits * 0.2)))  # *0.2 => under-confident
    return probs, y


def test_apply_temperature_identity():
    probs = np.array([0.1, 0.4, 0.6, 0.9])
    out = apply_temperature(probs, 1.0)
    assert np.allclose(out, probs, atol=1e-4)


def test_apply_temperature_monotonic_preserves_auc():
    from sklearn.metrics import roc_auc_score
    probs, y = _underconfident()
    cal = apply_temperature(probs, 0.5)
    # monotonic transform -> identical ranking -> identical AUC
    assert abs(roc_auc_score(y, probs) - roc_auc_score(y, cal)) < 1e-9


def test_fit_temperature_widens_underconfident():
    probs, y = _underconfident()
    T = fit_temperature(probs, y)
    assert T < 1.0  # sharpen an under-confident model
    cal = apply_temperature(probs, T)
    assert cal.std() > probs.std()  # spread widens


def test_fit_temperature_improves_ece():
    probs, y = _underconfident()
    T = fit_temperature(probs, y)
    cal = apply_temperature(probs, T)
    ece_raw = expected_calibration_error(probs, y)
    ece_cal = expected_calibration_error(cal, y)
    assert ece_cal <= ece_raw + 1e-9


def test_fit_temperature_single_class_is_identity():
    assert fit_temperature(np.array([0.6, 0.7, 0.8]), np.array([1, 1, 1])) == 1.0


def test_expected_calibration_error_perfect_is_zero():
    # confidence == accuracy in every occupied bin -> ECE 0
    probs = np.array([0.0, 0.0, 1.0, 1.0])
    targets = np.array([0, 0, 1, 1])
    assert expected_calibration_error(probs, targets) < 1e-9
