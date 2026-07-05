"""
common/calibration.py — post-hoc probability calibration (temperature scaling).

A trained classifier can rank well (high AUC) yet emit probabilities clustered near
the threshold — e.g. the GELSTM trajectory model, whose RNN bottleneck + shallow,
un-normalised head keep logits small. Temperature scaling rescales the logits by a
single learned scalar T to spread the probabilities so they reflect confidence:

    p_cal = sigmoid( logit(p) / T )

T is fit by minimising binary cross-entropy on a held-out set (the cross-validation
out-of-fold predictions — never the test set). T < 1 sharpens an under-confident
model (widens the spread); T > 1 softens an over-confident one; T = 1 is the identity.

Because the map is **monotonic**, temperature scaling does NOT change AUC or the
sens/spec achievable at the (re-mapped) best threshold — it improves the probability
*spread* and calibration (ECE), not discrimination.

Model-agnostic and deterministic (no RNG). Works on probabilities directly: callers
pass the sigmoid outputs they already have; logits are recovered as log(p/(1-p)).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.optimize import minimize_scalar

_EPS = 1e-6


def _to_logits(probs: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probs, dtype=float), _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Rescale probabilities by ``temperature`` (monotonic): ``sigmoid(logit(p)/T)``."""
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}.")
    return _sigmoid(_to_logits(probs) / float(temperature))


def fit_temperature(
    probs: np.ndarray,
    targets: np.ndarray,
    *,
    bounds: Tuple[float, float] = (0.05, 20.0),
) -> float:
    """Temperature minimising BCE NLL of ``apply_temperature(probs, T)`` vs ``targets``.

    Fit on a held-out split (e.g. OOF predictions), never on the test set. Returns the
    scalar ``T``; ``T < 1`` widens an under-confident model's spread.
    """
    probs = np.asarray(probs, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if probs.shape != targets.shape:
        raise ValueError(f"probs {probs.shape} and targets {targets.shape} must match.")
    if probs.size == 0 or len(np.unique(targets)) < 2:
        # Nothing to calibrate against — identity transform.
        return 1.0
    logits = _to_logits(probs)

    def nll(t: float) -> float:
        p = np.clip(_sigmoid(logits / t), _EPS, 1.0 - _EPS)
        return float(-np.mean(targets * np.log(p) + (1.0 - targets) * np.log(1.0 - p)))

    res = minimize_scalar(nll, bounds=bounds, method="bounded")
    return float(res.x)


def expected_calibration_error(
    probs: np.ndarray,
    targets: np.ndarray,
    *,
    n_bins: int = 15,
) -> float:
    """Equal-width-bin ECE: sum_b (n_b/N) * |accuracy_b - confidence_b|."""
    probs = np.asarray(probs, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if probs.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize: bin index in 1..n_bins; clip the p==1.0 edge into the last bin.
    idx = np.clip(np.digitize(probs, edges[1:-1], right=False), 0, n_bins - 1)
    ece = 0.0
    n = probs.size
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        conf = float(np.mean(probs[mask]))
        acc = float(np.mean(targets[mask]))
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


__all__ = ["fit_temperature", "apply_temperature", "expected_calibration_error"]
