"""
Paired statistical tests for cross-region model comparison.

All functions assume PAIRED predictions: the same subjects scored by two
classifiers (e.g. GEC trained on whole-brain vs DMN). This is the design
guaranteed by DELCODE's shared 463/346/70/47 splits across the 9 region
datasets — every (model x region) cell scores the same 47 test subjects, so
the within-subject correlation must be exploited rather than ignored.
"""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from scipy import stats


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """1-indexed midrank with average rank assigned to ties."""
    J = np.argsort(x, kind="mergesort")
    Z = x[J]
    N = len(x)
    T = np.empty(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    out = np.empty(N, dtype=float)
    out[J] = T
    return out


def _delong_placement(scores: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Placement values V10 (per positive) and V01 (per negative), plus AUC."""
    pos = labels == 1
    neg = labels == 0
    X = scores[pos]
    Y = scores[neg]
    m = len(X)
    n = len(Y)
    if m < 2 or n < 2:
        raise ValueError(
            "DeLong's test requires at least 2 positives and 2 negatives "
            f"(got m={m}, n={n}). Use paired_bootstrap_ci for very small samples."
        )

    combined = np.concatenate([X, Y])
    r_combined = _compute_midrank(combined)
    r_X_only = _compute_midrank(X)
    r_Y_only = _compute_midrank(Y)

    V10 = (r_combined[:m] - r_X_only) / n
    V01 = 1.0 - (r_combined[m:] - r_Y_only) / m
    auc = float(V10.mean())
    return V10, V01, auc


def paired_delong_test(
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    labels: np.ndarray,
    *,
    alpha: float = 0.05,
) -> Tuple[float, float, float, float]:
    """
    DeLong's test for the difference between two correlated AUCs on the same subjects.

    Returns (delta_auc, ci_lo, ci_hi, p_value) for two-sided H0: AUC_a == AUC_b.
    """
    probs_a = np.asarray(probs_a, dtype=float)
    probs_b = np.asarray(probs_b, dtype=float)
    labels = np.asarray(labels)
    if probs_a.shape != probs_b.shape or probs_a.shape != labels.shape:
        raise ValueError("probs_a, probs_b, and labels must have identical shape.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1).")

    V10_a, V01_a, auc_a = _delong_placement(probs_a, labels)
    V10_b, V01_b, auc_b = _delong_placement(probs_b, labels)

    m = len(V10_a)
    n = len(V01_a)
    s10 = np.cov(V10_a, V10_b, ddof=1)
    s01 = np.cov(V01_a, V01_b, ddof=1)
    cov = s10 / m + s01 / n
    var_diff = cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1]
    delta = auc_a - auc_b

    if var_diff <= 0:
        return float(delta), float(delta), float(delta), 1.0

    se = float(np.sqrt(var_diff))
    z = delta / se
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    return float(delta), float(delta - z_crit * se), float(delta + z_crit * se), float(p)


def paired_bootstrap_ci(
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    labels: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    """
    Paired bootstrap CI for metric_fn(labels, probs_a) - metric_fn(labels, probs_b).

    `metric_fn` follows the sklearn (y_true, y_score) convention — labels first.
    Subject-level resampling with replacement preserves the pairing (the same
    bootstrap index is applied to A and B). Iterations where either metric_fn
    returns NaN or raises (e.g. single-class bootstrap sample) are dropped.

    Returns (delta_point, ci_lo, ci_hi).
    """
    probs_a = np.asarray(probs_a, dtype=float)
    probs_b = np.asarray(probs_b, dtype=float)
    labels = np.asarray(labels)
    if probs_a.shape != probs_b.shape or probs_a.shape != labels.shape:
        raise ValueError("probs_a, probs_b, and labels must have identical shape.")
    if n_boot < 100:
        raise ValueError("n_boot must be >= 100 for stable percentile CIs.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1).")

    n = len(labels)
    point = float(metric_fn(labels, probs_a) - metric_fn(labels, probs_b))

    deltas = np.empty(n_boot, dtype=float)
    valid = 0
    for _k in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            m_a = metric_fn(labels[idx], probs_a[idx])
            m_b = metric_fn(labels[idx], probs_b[idx])
        except (ValueError, ZeroDivisionError):
            continue
        if np.isnan(m_a) or np.isnan(m_b):
            continue
        deltas[valid] = m_a - m_b
        valid += 1

    if valid < 0.5 * n_boot:
        raise ValueError(
            f"Bootstrap produced too many NaN metrics ({n_boot - valid} of {n_boot}). "
            "Inspect metric_fn or label balance."
        )

    deltas = deltas[:valid]
    return point, float(np.quantile(deltas, alpha / 2.0)), float(np.quantile(deltas, 1.0 - alpha / 2.0))


def holm_correction(
    pvalues: np.ndarray,
    *,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Holm-Bonferroni step-down correction for family-wise error rate.

    Returns (rejected, adjusted_p). adjusted_p is monotone non-decreasing in
    rank order and clipped to [0, 1]; rejected = adjusted_p < alpha.
    """
    pvalues = np.asarray(pvalues, dtype=float)
    if np.any((pvalues < 0) | (pvalues > 1)):
        raise ValueError("pvalues must be in [0, 1].")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1).")

    n = len(pvalues)
    order = np.argsort(pvalues)
    sorted_p = pvalues[order]
    multipliers = (n - np.arange(n)).astype(float)
    raw_adj = np.minimum(sorted_p * multipliers, 1.0)
    adj_sorted = np.maximum.accumulate(raw_adj)

    adjusted = np.empty_like(adj_sorted)
    adjusted[order] = adj_sorted
    rejected = adjusted < alpha
    return rejected, adjusted


def mcnemar_paired(
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    labels: np.ndarray,
    *,
    continuity: bool = True,
) -> float:
    """
    McNemar's test on discordant pairs between two paired binary classifiers.

    H0: P(A correct, B wrong) == P(A wrong, B correct). Returns two-sided p-value.
    """
    preds_a = np.asarray(preds_a)
    preds_b = np.asarray(preds_b)
    labels = np.asarray(labels)
    if preds_a.shape != preds_b.shape or preds_a.shape != labels.shape:
        raise ValueError("preds_a, preds_b, and labels must have identical shape.")

    correct_a = preds_a == labels
    correct_b = preds_b == labels
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    if b + c == 0:
        return 1.0

    if continuity:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    else:
        chi2 = (b - c) ** 2 / (b + c)
    return float(1.0 - stats.chi2.cdf(chi2, df=1))
