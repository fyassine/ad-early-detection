"""
effect_sizes.py — Cohen's d + bootstrap 95% CI for cohort comparisons.

Uses Hedges' small-sample correction so estimates are unbiased on the
typical converter cohorts (n=20-50 per group). 2023+ best practice
(Lakens 2024 update on effect-size reporting) prefers reporting d with
a percentile bootstrap CI rather than parametric SE.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


def cohens_d(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Hedges-corrected Cohen's d between two samples. Returns None if either is empty."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    y = np.asarray(y, dtype=np.float64)
    y = y[np.isfinite(y)]
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return None
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    pooled = math.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled < 1e-12:
        return None
    d = (mx - my) / pooled
    # Hedges' g correction
    j = 1.0 - (3.0 / (4.0 * (nx + ny) - 9.0))
    return float(d * j)


def bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[Optional[float], Optional[float]]:
    """Percentile bootstrap CI for Cohen's d. Returns (lower, upper)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    y = np.asarray(y, dtype=np.float64)
    y = y[np.isfinite(y)]
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return None, None

    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        bx = rng.choice(x, size=nx, replace=True)
        by = rng.choice(y, size=ny, replace=True)
        d = cohens_d(bx, by)
        samples[b] = d if d is not None else np.nan

    samples = samples[np.isfinite(samples)]
    if samples.size < 10:
        return None, None
    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1 - alpha / 2))
    return lo, hi


def pairwise_effect_sizes(
    cohort_values: dict[str, list[float]],
    n_boot: int = 1000,
    seed: int = 42,
) -> list[dict]:
    """
    All pairwise Cohen's d comparisons between cohorts.

    Input: ``{cohort_name: [values]}``.
    Returns: list of ``{a, b, n_a, n_b, d, ci_lo, ci_hi}`` dicts.
    """
    out: list[dict] = []
    cohorts = [c for c, v in cohort_values.items()
               if v is not None and len([x for x in v if x is not None and math.isfinite(x)]) >= 2]
    arrays = {c: np.asarray([v for v in cohort_values[c] if v is not None and math.isfinite(v)], dtype=np.float64)
              for c in cohorts}
    for i, a in enumerate(cohorts):
        for b in cohorts[i + 1:]:
            xa, xb = arrays[a], arrays[b]
            d = cohens_d(xa, xb)
            ci_lo, ci_hi = bootstrap_ci(xa, xb, n_boot=n_boot, seed=seed)
            out.append({
                "a": a, "b": b,
                "n_a": int(len(xa)), "n_b": int(len(xb)),
                "d": d, "ci_lo": ci_lo, "ci_hi": ci_hi,
            })
    return out
