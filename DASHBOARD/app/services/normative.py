"""
normative.py — Bootstrapped percentile bands for normative modelling.

Replaces the legacy ``mean ± 1·std`` bands with non-parametric percentile
curves (5/25/50/75/95) computed on a reference cohort (typically the
MCI-NC group). This mirrors the PCNtoolkit / centile-brain-chart
conventions that have become standard since Rutherford et al. 2022 *eLife*
and Bethlehem et al. 2022 *Nature*.

For each metric in ``BIOMARKER_KEYS`` we bootstrap ``n_boot`` resamples
of the reference cohort and report quantile estimates with a 95% CI on
the median. The frontend uses the percentiles to draw shaded bands and
the median CI to optionally shade the median line.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

PERCENTILES = (5, 25, 50, 75, 95)


def _safe_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if not math.isfinite(f) else f


def percentile_bands(
    values: list[float],
    percentiles: tuple = PERCENTILES,
    n_boot: int = 500,
    seed: int = 42,
) -> Optional[dict]:
    """
    Compute percentile values + bootstrap CIs from a 1-D sample.

    Returns ``{p5, p25, p50, p75, p95, n, mean, std,
               median_ci_lo, median_ci_hi}`` or ``None`` if too few data.
    """
    arr = np.asarray([v for v in values if v is not None and math.isfinite(float(v))], dtype=np.float64)
    if arr.size < 5:
        return None

    point = {f"p{p}": _safe_float(np.quantile(arr, p / 100.0)) for p in percentiles}

    rng = np.random.default_rng(seed)
    medians = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        medians[b] = float(np.median(sample))

    point.update({
        "n": int(arr.size),
        "mean": _safe_float(arr.mean()),
        "std": _safe_float(arr.std(ddof=1)) if arr.size > 1 else None,
        "median_ci_lo": _safe_float(np.quantile(medians, 0.025)),
        "median_ci_hi": _safe_float(np.quantile(medians, 0.975)),
    })
    return point


def cohort_percentile_bands(
    cohort_to_values: dict[str, list[float]],
    percentiles: tuple = PERCENTILES,
    n_boot: int = 500,
    seed: int = 42,
) -> dict:
    """
    Compute percentile bands for every cohort in the input mapping.
    Returns ``{cohort: bands_or_None}``.
    """
    out: dict = {}
    for cohort, values in cohort_to_values.items():
        out[cohort] = percentile_bands(values, percentiles=percentiles,
                                       n_boot=n_boot, seed=seed)
    return out


def patient_percentile(value: float, reference_values: list[float]) -> Optional[float]:
    """
    Empirical percentile of ``value`` within ``reference_values`` (0-100).
    Useful for the patient overview card: 'this visit sits at the 12th
    percentile of MCI-NC'.
    """
    if value is None:
        return None
    arr = np.asarray([v for v in reference_values if v is not None and math.isfinite(float(v))], dtype=np.float64)
    if arr.size < 5:
        return None
    rank = float(np.mean(arr <= value)) * 100.0
    return _safe_float(rank)
