"""
brain_age.py — Functional brain-age regressor + per-visit brain-age gap.

Trains a Ridge regression mapping vectorized FC (upper triangle of the
correlation matrix) -> chronological age on the *healthy CN* baseline
subjects of the active cohort. The brain-age gap (BAG = predicted age
- chronological age) is then computed for any visit and serves as a
single-number summary of accelerated functional aging.

Recent AD-specific brain-age work (Lee et al. 2024; Wei et al. 2024;
Yu et al. 2023) shows that an elevated BAG separates MCI / converter /
AD groups even when individual ROI features are unstable across sites.

Implementation choices:

  * Ridge with k-fold CV out-of-sample prediction on the CN baselines
    (so the cohort distribution we report is honest, not in-sample).
  * No site/scanner harmonisation here — we expect upstream pipelines
    (e.g. ComBat) or single-site cohorts. The frontend tells the user.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def _safe_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if not math.isfinite(f) else f


@dataclass
class BrainAgeModel:
    """A trained Ridge brain-age model + summary diagnostics."""
    n_train: int = 0
    n_features: int = 0
    age_mean: float = 0.0
    cv_mae: Optional[float] = None        # cross-validated mean absolute error
    cv_r2: Optional[float] = None
    bias_slope: Optional[float] = None    # slope of (predicted - true) on true (Smith 2019 correction)
    bias_intercept: Optional[float] = None
    # CV-based predicted ages for the healthy cohort (used to render the
    # cohort BAG distribution + percentile context).
    cohort_bag: list[float] = field(default_factory=list)
    # Persisted weights for inference. Ridge ``coef_`` + ``intercept_``.
    coef: Optional[np.ndarray] = None
    intercept: float = 0.0


def _vectorize(matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    iu = np.triu_indices(n, k=1)
    v = matrix[iu].astype(np.float32, copy=False)
    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)


def fit_brain_age(
    fc_features: np.ndarray,           # (N, n_edges)
    ages: np.ndarray,                  # (N,)
    n_splits: int = 5,
    alpha: float = 1.0,
    seed: int = 42,
) -> Optional[BrainAgeModel]:
    """
    Fit a Ridge brain-age model on healthy controls. Returns ``None`` if
    the input is too small (need ≥ 12 subjects with valid ages).
    """
    fc_features = np.asarray(fc_features, dtype=np.float32)
    ages = np.asarray(ages, dtype=np.float64)
    mask = np.isfinite(ages)
    fc_features = fc_features[mask]
    ages = ages[mask]
    n, n_features = fc_features.shape if fc_features.ndim == 2 else (0, 0)
    if n < 12:
        return None

    try:
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import KFold
    except ImportError:
        return None

    n_splits = min(n_splits, n)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cv_pred = np.empty_like(ages)
    for tr, te in kf.split(fc_features):
        m = Ridge(alpha=alpha)
        m.fit(fc_features[tr], ages[tr])
        cv_pred[te] = m.predict(fc_features[te])

    abs_err = np.abs(cv_pred - ages)
    cv_mae = float(abs_err.mean())
    ss_res = float(((ages - cv_pred) ** 2).sum())
    ss_tot = float(((ages - ages.mean()) ** 2).sum())
    cv_r2 = float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else None

    # Smith 2019 bias correction: regress (cv_pred - ages) on ages, store
    # slope/intercept so we can de-bias future predictions.
    diff = cv_pred - ages
    A = np.vstack([ages, np.ones_like(ages)]).T
    try:
        slope, intercept = np.linalg.lstsq(A, diff, rcond=None)[0]
    except Exception:
        slope, intercept = 0.0, 0.0

    cohort_bag = (cv_pred - ages).tolist()

    # Final model trained on all CN baselines for inference on patients.
    final = Ridge(alpha=alpha)
    final.fit(fc_features, ages)

    return BrainAgeModel(
        n_train=int(n),
        n_features=int(n_features),
        age_mean=float(ages.mean()),
        cv_mae=_safe_float(cv_mae),
        cv_r2=_safe_float(cv_r2),
        bias_slope=_safe_float(slope),
        bias_intercept=_safe_float(intercept),
        cohort_bag=[float(b) for b in cohort_bag],
        coef=np.asarray(final.coef_, dtype=np.float32),
        intercept=float(final.intercept_),
    )


def predict_brain_age(
    model: BrainAgeModel,
    matrix: np.ndarray,
    chronological_age: Optional[float] = None,
) -> dict:
    """
    Apply a trained brain-age model to one correlation matrix. Returns
    ``{predicted_age, brain_age_gap, brain_age_gap_corrected}`` with
    ``None`` fields when inputs are missing.
    """
    out: dict = {"predicted_age": None, "brain_age_gap": None, "brain_age_gap_corrected": None}
    if model is None or model.coef is None:
        return out
    if matrix is None or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return out
    vec = _vectorize(matrix)
    if vec.size != model.n_features:
        return out
    pred = float(np.dot(vec, model.coef) + model.intercept)
    out["predicted_age"] = _safe_float(pred)

    if chronological_age is None or not math.isfinite(chronological_age):
        return out
    gap = pred - float(chronological_age)
    out["brain_age_gap"] = _safe_float(gap)

    if model.bias_slope is not None and model.bias_intercept is not None:
        # Smith 2019 correction applied at inference time.
        bias = model.bias_slope * float(chronological_age) + model.bias_intercept
        out["brain_age_gap_corrected"] = _safe_float(gap - bias)
    return out
