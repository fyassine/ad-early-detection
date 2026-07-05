"""
kaplan_meier.py — Population-level Kaplan-Meier baseline (lifelines).
"""

from __future__ import annotations

import numpy as np

from .base import SurvivalModel


class KMBaseline(SurvivalModel):
    """Non-parametric population survival curve. Ignores X.
    `predict_risk` returns a constant (no per-subject differentiation)."""

    method_name = "kaplan_meier"

    def __init__(self):
        self.kmf_ = None

    def fit(self, X: np.ndarray, T: np.ndarray, E: np.ndarray, **kwargs) -> "KMBaseline":
        from lifelines import KaplanMeierFitter
        self.kmf_ = KaplanMeierFitter()
        self.kmf_.fit(T, event_observed=E)
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        if self.kmf_ is None:
            raise RuntimeError("KMBaseline not fitted")
        return np.zeros(X.shape[0], dtype=float)

    def predict_survival(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        if self.kmf_ is None:
            raise RuntimeError("KMBaseline not fitted")
        sf = self.kmf_.survival_function_at_times(times).to_numpy()
        return np.tile(sf, (X.shape[0], 1))

    @property
    def survival_function_(self):
        return self.kmf_.survival_function_ if self.kmf_ else None
