"""
base.py — SurvivalModel abstract base class. All concrete wrappers
(Cox, RSF, DeepSurv) implement this interface so evaluation and
sweep code can treat them uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import joblib
import numpy as np


class SurvivalModel(ABC):
    """Unified interface for survival models.

    Convention: `predict_risk` returns higher-is-worse risk scores.
    `predict_survival(X, times)` returns survival probabilities of
    shape (n_samples, len(times)).
    """

    feature_columns: list[str] | None = None
    method_name: str = "abstract"

    @abstractmethod
    def fit(self, X: np.ndarray, T: np.ndarray, E: np.ndarray, **kwargs) -> "SurvivalModel": ...

    @abstractmethod
    def predict_risk(self, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def predict_survival(self, X: np.ndarray, times: np.ndarray) -> np.ndarray: ...

    def score(
        self,
        X: np.ndarray, T: np.ndarray, E: np.ndarray,
        X_train: np.ndarray | None = None,
        T_train: np.ndarray | None = None,
        E_train: np.ndarray | None = None,
        eval_times: Iterable[float] = (12, 24, 36, 48, 60, 72),
    ) -> dict:
        """Compute (c_index, ibs, time-dependent AUC). If train arrays are
        not provided, IBS/AUC are computed using the test data as the
        reference distribution (less ideal, but works)."""
        from PROGNOSER.common.metrics import evaluate_model
        return evaluate_model(
            self,
            X_train if X_train is not None else X,
            T_train if T_train is not None else T,
            E_train if E_train is not None else E,
            X, T, E,
            eval_times=eval_times,
        )

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "SurvivalModel":
        return joblib.load(path)
