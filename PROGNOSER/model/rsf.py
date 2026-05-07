"""
rsf.py — Random Survival Forest wrapper around sksurv.ensemble.RandomSurvivalForest.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

from .base import SurvivalModel


class RSFWrapper(SurvivalModel):
    method_name = "rsf"

    def __init__(
        self,
        feature_columns: list[str],
        n_estimators: int = 200,
        min_samples_leaf: int = 5,
        max_features: str = "sqrt",
        random_state: int = 42,
        n_jobs: int = -1,
        scale_features: bool = True,
    ):
        self.feature_columns = list(feature_columns)
        self.n_estimators = int(n_estimators)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.random_state = int(random_state)
        self.n_jobs = int(n_jobs)
        self.scale_features = bool(scale_features)
        self.scaler_: StandardScaler | None = None
        self.rsf_: RandomSurvivalForest | None = None
        self.train_times_: np.ndarray | None = None

    def fit(self, X: np.ndarray, T: np.ndarray, E: np.ndarray, **kwargs) -> "RSFWrapper":
        Xt = X.astype(float)
        if self.scale_features:
            self.scaler_ = StandardScaler().fit(Xt)
            Xt = self.scaler_.transform(Xt)

        y = Surv.from_arrays(event=E.astype(bool), time=T.astype(float))
        self.rsf_ = RandomSurvivalForest(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.rsf_.fit(Xt, y)
        self.train_times_ = np.array(self.rsf_.unique_times_, dtype=float)
        return self

    def _transform(self, X: np.ndarray) -> np.ndarray:
        Xt = X.astype(float)
        if self.scaler_ is not None:
            Xt = self.scaler_.transform(Xt)
        return Xt

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        if self.rsf_ is None:
            raise RuntimeError("RSFWrapper not fitted")
        Xt = self._transform(X)
        return self.rsf_.predict(Xt)

    def predict_survival(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        if self.rsf_ is None:
            raise RuntimeError("RSFWrapper not fitted")
        Xt = self._transform(X)
        # predict_survival_function returns one StepFunction per sample
        sfs = self.rsf_.predict_survival_function(Xt, return_array=False)
        out = np.empty((len(sfs), len(times)), dtype=float)
        for i, sf in enumerate(sfs):
            out[i, :] = sf(times)
        return out

    @property
    def feature_importances_(self):
        if self.rsf_ is None:
            return None
        try:
            return self.rsf_.feature_importances_
        except (NotImplementedError, AttributeError):
            return None
