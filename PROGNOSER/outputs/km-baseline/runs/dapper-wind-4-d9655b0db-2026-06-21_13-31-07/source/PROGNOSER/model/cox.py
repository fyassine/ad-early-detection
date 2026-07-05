"""
cox.py — Cox Proportional Hazards wrapper around lifelines.CoxPHFitter.

Three factory helpers for the common feature configurations:
    CoxPHWrapper.with_clinical_features()
    CoxPHWrapper.with_embedding_features(latent_dim=64)
    CoxPHWrapper.with_clinical_plus_embedding(latent_dim=64, use_pca=True)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .base import SurvivalModel


class CoxPHWrapper(SurvivalModel):
    method_name = "cox"

    def __init__(
        self,
        feature_columns: list[str],
        penalizer: float = 0.01,
        l1_ratio: float = 0.0,
        scale_features: bool = True,
        pca_components: int | None = None,
    ):
        self.feature_columns = list(feature_columns)
        self.penalizer = float(penalizer)
        self.l1_ratio = float(l1_ratio)
        self.scale_features = bool(scale_features)
        self.pca_components = pca_components
        self.scaler_: StandardScaler | None = None
        self.pca_: PCA | None = None
        self.cph_ = None
        self.fitted_columns_: list[str] | None = None

    @classmethod
    def with_clinical_features(cls, **kwargs) -> "CoxPHWrapper":
        return cls(
            feature_columns=["age", "sex", "mmstot", "cdrglobal", "apoe4"],
            **kwargs,
        )

    @classmethod
    def with_embedding_features(cls, latent_dim: int = 64, use_pca: bool = False,
                                pca_components: int = 16, **kwargs) -> "CoxPHWrapper":
        cols = [f"z_{i}" for i in range(latent_dim)]
        return cls(
            feature_columns=cols,
            pca_components=pca_components if use_pca else None,
            **kwargs,
        )

    @classmethod
    def with_clinical_plus_embedding(
        cls, latent_dim: int = 64, use_pca: bool = True, pca_components: int = 16, **kwargs
    ) -> "CoxPHWrapper":
        cols = ["age", "sex", "mmstot", "cdrglobal", "apoe4"] + [f"z_{i}" for i in range(latent_dim)]
        return cls(
            feature_columns=cols,
            pca_components=pca_components if use_pca else None,
            **kwargs,
        )

    def _transform(self, X: np.ndarray) -> np.ndarray:
        Xt = X.astype(float)
        if self.scaler_ is not None:
            Xt = self.scaler_.transform(Xt)
        if self.pca_ is not None:
            Xt = self.pca_.transform(Xt)
        return Xt

    def fit(self, X: np.ndarray, T: np.ndarray, E: np.ndarray, **kwargs) -> "CoxPHWrapper":
        from lifelines import CoxPHFitter

        Xt = X.astype(float)
        if self.scale_features:
            self.scaler_ = StandardScaler().fit(Xt)
            Xt = self.scaler_.transform(Xt)

        if self.pca_components is not None and self.pca_components < Xt.shape[1]:
            self.pca_ = PCA(n_components=self.pca_components, random_state=42).fit(Xt)
            Xt = self.pca_.transform(Xt)
            cols = [f"pc_{i}" for i in range(Xt.shape[1])]
        else:
            cols = list(self.feature_columns)
            if len(cols) != Xt.shape[1]:
                cols = [f"x_{i}" for i in range(Xt.shape[1])]

        self.fitted_columns_ = cols
        df = pd.DataFrame(Xt, columns=cols)
        df["duration"] = T
        df["event"] = E

        self.cph_ = CoxPHFitter(penalizer=self.penalizer, l1_ratio=self.l1_ratio)
        self.cph_.fit(df, duration_col="duration", event_col="event", show_progress=False)
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        if self.cph_ is None:
            raise RuntimeError("CoxPHWrapper not fitted")
        Xt = self._transform(X)
        df = pd.DataFrame(Xt, columns=self.fitted_columns_)
        return self.cph_.predict_partial_hazard(df).to_numpy()

    def predict_survival(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        if self.cph_ is None:
            raise RuntimeError("CoxPHWrapper not fitted")
        Xt = self._transform(X)
        df = pd.DataFrame(Xt, columns=self.fitted_columns_)
        # lifelines.predict_survival_function returns DataFrame indexed by time
        sf = self.cph_.predict_survival_function(df, times=times)
        # sf has shape (n_times, n_samples)
        return sf.to_numpy().T

    def summary(self) -> pd.DataFrame:
        if self.cph_ is None:
            return pd.DataFrame()
        return self.cph_.summary
