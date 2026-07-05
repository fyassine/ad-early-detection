"""
cox_time_varying.py — Time-varying Cox proportional hazards model.

Uses lifelines.CoxTimeVaryingFitter, which takes long-format data where each row
represents one inter-visit interval (start_months, stop_months, event, *features).
This correctly accounts for longitudinal covariate trajectories, treating each
visit's measurement as the hazard-relevant value for the interval that follows it.

Compatible with PROGNOSER_RUNNER.ipynb via the 'cox_time_varying' method option.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .base import SurvivalModel


class CoxTimeVaryingWrapper(SurvivalModel):
    """
    Wraps lifelines.CoxTimeVaryingFitter for long-format (start/stop) survival data.

    Unlike the wide-format CoxPHWrapper, this model is fitted with `fit_long(df_long)`
    rather than `fit(X, T, E)`. The wide-format methods `predict_risk` and
    `predict_survival` evaluate at the last available visit per subject.
    """

    method_name = "cox_time_varying"

    def __init__(
        self,
        feature_columns: list[str],
        penalizer: float = 0.01,
        l1_ratio: float = 0.0,
        scale_features: bool = True,
    ):
        self.feature_columns = list(feature_columns)
        self.penalizer = float(penalizer)
        self.l1_ratio = float(l1_ratio)
        self.scale_features = bool(scale_features)
        self.scaler_: StandardScaler | None = None
        self.cph_ = None
        self._long_df_: pd.DataFrame | None = None

    def fit(self, X: np.ndarray, T: np.ndarray, E: np.ndarray, **kwargs) -> "CoxTimeVaryingWrapper":
        """Wide-format fit (builds a single-interval long-format internally)."""
        df = pd.DataFrame(X, columns=self.feature_columns)
        df['subject_id'] = [f'subj_{i}' for i in range(len(X))]
        df['start_months'] = 0.0
        df['stop_months'] = T.astype(float)
        df['event'] = E.astype(int)
        return self.fit_long(df)

    def fit_long(
        self,
        df_long: pd.DataFrame,
        id_col: str = 'subject_id',
        start_col: str = 'start_months',
        stop_col: str = 'stop_months',
        event_col: str = 'event',
    ) -> "CoxTimeVaryingWrapper":
        """
        Fit on long-format DataFrame. Each row is one (subject, interval).

        df_long must have columns: id_col, start_col, stop_col, event_col, + feature_columns.
        """
        from lifelines import CoxTimeVaryingFitter

        df = df_long.copy()
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise KeyError(f'Missing feature columns in df_long: {missing}')

        df = df.dropna(subset=self.feature_columns + [start_col, stop_col, event_col])
        df = df[df[stop_col] > df[start_col]]  # drop zero-duration rows

        Xt = df[self.feature_columns].to_numpy(dtype=float)
        if self.scale_features:
            self.scaler_ = StandardScaler().fit(Xt)
            Xt = self.scaler_.transform(Xt)
        df = df.copy()
        df[self.feature_columns] = Xt

        self._long_df_ = df[[id_col, start_col, stop_col, event_col, *self.feature_columns]].copy()
        self.cph_ = CoxTimeVaryingFitter(penalizer=self.penalizer, l1_ratio=self.l1_ratio)
        self.cph_.fit(
            df,
            id_col=id_col,
            start_col=start_col,
            stop_col=stop_col,
            event_col=event_col,
            show_progress=False,
        )
        return self

    def _last_visit_features(self, X: np.ndarray) -> np.ndarray:
        Xt = X.astype(float)
        if self.scaler_ is not None:
            Xt = self.scaler_.transform(Xt)
        return Xt

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        """Evaluate partial hazard at the provided feature snapshot (e.g. last visit)."""
        if self.cph_ is None:
            raise RuntimeError("CoxTimeVaryingWrapper not fitted")
        Xt = self._last_visit_features(X)
        df = pd.DataFrame(Xt, columns=self.feature_columns)
        return self.cph_.predict_partial_hazard(df).to_numpy()

    def predict_survival(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        """
        Approximate survival via: S(t|x) = exp(-H_0(t) * exp(log_ph(x)))
        Uses the baseline cumulative hazard from the fitted model.
        """
        if self.cph_ is None:
            raise RuntimeError("CoxTimeVaryingWrapper not fitted")
        Xt = self._last_visit_features(X)
        df = pd.DataFrame(Xt, columns=self.feature_columns)

        # log partial hazard per subject (n,)
        log_ph = self.cph_.predict_log_partial_hazard(df).to_numpy()

        # Baseline cumulative hazard H_0(t) — indexed by event times
        bch = self.cph_.baseline_cumulative_hazard_  # DataFrame with one column
        bch_times = bch.index.to_numpy(dtype=float)
        bch_values = bch.iloc[:, 0].to_numpy(dtype=float)

        out = np.ones((len(X), len(times)), dtype=float)
        for j, t in enumerate(times):
            # Interpolate H_0(t)
            if t <= bch_times[0]:
                h0_t = 0.0
            elif t >= bch_times[-1]:
                h0_t = float(bch_values[-1])
            else:
                h0_t = float(np.interp(t, bch_times, bch_values))
            # S(t|x) = exp(-H_0(t) * exp(log_ph))
            out[:, j] = np.exp(-h0_t * np.exp(log_ph))
        return out

    def summary(self) -> pd.DataFrame:
        return self.cph_.summary if self.cph_ else pd.DataFrame()
