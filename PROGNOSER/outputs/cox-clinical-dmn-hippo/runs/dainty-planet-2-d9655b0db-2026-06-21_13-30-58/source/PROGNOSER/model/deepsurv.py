"""
deepsurv.py — DeepSurv (neural Cox PH) wrapper around pycox.

Stretch goal — requires `pip install pycox torchtuples`. Raises ImportError
with a helpful hint if those are not installed.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from .base import SurvivalModel


def _require_pycox():
    try:
        import torchtuples as tt  # noqa: F401
        from pycox.models import CoxPH  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            'DeepSurvWrapper requires pycox + torchtuples. Install with:\n'
            '  pip install pycox torchtuples'
        ) from exc


class DeepSurvWrapper(SurvivalModel):
    method_name = "deepsurv"

    def __init__(
        self,
        feature_columns: list[str],
        hidden_layers: list[int] | None = None,
        dropout: float = 0.1,
        lr: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 200,
        early_stopping_patience: int = 20,
        scale_features: bool = True,
        random_state: int = 42,
    ):
        self.feature_columns = list(feature_columns)
        self.hidden_layers = hidden_layers if hidden_layers is not None else [64, 32]
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.early_stopping_patience = int(early_stopping_patience)
        self.scale_features = bool(scale_features)
        self.random_state = int(random_state)
        self.scaler_: StandardScaler | None = None
        self.model_ = None
        self.n_features_in_: int | None = None
        self.train_times_: np.ndarray | None = None

    def fit(
        self, X: np.ndarray, T: np.ndarray, E: np.ndarray,
        X_val: np.ndarray | None = None,
        T_val: np.ndarray | None = None,
        E_val: np.ndarray | None = None,
        **kwargs,
    ) -> "DeepSurvWrapper":
        _require_pycox()
        import torch
        import torchtuples as tt
        from pycox.models import CoxPH

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        Xt = X.astype(np.float32)
        if self.scale_features:
            self.scaler_ = StandardScaler().fit(Xt)
            Xt = self.scaler_.transform(Xt).astype(np.float32)

        self.n_features_in_ = Xt.shape[1]
        net = tt.practical.MLPVanilla(
            in_features=self.n_features_in_,
            num_nodes=self.hidden_layers,
            out_features=1,
            batch_norm=True,
            dropout=self.dropout,
            output_bias=False,
        )
        self.model_ = CoxPH(net, tt.optim.Adam(lr=self.lr))
        y_train = (T.astype(np.float32), E.astype(np.float32))

        val_data = None
        if X_val is not None and T_val is not None and E_val is not None and len(X_val) > 0:
            Xv = X_val.astype(np.float32)
            if self.scaler_ is not None:
                Xv = self.scaler_.transform(Xv).astype(np.float32)
            val_data = (Xv, (T_val.astype(np.float32), E_val.astype(np.float32)))

        callbacks = [tt.callbacks.EarlyStopping(patience=self.early_stopping_patience)] if val_data else []
        self.model_.fit(
            Xt, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=callbacks,
            val_data=val_data,
            verbose=False,
        )
        # Compute baseline hazards needed for survival functions
        self.model_.compute_baseline_hazards()
        self.train_times_ = np.array(self.model_.baseline_hazards_.index, dtype=float)
        return self

    def _transform(self, X: np.ndarray) -> np.ndarray:
        Xt = X.astype(np.float32)
        if self.scaler_ is not None:
            Xt = self.scaler_.transform(Xt).astype(np.float32)
        return Xt

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("DeepSurvWrapper not fitted")
        Xt = self._transform(X)
        # log-partial-hazard; higher = worse
        return np.asarray(self.model_.predict(Xt)).flatten()

    def predict_survival(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("DeepSurvWrapper not fitted")
        Xt = self._transform(X)
        # surv: DataFrame indexed by baseline_hazards_ times, columns = samples
        surv = self.model_.predict_surv_df(Xt)
        # Interpolate at requested times
        out = np.empty((Xt.shape[0], len(times)), dtype=float)
        surv_times = surv.index.to_numpy().astype(float)
        for j, t in enumerate(times):
            # Use last-observed-value (step function evaluation)
            idx = np.searchsorted(surv_times, t, side='right') - 1
            if idx < 0:
                out[:, j] = 1.0
            else:
                out[:, j] = surv.iloc[idx].to_numpy()
        return out
