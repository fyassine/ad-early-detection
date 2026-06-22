"""
lstm_surv.py — LSTM-based discrete-time survival model (DeepHit-lite).

Takes variable-length visit sequences → predicts per-time-bin hazard probabilities
→ survival curve.

Architecture:
    Input  : (batch, seq_len, n_features) padded visit sequences
    LSTM   : 1-2 layers, hidden_dim=64, dropout
    Output : Linear(hidden, n_time_bins) → sigmoid hazard per bin
    Loss   : Negative log-likelihood for discrete-time survival (interval-censoring aware)

Compatible with PROGNOSER_RUNNER.ipynb via the 'lstm_surv' method option.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .base import SurvivalModel


class _LSTMSurvNet(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_time_bins: int,
        hidden_dim: int = 64,
        n_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, n_time_bins)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """x: (batch, max_seq, n_feat), lengths: (batch,). Returns logit hazards (batch, n_bins)."""
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, (h_n, _) = self.lstm(packed)
        # Use final hidden state of last layer
        last_hidden = h_n[-1]  # (batch, hidden_dim)
        last_hidden = self.drop(last_hidden)
        return self.head(last_hidden)  # (batch, n_time_bins)


def _discrete_surv_nll(
    logits: torch.Tensor,
    T: torch.Tensor,
    E: torch.Tensor,
    bin_edges: torch.Tensor,
) -> torch.Tensor:
    """
    Discrete-time survival negative log-likelihood (Brown et al. style).

    logits  : (batch, n_bins) — raw outputs from head
    T       : (batch,)        — event/censoring times in original units
    E       : (batch,)        — event indicator (1=event, 0=censored)
    bin_edges : (n_bins+1,)   — time bin boundaries including 0 and max
    """
    hazards = torch.sigmoid(logits)  # (batch, n_bins)
    surv = torch.cumprod(1.0 - hazards + 1e-8, dim=1)  # (batch, n_bins)

    # Which bin does each subject's event/censor time fall into?
    n_bins = logits.size(1)
    # bin_idx[i] = index of first bin where bin_edges[idx+1] > T[i]
    bin_idx = torch.searchsorted(bin_edges[1:].contiguous(), T.unsqueeze(1)).squeeze(1)
    bin_idx = bin_idx.clamp(0, n_bins - 1)

    # Log-likelihood
    log_surv_prev = torch.log(
        torch.gather(torch.cat([torch.ones(surv.size(0), 1, device=surv.device), surv[:, :-1]], dim=1),
                     1, bin_idx.unsqueeze(1)).squeeze(1) + 1e-8
    )
    log_haz_at_event = torch.log(
        torch.gather(hazards, 1, bin_idx.unsqueeze(1)).squeeze(1) + 1e-8
    )
    ll = E * (log_surv_prev + log_haz_at_event) + (1 - E) * torch.log(
        torch.gather(surv, 1, bin_idx.unsqueeze(1)).squeeze(1) + 1e-8
    )
    return -ll.mean()


class LSTMSurvWrapper(SurvivalModel):
    """LSTM discrete-time survival model."""

    method_name = "lstm_surv"

    def __init__(
        self,
        feature_columns: list[str],
        n_time_bins: int = 12,
        max_horizon_months: int = 72,
        hidden_dim: int = 64,
        n_layers: int = 1,
        dropout: float = 0.2,
        lr: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 100,
        early_stopping_patience: int = 15,
        scale_features: bool = True,
        random_state: int = 42,
        device: str = 'cuda',
    ):
        self.feature_columns = list(feature_columns)
        self.n_time_bins = int(n_time_bins)
        self.max_horizon_months = float(max_horizon_months)
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.early_stopping_patience = int(early_stopping_patience)
        self.scale_features = bool(scale_features)
        self.random_state = int(random_state)
        self._device_str = device if (device == 'cpu' or torch.cuda.is_available()) else 'cpu'
        self.device = torch.device(self._device_str)

        self.scaler_: StandardScaler | None = None
        self.net_: _LSTMSurvNet | None = None
        self.bin_edges_: np.ndarray | None = None

    def _make_bins(self, T_train: np.ndarray) -> np.ndarray:
        t_max = min(self.max_horizon_months, float(T_train.max()))
        return np.linspace(0.0, t_max, self.n_time_bins + 1)

    def fit(self, X: np.ndarray, T: np.ndarray, E: np.ndarray, **kwargs) -> "LSTMSurvWrapper":
        """Wide-format fit: treats each subject as a single-visit sequence."""
        seq = X[:, np.newaxis, :]  # (n, 1, n_feat)
        lengths = np.ones(len(X), dtype=np.int64)
        return self.fit_sequences(seq, lengths, T, E)

    def fit_sequences(
        self,
        sequences: np.ndarray,
        lengths: np.ndarray,
        T: np.ndarray,
        E: np.ndarray,
        val_data: tuple | None = None,
        epoch_callback: "Callable[[int, float, float], None] | None" = None,
    ) -> "LSTMSurvWrapper":
        """
        Train on padded sequence tensors.

        sequences : (n_subjects, max_seq_len, n_features)
        lengths   : (n_subjects,) actual visit counts
        T         : (n_subjects,) durations
        E         : (n_subjects,) event flags
        val_data  : optional tuple (sequences_val, lengths_val, T_val, E_val)
        epoch_callback : optional ``fn(epoch, train_loss, val_loss)`` invoked once
            per epoch. Pure hook for external logging (e.g. W&B) — this module
            never imports the logger itself (keeps the model layer I/O-free).
        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        n_feat = sequences.shape[2]
        n_subj = len(sequences)

        if self.scale_features:
            # Fit scaler on all valid (non-padded) feature values
            flat = sequences.reshape(-1, n_feat)
            flat_mask = np.zeros(len(flat), dtype=bool)
            idx = 0
            for length in lengths:
                flat_mask[idx:idx + length] = True
                idx += length
            self.scaler_ = StandardScaler().fit(flat[flat_mask])
            seq_scaled = sequences.copy()
            seq_scaled = seq_scaled.reshape(-1, n_feat)
            seq_scaled[flat_mask] = self.scaler_.transform(seq_scaled[flat_mask])
            sequences = seq_scaled.reshape(sequences.shape)

        self.bin_edges_ = self._make_bins(T)
        bin_edges_t = torch.tensor(self.bin_edges_, dtype=torch.float32, device=self.device)

        self.net_ = _LSTMSurvNet(
            n_features=n_feat,
            n_time_bins=self.n_time_bins,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        seq_t = torch.tensor(sequences, dtype=torch.float32)
        len_t = torch.tensor(lengths, dtype=torch.long)
        T_t = torch.tensor(T, dtype=torch.float32)
        E_t = torch.tensor(E, dtype=torch.float32)

        best_val_loss = math.inf
        best_state = None
        patience_count = 0

        for epoch in range(self.epochs):
            self.net_.train()
            perm = torch.randperm(n_subj)
            train_loss = 0.0
            n_batches = 0
            for start in range(0, n_subj, self.batch_size):
                idx = perm[start: start + self.batch_size]
                xb = seq_t[idx].to(self.device)
                lb = len_t[idx]
                Tb = T_t[idx].to(self.device)
                Eb = E_t[idx].to(self.device)
                logits = self.net_(xb, lb)
                loss = _discrete_surv_nll(logits, Tb, Eb, bin_edges_t)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                n_batches += 1

            avg_train = train_loss / max(n_batches, 1)
            val_loss = avg_train

            if val_data is not None:
                seq_v, len_v, T_v, E_v = val_data
                val_loss = self._compute_loss(seq_v, len_v, T_v, E_v, bin_edges_t)

            scheduler.step(val_loss)

            if epoch_callback is not None:
                epoch_callback(epoch, avg_train, val_loss)

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.net_.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= self.early_stopping_patience:
                    print(f'Early stopping at epoch {epoch + 1} (val_loss={val_loss:.4f})')
                    break

        if best_state is not None:
            self.net_.load_state_dict(best_state)
        self.net_.eval()
        return self

    def _compute_loss(
        self, seq: np.ndarray, lengths: np.ndarray, T: np.ndarray, E: np.ndarray,
        bin_edges_t: torch.Tensor,
    ) -> float:
        if self.scale_features and self.scaler_ is not None:
            seq = seq.reshape(-1, seq.shape[2])
            seq = self.scaler_.transform(seq)
            seq = seq.reshape(seq.shape)
        seq_t = torch.tensor(seq, dtype=torch.float32).to(self.device)
        len_t = torch.tensor(lengths, dtype=torch.long)
        T_t = torch.tensor(T, dtype=torch.float32).to(self.device)
        E_t = torch.tensor(E, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.net_(seq_t, len_t)
            loss = _discrete_surv_nll(logits, T_t, E_t, bin_edges_t)
        return float(loss.item())

    def _seq_to_tensor(self, sequences: np.ndarray) -> torch.Tensor:
        if self.scale_features and self.scaler_ is not None:
            n, s, f = sequences.shape
            sequences = self.scaler_.transform(sequences.reshape(-1, f)).reshape(n, s, f)
        return torch.tensor(sequences, dtype=torch.float32)

    def _predict_hazards(self, sequences: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        if self.net_ is None:
            raise RuntimeError("LSTMSurvWrapper not fitted")
        seq_t = self._seq_to_tensor(sequences).to(self.device)
        len_t = torch.tensor(lengths, dtype=torch.long)
        with torch.no_grad():
            logits = self.net_(seq_t, len_t)
        return torch.sigmoid(logits).cpu().numpy()  # (n, n_bins)

    def _hazards_from_X(self, X: np.ndarray) -> np.ndarray:
        """Treat wide-format X as single-visit sequences."""
        seq = X[:, np.newaxis, :]
        lengths = np.ones(len(X), dtype=np.int64)
        return self._predict_hazards(seq, lengths)

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        """Cumulative hazard at max_horizon (higher = higher risk)."""
        hazards = self._hazards_from_X(X)
        # Complement of survival at final bin
        surv = np.cumprod(1.0 - hazards, axis=1)
        return 1.0 - surv[:, -1]

    def predict_risk_sequences(self, sequences: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        """Risk scores from padded sequence tensors."""
        hazards = self._predict_hazards(sequences, lengths)
        surv = np.cumprod(1.0 - hazards, axis=1)
        return 1.0 - surv[:, -1]

    def predict_survival(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Survival at requested times using discrete-bin interpolation."""
        hazards = self._hazards_from_X(X)
        return self._surv_at_times(hazards, times)

    def predict_survival_sequences(
        self, sequences: np.ndarray, lengths: np.ndarray, times: np.ndarray
    ) -> np.ndarray:
        hazards = self._predict_hazards(sequences, lengths)
        return self._surv_at_times(hazards, times)

    def _surv_at_times(self, hazards: np.ndarray, times: np.ndarray) -> np.ndarray:
        surv = np.cumprod(1.0 - hazards, axis=1)  # (n, n_bins)
        bin_edges = self.bin_edges_
        out = np.ones((len(hazards), len(times)), dtype=float)
        for j, t in enumerate(times):
            # Which bin does t fall into?
            bin_idx = int(np.searchsorted(bin_edges[1:], t))
            bin_idx = min(bin_idx, surv.shape[1] - 1)
            if t > 0:
                out[:, j] = surv[:, bin_idx]
        return out
