"""
GELSTM/models.py — Graph Encoder + LSTM Classifier for longitudinal MCI conversion.

Architecture:
    Per-visit FC matrix → shared GAAE encoder → z_t ∈ R^d
    [z_t ‖ Δt_t] → LSTM → h_n → Linear(1) → P(converter)

Δt_t = months since previous visit / 96  (0 for first visit)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Allow importing GAAE model relative to the CLASSIFIER root
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model.GAAE.models import GraphAttentionAutoencoderConditioned


def build_classifier_head(
    lstm_hidden: int, classifier_hidden: int, dropout: float, classifier_norm: str = "none"
) -> nn.Module:
    """RNN-head: ``Linear→[LayerNorm]→ReLU→Dropout→Linear(1)`` (or a direct Linear).

    ``classifier_norm="layernorm"`` inserts a ``LayerNorm`` after the first linear —
    LayerNorm (not BatchNorm) so it is safe for variable-length RNN eval and batch
    size 1. Shared by the constructor and the FDR recurrent-core patch so both heads
    stay identical. ``classifier_hidden <= 0`` gives a direct ``Linear(lstm_hidden, 1)``.
    """
    if classifier_hidden <= 0:
        return nn.Linear(lstm_hidden, 1)
    layers: list[nn.Module] = [nn.Linear(lstm_hidden, classifier_hidden)]
    if str(classifier_norm).lower() == "layernorm":
        layers.append(nn.LayerNorm(classifier_hidden))
    layers += [nn.ReLU(), nn.Dropout(dropout), nn.Linear(classifier_hidden, 1)]
    return nn.Sequential(*layers)


class GELSTMClassifier(nn.Module):
    """
    GNN Encoder + LSTM Classifier for longitudinal MCI → AD conversion prediction.

    The GAAE encoder is applied independently to each visit graph (shared weights),
    producing a graph-level embedding z_t per visit. Optionally, the inter-visit
    interval Δt_t (normalised months) is concatenated to z_t before the LSTM.

    Parameters
    ----------
    in_features : int
        Number of ROIs (= FC matrix dimension, e.g. 200 for Schaefer-200).
    gaae_hidden : int
        Hidden dim of GAAE encoder (typically == in_features).
    gaae_latent : int
        Latent dim of GAAE encoder (e.g. 64).
    gaae_heads : int
        Number of GAT attention heads.
    gaae_cond_dim : int
        Conditioning vector size (2: sex + age).
    gaae_dropout : float
        Dropout used in GAAE encoder.
    lstm_hidden : int
        LSTM hidden state size.
    lstm_layers : int
        Number of LSTM layers.
    lstm_dropout : float
        Dropout between LSTM layers (only active when lstm_layers > 1).
    use_time_delta : bool
        Whether to concatenate normalised Δt to z_t before LSTM input.
    classifier_hidden : int
        Size of optional hidden layer before final linear classifier (0 = direct).
    rnn_type : str
        Recurrent cell type: 'lstm' (default) or 'gru'. A GRU has 3 gates vs the
        LSTM's 4, so ~25% fewer recurrent parameters at the same hidden size.
    """

    def __init__(
        self,
        in_features: int,
        gaae_hidden: int,
        gaae_latent: int,
        gaae_heads: int,
        gaae_cond_dim: int,
        gaae_dropout: float,
        lstm_hidden: int,
        lstm_layers: int,
        lstm_dropout: float,
        use_time_delta: bool = True,
        classifier_hidden: int = 64,
        rnn_type: str = "lstm",
        classifier_norm: str = "none",
    ):
        super().__init__()
        self.gaae_latent    = gaae_latent
        self.use_time_delta = use_time_delta
        self.rnn_type       = rnn_type.lower()
        if self.rnn_type not in ("lstm", "gru"):
            raise ValueError(f"rnn_type must be 'lstm' or 'gru', got {rnn_type!r}")
        self.classifier_norm = str(classifier_norm).lower()
        if self.classifier_norm not in ("none", "layernorm"):
            raise ValueError(
                f"classifier_norm must be 'none' or 'layernorm', got {classifier_norm!r}"
            )

        # ── Per-visit embedding standardisation (z-score) ───────────────────
        # Fitted on the *training fold's* pooled GAAE embeddings via
        # ``set_feature_norm`` and applied inside ``encode_visit``. Persistent
        # buffers so the fitted statistics round-trip through the checkpoint.
        # Defaults are the identity transform (mean=0, std=1), so a model that
        # never calls ``set_feature_norm`` behaves exactly as before.
        self.register_buffer("feat_mean", torch.zeros(gaae_latent))
        self.register_buffer("feat_std",  torch.ones(gaae_latent))

        # ── Shared GAAE encoder (applied per-visit) ─────────────────────────
        self.encoder = GraphAttentionAutoencoderConditioned(
            in_features=in_features,
            hidden_dim=gaae_hidden,
            out_features=gaae_latent,
            cond_dim=gaae_cond_dim,
            num_heads=gaae_heads,
            dropout=gaae_dropout,
        )

        # ── Recurrent core (LSTM or GRU) ─────────────────────────────────────
        # The attribute stays named ``self.lstm`` so checkpoint state-dict keys
        # ("lstm.*") and downstream code are unchanged; only the cell type
        # switches with ``rnn_type``.
        self.lstm_input_dim = gaae_latent + (1 if use_time_delta else 0)
        rnn_cls = nn.GRU if self.rnn_type == "gru" else nn.LSTM
        self.lstm = rnn_cls(
            input_size=self.lstm_input_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )

        # ── Classifier head ─────────────────────────────────────────────────
        self.classifier = build_classifier_head(
            lstm_hidden, classifier_hidden, lstm_dropout, self.classifier_norm
        )

    # ── Encoder helpers ──────────────────────────────────────────────────────

    def encode_visit(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
        pool: str = "mean",
    ) -> torch.Tensor:
        """
        Encode a single visit graph → graph-level embedding z ∈ R^gaae_latent.

        Parameters
        ----------
        x : (N_nodes, in_features)
        edge_index : (2, E)
        edge_attr : (E,) or (E, 1) or None
        pool : 'mean' | 'max' | 'sum'
        """
        z = self.encoder.encode(x, edge_index, edge_attr)  # (N_nodes, latent)
        if pool == "mean":
            z = z.mean(dim=0)
        elif pool == "max":
            z = z.max(dim=0).values
        else:
            z = z.sum(dim=0)
        # Standardise with the fitted (or identity, by default) statistics.
        return (z - self.feat_mean) / self.feat_std

    def set_feature_norm(self, mean, std, *, eps: float = 1e-8) -> None:
        """Set the per-visit embedding standardisation statistics.

        ``mean`` / ``std`` are length-``gaae_latent`` arrays (e.g. a fitted
        ``sklearn.preprocessing.StandardScaler``'s ``mean_`` / ``scale_``).
        Fit on the training fold only and call once per fold *before* training
        so test/val embeddings are standardised with train-fold statistics —
        the GELSTM analogue of the per-fold StandardScaler in the GEC-MLP.
        """
        mean_t = torch.as_tensor(mean, dtype=self.feat_mean.dtype, device=self.feat_mean.device)
        std_t  = torch.as_tensor(std,  dtype=self.feat_std.dtype,  device=self.feat_std.device)
        if mean_t.shape != self.feat_mean.shape or std_t.shape != self.feat_std.shape:
            raise ValueError(
                f"feature-norm shapes {tuple(mean_t.shape)}/{tuple(std_t.shape)} "
                f"do not match latent dim {tuple(self.feat_mean.shape)}"
            )
        self.feat_mean.copy_(mean_t)
        self.feat_std.copy_(std_t.clamp_min(eps))

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        packed_seqs: "torch.nn.utils.rnn.PackedSequence",
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        packed_seqs : PackedSequence of shape (sum_T, lstm_input_dim)
            Pre-packed sequence of embeddings (optionally with Δt appended).

        Returns
        -------
        logits : (B,)  — one scalar per subject
        """
        if self.rnn_type == "gru":
            _, h_n = self.lstm(packed_seqs)        # GRU returns (output, h_n)
        else:
            _, (h_n, _) = self.lstm(packed_seqs)   # LSTM returns (output, (h_n, c_n))
        # h_n : (num_layers, B, hidden)
        h_last = h_n[-1]           # (B, hidden) — last layer hidden state
        logits = self.classifier(h_last).squeeze(-1)   # (B,)
        return logits

    # ── Freeze / unfreeze encoder ────────────────────────────────────────────

    def freeze_encoder(self):
        """Freeze all GAAE encoder + FiLM parameters."""
        enc_modules = [
            self.encoder.encoder_gat1, self.encoder.encoder_bn1,
            self.encoder.encoder_gat2, self.encoder.encoder_bn2,
            self.encoder.encoder_gat3,
            self.encoder.film_gamma,   self.encoder.film_beta,
        ]
        for mod in enc_modules:
            for p in mod.parameters():
                p.requires_grad_(False)

    def unfreeze_encoder(self):
        """Unfreeze all GAAE encoder + FiLM parameters."""
        enc_modules = [
            self.encoder.encoder_gat1, self.encoder.encoder_bn1,
            self.encoder.encoder_gat2, self.encoder.encoder_bn2,
            self.encoder.encoder_gat3,
            self.encoder.film_gamma,   self.encoder.film_beta,
        ]
        for mod in enc_modules:
            for p in mod.parameters():
                p.requires_grad_(True)

    def load_gaae_weights(self, ckpt_path: str, device: str | torch.device = "cpu"):
        """
        Load GAAE encoder weights from a GAAE checkpoint into self.encoder.
        Automatically freezes the encoder after loading.

        Parameters
        ----------
        ckpt_path : str
            Path to model_{run_name}.pth saved by GAAE training.
        device : str or torch.device
        """
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict):
            gaae_sd = ckpt
        else:
            gaae_sd = ckpt.state_dict()

        own_sd  = self.encoder.state_dict()
        to_load = {k: v for k, v in gaae_sd.items() if k in own_sd and v.shape == own_sd[k].shape}
        missing = set(own_sd) - set(to_load)
        if missing:
            print(f"[GAAE load] Keys not transferred (shape mismatch or absent): {sorted(missing)}")

        own_sd.update(to_load)
        self.encoder.load_state_dict(own_sd)
        self.freeze_encoder()
        print(f"[GAAE load] Loaded {len(to_load)} parameters from {ckpt_path}")

    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]
