"""TFGN/models.py — Temporal-First Graph Network classifier.

Assembles the TFGN pipeline from switchable stages:
  NodeSharedLSTM → [TemporalSaliencyGate] → [GVAEEncoder → Fusion] → Readout → DualScoreReadout

Every stage is controlled by config flags so the ablation ladder is one model with knobs.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT = _ROOT.parent
for _p in (str(_REPO_ROOT), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from .layers import (  # noqa: E402
    AttentivePool,
    ConcatResidualFusion,
    DualScoreReadout,
    GVAEEncoder,
    NodeSharedLSTM,
    TemporalSaliencyGate,
)


class TFGNClassifier(nn.Module):
    def __init__(
        self,
        *,
        n_rois: int,
        lstm_hidden: int,
        lstm_layers: int,
        lstm_dropout: float,
        gvae_hidden: int,
        gvae_latent: int,
        gvae_heads: int,
        gvae_dropout: float,
        cond_dim: int = 2,
        use_gate: bool,
        recon_target: str,
        fusion: str,
        readout: str,
        dual_score: bool,
    ):
        super().__init__()

        valid_recon_targets = ("delta_a_topk", "delta_a_mse", "a_last", "none")
        if recon_target not in valid_recon_targets:
            raise ValueError(f"Unknown recon_target '{recon_target}'. Valid options: {valid_recon_targets}")

        valid_fusions = ("concat_residual", "z_only")
        if fusion not in valid_fusions:
            raise ValueError(f"Unknown fusion '{fusion}'. Valid options: {valid_fusions}")

        valid_readouts = ("mean", "attention")
        if readout not in valid_readouts:
            raise ValueError(f"Unknown readout '{readout}'. Valid options: {valid_readouts}")

        self.use_gate = use_gate
        self.recon_target = recon_target
        self.fusion = fusion
        self.readout = readout
        self.dual_score = dual_score

        self.lstm = NodeSharedLSTM(
            input_dim=n_rois,
            hidden_dim=lstm_hidden,
            num_layers=lstm_layers,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0
        )

        if self.use_gate:
            self.gate = TemporalSaliencyGate(hidden_dim=lstm_hidden)
        else:
            self.gate = None

        if self.recon_target != "none":
            self.gvae = GVAEEncoder(
                in_features=n_rois,
                hidden_dim=gvae_hidden,
                latent_dim=gvae_latent,
                num_heads=gvae_heads,
                dropout=gvae_dropout,
                cond_dim=cond_dim
            )
        else:
            self.gvae = None

        if self.recon_target != "none":
            if self.fusion == "concat_residual":
                self.fusion_module = ConcatResidualFusion(
                    h_dim=lstm_hidden,
                    z_dim=gvae_latent,
                    out_dim=lstm_hidden
                )
                readout_dim = lstm_hidden
            else:
                self.fusion_module = None
                readout_dim = gvae_latent
        else:
            self.fusion_module = None
            readout_dim = lstm_hidden

        if self.readout == "attention":
            self.att_pool = AttentivePool(hidden_dim=readout_dim)
        else:
            self.att_pool = None

        self.cls_head = DualScoreReadout(
            hidden_dim=readout_dim,
            use_dual=self.dual_score
        )

    def forward(
        self,
        X: torch.Tensor,
        log_dt: torch.Tensor,
        A0_edge_index: torch.Tensor,
        A0_edge_attr: torch.Tensor | None,
        cond_vec: torch.Tensor | None
    ) -> dict:
        is_single = (X.dim() == 3)
        if is_single:
            X_expanded = X.unsqueeze(0)
        else:
            X_expanded = X

        h = self.lstm(X_expanded)

        h_T = h[:, -1, :, :]
        if is_single:
            h_T = h_T.squeeze(0)

        gate_scores = None
        if self.use_gate:
            h_T, gate_scores = self.gate(h_T)

        mu, logvar, recon_logits = None, None, None
        if self.recon_target != "none":
            X_0 = X[0] if is_single else X[:, 0, :, :]
            z, mu, logvar = self.gvae(X_0, A0_edge_index, A0_edge_attr, cond_vec)

            if self.recon_target in ("delta_a_topk", "a_last"):
                recon_logits = z @ z.transpose(-2, -1)
            elif self.recon_target == "delta_a_mse":
                d = z.size(-1)
                recon_logits = torch.tanh((z @ z.transpose(-2, -1)) / math.sqrt(d))

        if self.recon_target != "none":
            if self.fusion == "concat_residual":
                h_fused = self.fusion_module(h_T, z)
            elif self.fusion == "z_only":
                h_fused = z
            else:
                h_fused = h_T
        else:
            h_fused = h_T

        if self.readout == "attention":
            pooled = self.att_pool(h_fused)
        else:
            pooled = h_fused.mean(dim=-2)

        logits, aux_logits = self.cls_head(pooled)

        if is_single:
            logits = logits.unsqueeze(0)
            if aux_logits is not None:
                aux_logits = aux_logits.unsqueeze(0)

        return {
            "logits": logits,
            "aux_logits": aux_logits,
            "gate_scores": gate_scores,
            "mu": mu,
            "logvar": logvar,
            "recon_logits": recon_logits,
        }

    def get_trainable_params(self) -> list:
        return [p for p in self.parameters() if p.requires_grad]
