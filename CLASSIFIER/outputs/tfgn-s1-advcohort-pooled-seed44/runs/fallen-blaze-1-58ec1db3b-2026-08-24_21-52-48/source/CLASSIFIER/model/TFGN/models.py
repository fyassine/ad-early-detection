"""TFGN/models.py — Temporal-First Graph Network classifier.

Assembles the TFGN pipeline from switchable stages:
  NodeSharedLSTM → [TemporalSaliencyGate] → [GVAEEncoder → Fusion] → Readout → DualScoreReadout

Every stage is controlled by config flags so the ablation ladder is one model with knobs.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT = _ROOT.parent
for _p in (str(_REPO_ROOT), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from .layers import (  # noqa: E402
    AttentivePool,
    CohortAdversaryHead,
    ConcatResidualFusion,
    DualScoreReadout,
    GVAEEncoder,
    NodeSharedLSTM,
    TemporalSaliencyGate,
    grad_reverse,
)


class TFGNClassifier(nn.Module):
    """Temporal-First Graph Network for longitudinal conversion classification."""

    def __init__(
        self,
        *,
        n_rois: int = 200,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        lstm_dropout: float = 0.3,
        gvae_hidden: int = 128,
        gvae_latent: int = 64,
        gvae_heads: int = 2,
        gvae_dropout: float = 0.3,
        cond_dim: int = 2,
        use_time_delta: bool = True,
        use_gate: bool = True,
        recon_target: str = "delta_a_topk",
        fusion: str = "concat_residual",
        readout: str = "attention",
        dual_score: bool = True,
        cohort_conditioning: str = "none",
        cohort_adv_hidden: int = 32,
    ):
        super().__init__()

        valid_recon_targets = ("delta_a_topk", "delta_a_mse", "a_last", "none")
        if recon_target not in valid_recon_targets:
            raise ValueError(
                f"Unknown recon_target '{recon_target}'. Valid options: {valid_recon_targets}"
            )

        valid_fusions = ("concat_residual", "z_only")
        if fusion not in valid_fusions:
            raise ValueError(f"Unknown fusion '{fusion}'. Valid options: {valid_fusions}")

        valid_readouts = ("mean", "attention")
        if readout not in valid_readouts:
            raise ValueError(f"Unknown readout '{readout}'. Valid options: {valid_readouts}")

        valid_cohort_conditioning = ("none", "adversarial")
        if cohort_conditioning not in valid_cohort_conditioning:
            raise ValueError(
                f"Unknown cohort_conditioning '{cohort_conditioning}'. Valid options: "
                f"{valid_cohort_conditioning}. Note: 'film' is documented in "
                "DOCS/flipped/PLAN.md Phase 3 as a design option but was never "
                "implemented -- do not pass it expecting FiLM conditioning to happen."
            )

        self.n_rois = n_rois
        self.lstm_hidden = lstm_hidden
        self.use_time_delta = use_time_delta
        self.use_gate = use_gate
        self.recon_target = recon_target
        self.fusion = fusion
        self.readout = readout
        self.dual_score = dual_score
        self.cohort_conditioning = cohort_conditioning

        # 1. Node-shared LSTM: processes per-node temporal FC profiles (+ log Δt)
        self.lstm = NodeSharedLSTM(
            input_dim=n_rois,
            hidden_dim=lstm_hidden,
            num_layers=lstm_layers,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
            use_time_delta=use_time_delta,
        )

        # 2. Temporal Saliency Gate: learned suppression of static regions
        if self.use_gate:
            self.gate = TemporalSaliencyGate(hidden_dim=lstm_hidden)
        else:
            self.gate = None

        # 3. GVAE Encoder: propagates surviving dynamics over baseline topology A_0
        if self.recon_target != "none":
            self.gvae = GVAEEncoder(
                in_features=lstm_hidden,
                hidden_dim=gvae_hidden,
                latent_dim=gvae_latent,
                num_heads=gvae_heads,
                dropout=gvae_dropout,
                cond_dim=cond_dim,
            )
        else:
            self.gvae = None

        # 4. Fusion layer: combines temporal features h̃_T and graph latents z
        if self.recon_target != "none":
            if self.fusion == "concat_residual":
                self.fusion_module = ConcatResidualFusion(
                    h_dim=lstm_hidden,
                    z_dim=gvae_latent,
                    out_dim=lstm_hidden,
                )
                readout_dim = lstm_hidden
            else:
                self.fusion_module = None
                readout_dim = gvae_latent
        else:
            self.fusion_module = None
            readout_dim = lstm_hidden

        # 5. Readout pooling (Sparsemax AttentivePool or Mean pool)
        if self.readout == "attention":
            self.att_pool = AttentivePool(hidden_dim=readout_dim)
        else:
            self.att_pool = None

        # 6. Dual-score head (classification logit + topological saliency)
        self.cls_head = DualScoreReadout(
            hidden_dim=readout_dim,
            use_dual=self.dual_score,
        )

        # 7. Cohort-adversary head (gradient-reversal, ADNI vs. DELCODE) -- only
        # built when explicitly requested; adds no parameters or forward cost
        # to every arm that isn't testing this escalation.
        if self.cohort_conditioning == "adversarial":
            self.cohort_head = CohortAdversaryHead(
                hidden_dim=readout_dim, adv_hidden=cohort_adv_hidden
            )
        else:
            self.cohort_head = None

    def _encode_sequence(
        self,
        X: torch.Tensor,
        log_dt: Optional[torch.Tensor],
        A0_edge_index: torch.Tensor,
        A0_edge_attr: Optional[torch.Tensor],
        cond_vec: Optional[torch.Tensor],
    ) -> Dict[str, Optional[torch.Tensor]]:
        is_single = X.dim() == 3
        if is_single:
            X_expanded = X.unsqueeze(0)  # (1, T, N, N)
            log_dt_expanded = log_dt.unsqueeze(0) if log_dt is not None else None
        else:
            X_expanded = X
            log_dt_expanded = log_dt

        # NodeSharedLSTM over visits with time delta
        h = self.lstm(X_expanded, log_dt_expanded)  # (B, T, N, lstm_hidden)
        h_T = h[:, -1, :, :]  # (B, N, lstm_hidden)

        if is_single:
            h_T = h_T.squeeze(0)  # (N, lstm_hidden)

        # Temporal Saliency Gate
        gate_scores = None
        if self.use_gate:
            h_T, gate_scores = self.gate(h_T)

        # GVAE dynamics propagation over baseline topology A_0
        z, mu, logvar, mu_raw, recon_logits = None, None, None, None, None
        if self.recon_target != "none":
            z, mu, logvar, mu_raw = self.gvae(h_T, A0_edge_index, A0_edge_attr, cond_vec)

            if self.recon_target in ("delta_a_topk", "a_last"):
                recon_logits = z @ z.transpose(-2, -1)
            elif self.recon_target == "delta_a_mse":
                d = z.size(-1)
                recon_logits = torch.tanh((z @ z.transpose(-2, -1)) / math.sqrt(d))

        # Fusion: concat_residual or z_only
        if self.recon_target != "none":
            if self.fusion == "concat_residual":
                h_fused = self.fusion_module(h_T, z)
            elif self.fusion == "z_only":
                h_fused = z
            else:
                h_fused = h_T
        else:
            h_fused = h_T

        # Readout pooling over nodes
        if self.readout == "attention":
            h_pooled = self.att_pool(h_fused)
        else:
            h_pooled = h_fused.mean(dim=-2)

        return {
            "h_pooled": h_pooled,
            "h_fused": h_fused,
            "gate_scores": gate_scores,
            "mu": mu,
            "logvar": logvar,
            "mu_raw": mu_raw,
            "recon_logits": recon_logits,
            "is_single": torch.tensor(is_single),
        }

    def encode_patient(
        self,
        X: torch.Tensor,
        log_dt: Optional[torch.Tensor],
        A0_edge_index: torch.Tensor,
        A0_edge_attr: Optional[torch.Tensor] = None,
        cond_vec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode a subject sequence into a single patient embedding for the cohort probe."""
        enc = self._encode_sequence(X, log_dt, A0_edge_index, A0_edge_attr, cond_vec)
        return enc["h_pooled"]

    def forward(
        self,
        X: torch.Tensor,
        log_dt: Optional[torch.Tensor] = None,
        A0_edge_index: torch.Tensor = None,
        A0_edge_attr: Optional[torch.Tensor] = None,
        cond_vec: Optional[torch.Tensor] = None,
        cohort_adv_lambda: float = 1.0,
    ) -> Dict[str, Optional[torch.Tensor]]:
        enc = self._encode_sequence(X, log_dt, A0_edge_index, A0_edge_attr, cond_vec)
        h_pooled = enc["h_pooled"]
        h_fused = enc["h_fused"]
        is_single = bool(enc["is_single"].item())

        logits, s_topo = self.cls_head(h_pooled, h_fused)

        cohort_logit = None
        if self.cohort_head is not None:
            # Reversal happens on the backward pass only -- see _GradientReversal.
            # cohort_adv_lambda is the caller's per-epoch warmed-up value (mirrors
            # how beta_kl_eff is computed externally in train.py), not stored state.
            cohort_logit = self.cohort_head(grad_reverse(h_pooled, cohort_adv_lambda))

        if is_single:
            logits = logits.unsqueeze(0)
            if s_topo is not None:
                s_topo = s_topo.squeeze(0) if s_topo.dim() > 1 else s_topo
            if cohort_logit is not None:
                cohort_logit = cohort_logit.unsqueeze(0)

        return {
            "logits": logits,
            "s_topo": s_topo,
            "gate_scores": enc["gate_scores"],
            "mu": enc["mu"],
            "logvar": enc["logvar"],
            "mu_raw": enc["mu_raw"],
            "recon_logits": enc["recon_logits"],
            "cohort_logit": cohort_logit,
        }

    # ── Freeze / unfreeze / load helpers ────────────────────────────────────

    def freeze_node_lstm(self) -> None:
        for p in self.lstm.parameters():
            p.requires_grad_(False)

    def unfreeze_node_lstm(self) -> None:
        for p in self.lstm.parameters():
            p.requires_grad_(True)

    def load_node_lstm_weights(self, ckpt_path: str, device: str | torch.device = "cpu") -> None:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = (
            ckpt["model_state_dict"]
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt
            else ckpt
        )
        own_sd = self.lstm.state_dict()
        to_load = {k: v for k, v in sd.items() if k in own_sd and v.shape == own_sd[k].shape}
        own_sd.update(to_load)
        self.lstm.load_state_dict(own_sd)

    def freeze_gvae(self) -> None:
        if self.gvae is not None:
            for p in self.gvae.parameters():
                p.requires_grad_(False)

    def unfreeze_gvae(self) -> None:
        if self.gvae is not None:
            for p in self.gvae.parameters():
                p.requires_grad_(True)

    def load_gvae_weights(self, ckpt_path: str, device: str | torch.device = "cpu") -> None:
        if self.gvae is None:
            raise ValueError("Cannot load GVAE weights when recon_target='none' (no GVAE built).")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = (
            ckpt["model_state_dict"]
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt
            else ckpt
        )
        own_sd = self.gvae.state_dict()
        to_load = {k: v for k, v in sd.items() if k in own_sd and v.shape == own_sd[k].shape}
        own_sd.update(to_load)
        self.gvae.load_state_dict(own_sd)

    def get_trainable_params(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]
