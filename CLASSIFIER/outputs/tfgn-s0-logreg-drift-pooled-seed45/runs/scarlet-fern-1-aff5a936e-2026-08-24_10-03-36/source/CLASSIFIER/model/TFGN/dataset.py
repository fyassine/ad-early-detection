"""TFGN/dataset.py — TFGNItem: per-subject derived quantities for the TFGN model.

Wraps a LongitudinalSubjectDataset dict item with stacked FC matrices,
log Δt, baseline graph topology, change-mask, strength centrality,
and drift anchor. All derived quantities are computed per subject.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch
from scipy.stats import rankdata


@dataclass
class TFGNItem:
    """All tensors a single subject contributes to the TFGN forward pass."""

    subject_id: str
    label: int
    n_visits: int
    # Stacked FC matrices, shape (T, N_rois, N_rois)
    X: torch.Tensor
    # log(1 + cumulative_months), shape (T,)
    log_dt: torch.Tensor
    # Baseline graph edge_index from graphs[0], shape (2, E)
    A0_edge_index: torch.Tensor
    # Baseline graph edge_attr, shape (E,) or None
    A0_edge_attr: Optional[torch.Tensor]
    # Change-mask M ∈ {0,1}^{N×N}, shape (N_rois, N_rois)
    change_mask: torch.Tensor
    # Strength centrality of |A_0| (row sums of abs baseline FC), shape (N_rois,)
    # Before z-scoring: finite and non-negative. After z-scoring (by adapter): can be negative.
    strength_centrality: torch.Tensor
    # Rank-sigmoid drift anchor d̃ ∈ (0,1)^N per node, shape (N_rois,)
    drift_anchor: torch.Tensor
    # Covariates
    age: float
    sex: int
    # Cohort tag — carried for probe, NOT fed to model
    cohort: str = "unknown"
    # Visit months (for truncation hooks)
    visit_months: list = field(default_factory=list)
    # Raw delta_t list (for compatibility)
    delta_t: list = field(default_factory=list)


def compute_change_mask(
    X: torch.Tensor,
    kappa: float = 0.10,
) -> torch.Tensor:
    """Binary change-mask M_ij = 1[|ΔA_ij| >= q_{1-κ}], quantile per subject.

    Parameters
    ----------
    X : (T, N, N) stacked FC matrices
    kappa : fraction of top changes to mark (default 0.10)

    Returns
    -------
    M : (N, N) binary mask in {0, 1}

    Raises
    ------
    ValueError
        If density < 0.01 or > 0.50 (for T >= 2) — the mask has degenerated.
    """
    if kappa <= 0.0 or kappa >= 1.0:
        raise ValueError(f"kappa must be in (0, 1), got {kappa}")

    T, N, _ = X.shape
    if T <= 1:
        # Single-visit sequence (e.g. baseline-only evaluation or per-visit N=1 prefix):
        # No follow-up delta exists, return an all-zeros mask.
        return torch.zeros((N, N), dtype=torch.float32)

    delta_A = X[-1] - X[0]  # A^(T) - A^(1)
    abs_delta = torch.abs(delta_A)
    # Quantile computed per subject (over all edges)
    threshold = torch.quantile(abs_delta, 1.0 - kappa)
    M = (abs_delta >= threshold).float()

    # Guard: density check
    density = M.mean().item()
    if density < 0.01 or density > 0.50:
        raise ValueError(
            f"Change-mask density {density:.4f} outside [0.01, 0.50]. "
            f"kappa={kappa}, threshold={threshold.item():.6f}. "
            "The mask has degenerated — consider using recon_target='delta_a_mse' instead."
        )
    return M


def compute_strength_centrality(X_baseline: torch.Tensor) -> torch.Tensor:
    """Strength centrality: row sums of |A_0| (abs of baseline FC).

    Returns a non-negative, finite vector of shape (N_rois,).
    This is BEFORE z-scoring (z-scoring with train-fold stats is the adapter's job).

    Raises
    ------
    ValueError
        If result contains non-finite values.
    """
    abs_A0 = torch.abs(X_baseline)
    # Zero the diagonal (self-connections)
    abs_A0 = abs_A0 - torch.diag(torch.diag(abs_A0))
    centrality = abs_A0.sum(dim=1)  # row sums

    if not torch.isfinite(centrality).all():
        raise ValueError(
            "Strength centrality contains non-finite values. "
            "Input baseline FC matrix may have NaN/Inf entries."
        )
    return centrality


def compute_drift_anchor(
    X: torch.Tensor,
    gate_rho: float = 0.15,
    tau_d: float = 0.05,
) -> torch.Tensor:
    """Within-subject rank-sigmoid drift anchor d̃ ∈ (0, 1)^N across nodes.

    d_i = ||x_i^(T) - x_i^(1)||_2  (Frobenius/L2 norm of node i's FC change profile)
    q_i = rank(d_i) / (N-1)         (within-subject quantile across the N nodes)
    d̃_i = σ((q_i - (1-ρ)) / τ_d)

    Parameters
    ----------
    X : (T, N, N) stacked FC matrices
    gate_rho : target gate sparsity (default: 0.15)
    tau_d : temperature for sharp sigmoid (default: 0.05)

    Returns
    -------
    d_tilde : (N,) float tensor in (0, 1)
    """
    T, N, _ = X.shape
    if T <= 1:
        return torch.zeros(N, dtype=torch.float32)

    delta_A = X[-1] - X[0]  # (N, N)
    d = torch.norm(delta_A, p=2, dim=-1)  # (N,) norm of each node's row change

    d_arr = d.cpu().numpy()
    ranks = rankdata(d_arr, method="average")  # 1-based ranks
    q = (ranks - 1.0) / max(N - 1, 1)  # [0, 1]

    # Sharp sigmoid centered at (1 - rho) quantile
    d_tilde = 1.0 / (1.0 + np.exp(-(q - (1.0 - gate_rho)) / tau_d))
    return torch.tensor(d_tilde, dtype=torch.float32)


def prepare_tfgn_item(
    item: Dict,
    *,
    kappa: float = 0.10,
    gate_rho: float = 0.15,
    tau_d: float = 0.05,
) -> TFGNItem:
    """Convert a LongitudinalSubjectDataset dict to a TFGNItem."""
    graphs = item["graphs"]
    T = len(graphs)

    # Stack FC matrices: (T, N, N)
    X = torch.stack([g.x for g in graphs], dim=0)  # g.x is (N, N) FC matrix

    # log Δt from cumulative months derived from delta_t:
    # delta_t is normalized inter-visit intervals (sum * 108 gives cumulative months)
    # Unit-harmonised across ADNI (days) and DELCODE (nominal months) via visit_identity
    delta_t = item.get("delta_t", [0.0] * T)
    cum_months = np.cumsum(delta_t) * 108.0
    log_dt = torch.tensor([math.log1p(m) for m in cum_months], dtype=torch.float32)

    # Baseline graph
    A0_edge_index = graphs[0].edge_index
    A0_edge_attr = graphs[0].edge_attr if hasattr(graphs[0], "edge_attr") else None

    # Change-mask (fails loud if invalid on multi-visit sequences)
    change_mask = compute_change_mask(X, kappa=kappa)

    # Strength centrality of |A_0|
    strength_centrality = compute_strength_centrality(X[0])

    # Per-node drift anchor vector d̃ ∈ (0, 1)^N
    drift_anchor = compute_drift_anchor(X, gate_rho=gate_rho, tau_d=tau_d)

    return TFGNItem(
        subject_id=item["subject_id"],
        label=item["label"],
        n_visits=T,
        X=X,
        log_dt=log_dt,
        A0_edge_index=A0_edge_index,
        A0_edge_attr=A0_edge_attr,
        change_mask=change_mask,
        strength_centrality=strength_centrality,
        drift_anchor=drift_anchor,
        age=item.get("age", 0.5),
        sex=item.get("sex", 0),
        cohort=item.get("cohort", "unknown"),
        visit_months=item.get("visit_months", []),
        delta_t=delta_t,
    )
