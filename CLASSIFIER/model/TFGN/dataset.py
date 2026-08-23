"""TFGN/dataset.py — TFGNItem: per-subject derived quantities for the TFGN model.

Wraps a LongitudinalSubjectDataset dict item with stacked FC matrices,
log Δt, baseline graph topology, change-mask, strength centrality,
and drift anchor. All derived quantities are computed once per subject
and cached.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch


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
    # Rank-sigmoid drift anchor d̃ ∈ (0,1), scalar
    drift_anchor: float
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
        If density < 0.01 or > 0.50 — the mask has degenerated.
    """
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


def compute_drift_anchors(
    X_list: List[torch.Tensor],
    gate_rho: float = 0.15,
    tau_d: float = 0.05,
) -> List[float]:
    """Rank-sigmoid drift anchors for a batch of subjects.

    d_i = ||x_i^(T) - x_i^(1)||_2  (Frobenius norm of FC change)
    q_i = rank(d_i) / (N-1)
    d̃_i = σ((q_i - (1-ρ)) / τ_d)

    Must be called on the full training set to compute ranks.
    For a single subject, returns 0.5 (sigmoid of 0).
    """
    N = len(X_list)
    if N <= 1:
        return [0.5] * N  # sigmoid(0) for single subject

    # Compute drift magnitudes
    drifts = []
    for X in X_list:
        d = torch.norm(X[-1] - X[0], p='fro').item()  # ||A^(T) - A^(1)||_F
        drifts.append(d)

    drifts_arr = np.array(drifts)
    # Rank: scipy-style average ranking, then normalize
    from scipy.stats import rankdata
    ranks = rankdata(drifts_arr, method='average')  # 1-based
    q = (ranks - 1) / (N - 1)  # 0-based, in [0, 1]

    # Sharp sigmoid centered at (1-rho) quantile
    d_tilde = 1.0 / (1.0 + np.exp(-(q - (1 - gate_rho)) / tau_d))
    return d_tilde.tolist()


def prepare_tfgn_item(
    item: Dict,
    *,
    kappa: float = 0.10,
    drift_anchor: float = 0.5,  # pre-computed by compute_drift_anchors
) -> TFGNItem:
    """Convert a LongitudinalSubjectDataset dict to a TFGNItem.

    The drift_anchor should be pre-computed by compute_drift_anchors on the
    full training set (it requires cross-subject ranking).
    """
    graphs = item["graphs"]
    T = len(graphs)

    # Stack FC matrices: (T, N, N)
    X = torch.stack([g.x for g in graphs], dim=0)  # g.x is (N, N) FC matrix

    # log Δt from cumulative months in delta_t
    # delta_t from LongitudinalSubjectDataset is normalized inter-visit intervals
    # We need cumulative months — reconstruct from visit_months
    visit_months = item.get('visit_months', [])
    if visit_months:
        cum_months = [float(m) for m in visit_months]
    else:
        # Fallback to delta_t (already cumulative-ish)
        cum_months = [0.0] * T
    log_dt = torch.tensor([math.log1p(m) for m in cum_months], dtype=torch.float32)

    # Baseline graph
    A0_edge_index = graphs[0].edge_index
    A0_edge_attr = graphs[0].edge_attr if hasattr(graphs[0], 'edge_attr') else None

    # Change-mask
    change_mask = compute_change_mask(X, kappa=kappa)

    # Strength centrality of |A_0|
    strength_centrality = compute_strength_centrality(X[0])

    return TFGNItem(
        subject_id=item['subject_id'],
        label=item['label'],
        n_visits=T,
        X=X,
        log_dt=log_dt,
        A0_edge_index=A0_edge_index,
        A0_edge_attr=A0_edge_attr,
        change_mask=change_mask,
        strength_centrality=strength_centrality,
        drift_anchor=drift_anchor,
        age=item.get('age', 0.5),
        sex=item.get('sex', 0),
        cohort=item.get('cohort', 'unknown'),
        visit_months=visit_months,
        delta_t=item.get('delta_t', []),
    )
