"""Loss functions for TFGN."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from model.VGAE.losses import kl_divergence  # noqa: E402


def gate_sparsity_kl(s: torch.Tensor, rho: float = 0.15) -> torch.Tensor:
    """KL divergence between mean gate activation and target sparsity.

    Args:
        s: gate scores, any shape
        rho: target sparsity (default: 0.15)

    Returns:
        Scalar KL divergence.
    """
    if not isinstance(s, torch.Tensor):
        raise ValueError("s must be a torch.Tensor")
    eps = 1e-7
    s_bar = s.mean().clamp(min=eps, max=1.0 - eps)
    # KL(s_bar || rho) = s_bar * log(s_bar / rho) + (1 - s_bar) * log((1 - s_bar) / (1 - rho))
    return s_bar * torch.log(s_bar / rho) + (1.0 - s_bar) * torch.log((1.0 - s_bar) / (1.0 - rho))


def drift_anchor_mse(s: torch.Tensor, d_tilde: torch.Tensor) -> torch.Tensor:
    """MSE between gate scores and drift anchor targets.

    Args:
        s: gate scores (N,) or (N, 1) or (B, N)
        d_tilde: drift anchor targets, matching shape

    Returns:
        Scalar MSE.
    """
    if not isinstance(s, torch.Tensor) or not isinstance(d_tilde, torch.Tensor):
        raise ValueError("Inputs must be torch.Tensors")
    if s.dim() > d_tilde.dim() and s.size(-1) == 1:
        s = s.squeeze(-1)
    if s.shape != d_tilde.shape:
        raise ValueError(f"Shape mismatch: {s.shape} vs {d_tilde.shape}")
    return F.mse_loss(s, d_tilde)


def centrality_anchor_mse(s_topo: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """MSE between topological attention and centrality.

    Args:
        s_topo: attention/saliency scores for topology (N,) or (N, 1)
        c: z-scored strength centrality (N,)

    Returns:
        Scalar MSE.
    """
    if not isinstance(s_topo, torch.Tensor) or not isinstance(c, torch.Tensor):
        raise ValueError("Inputs must be torch.Tensors")
    if s_topo.dim() > c.dim() and s_topo.size(-1) == 1:
        s_topo = s_topo.squeeze(-1)
    if s_topo.shape != c.shape:
        raise ValueError(f"Shape mismatch: {s_topo.shape} vs {c.shape}")
    return F.mse_loss(s_topo, c)


def change_mask_bce(logits: torch.Tensor, M: torch.Tensor, pos_weight: float) -> torch.Tensor:
    """BCE with logits for change-mask prediction.

    Args:
        logits: reconstructed adjacency logits (N, N)
        M: binary change-mask target (N, N) in {0, 1}
        pos_weight: weight for positive examples

    Returns:
        Scalar BCEWithLogitsLoss.
    """
    if not isinstance(logits, torch.Tensor) or not isinstance(M, torch.Tensor):
        raise ValueError("Inputs must be torch.Tensors")
    if logits.shape != M.shape:
        raise ValueError(f"Shape mismatch: {logits.shape} vs {M.shape}")
    return F.binary_cross_entropy_with_logits(
        logits, M.float(), pos_weight=torch.tensor([pos_weight], device=logits.device)
    )


def delta_a_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE loss for ΔA regression.

    Args:
        pred: tanh(ZZ^T/sqrt(d)), shape (N, N)
        target: ΔA/2 in [-1, 1], shape (N, N)

    Returns:
        Scalar MSE.
    """
    if not isinstance(pred, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise ValueError("Inputs must be torch.Tensors")
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {target.shape}")
    return F.mse_loss(pred, target)


def free_bits_kl(mu: torch.Tensor, logvar: torch.Tensor, free_bits: float = 0.5) -> torch.Tensor:
    """Wraps model.VGAE.losses.kl_divergence.

    Args:
        mu: latent mean
        logvar: latent log variance
        free_bits: KL floor

    Returns:
        Scalar KL divergence.
    """
    if not isinstance(mu, torch.Tensor) or not isinstance(logvar, torch.Tensor):
        raise ValueError("Inputs must be torch.Tensors")
    return kl_divergence(mu, logvar, free_bits=free_bits)


def cohort_adversarial_bce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """BCE for the cohort-adversary head.

    Ordinary (unweighted) BCE forward pass -- the adversarial effect comes
    entirely from the gradient-reversal layer applied upstream of this head's
    input, not from anything in this loss. Two cohorts, so ``labels`` is a
    0/1 float tensor (see ``model.TFGN.layers.CohortAdversaryHead``).

    Args:
        logits: cohort-classification logit(s), any shape
        labels: binary cohort labels, matching shape

    Returns:
        Scalar BCEWithLogitsLoss.
    """
    if not isinstance(logits, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise ValueError("Inputs must be torch.Tensors")
    if logits.shape != labels.shape:
        raise ValueError(f"Shape mismatch: {logits.shape} vs {labels.shape}")
    return F.binary_cross_entropy_with_logits(logits, labels.float())

