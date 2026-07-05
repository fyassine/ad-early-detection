"""VGAE losses: masked adjacency BCE + KL divergence.

The reconstruction term reuses the GAAE's masked adjacency BCE (so batched
block-diagonal padding is ignored identically); the variational term is the
standard KL divergence between the approximate posterior ``N(mu, exp(logvar))``
and the unit Gaussian prior, normalised per node.
"""
from __future__ import annotations

import torch

from ..GAAE.losses import adjacency_reconstruction_loss


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Mean per-node KL[ N(mu, σ²) || N(0, I) ]."""
    # -0.5 * Σ_dims (1 + logvar - mu² - exp(logvar)), averaged over nodes.
    per_node = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return per_node.mean()


def vgae_total_loss(
    adj_original: torch.Tensor,
    adj_reconstructed: torch.Tensor,
    mask: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
):
    """Combine masked adjacency BCE and (β-weighted) KL.

    Returns ``(total, recon_loss, kl_loss)`` so callers can log the split, mirroring
    ``model.GAAE.losses.total_loss_fn``.
    """
    recon_loss = adjacency_reconstruction_loss(adj_original, adj_reconstructed, mask)
    kl_loss = kl_divergence(mu, logvar)
    total = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss
