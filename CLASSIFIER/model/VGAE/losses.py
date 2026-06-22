"""VGAE losses: masked adjacency BCE + KL divergence (+ optional feature recon).

The reconstruction term reuses the GAAE's masked adjacency BCE (so batched
block-diagonal padding is ignored identically); the variational term is the
standard KL divergence between the approximate posterior ``N(mu, exp(logvar))``
and the unit Gaussian prior, normalised per node.

Anti-collapse knobs (all no-ops at their defaults, so the prior objective is
unchanged):

  * ``free_bits`` — a per-latent-dimension KL floor (Kingma et al. 2016). Each
    dimension is allowed ``free_bits`` nats "for free"; only KL above that floor
    is penalised, so the optimiser cannot drive a dimension's KL to zero to game
    the β·KL term (posterior collapse).
  * ``feature_loss_weight`` + ``x_reconstructed`` — an optional node-feature
    reconstruction MSE. Forcing the latent to reconstruct node features (not just
    adjacency) gives the encoder a signal the prior cannot satisfy by collapsing.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..GAAE.losses import adjacency_reconstruction_loss


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor, free_bits: float = 0.0) -> torch.Tensor:
    """KL[ N(mu, σ²) || N(0, I) ], mean over nodes, summed over dims.

    With ``free_bits=0`` this is the mean per-node KL (the prior behaviour). With
    ``free_bits>0`` a per-dimension floor is applied: the batch-mean KL of each
    latent dimension is clamped up to ``free_bits`` before summing, so no single
    dimension can be optimised below that floor.
    """
    # Per-node, per-dim KL contribution: -0.5 * (1 + logvar - mu² - exp(logvar)).
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # [N, D]
    if free_bits > 0.0:
        kl_dim_mean = kl_per_dim.mean(dim=0)                      # [D] batch-mean per dim
        return torch.clamp(kl_dim_mean, min=free_bits).sum()
    return kl_per_dim.sum(dim=1).mean()


def vgae_total_loss(
    adj_original: torch.Tensor,
    adj_reconstructed: torch.Tensor,
    mask: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    *,
    free_bits: float = 0.0,
    x_original: torch.Tensor | None = None,
    x_reconstructed: torch.Tensor | None = None,
    feature_loss_weight: float = 0.0,
):
    """Combine masked adjacency BCE, (β-weighted) KL, and optional feature MSE.

    Returns ``(total, recon_loss, kl_loss, feat_loss)`` so callers can log the
    split, mirroring ``model.GAAE.losses.total_loss_fn``. ``feat_loss`` is a zero
    scalar when feature reconstruction is disabled.
    """
    recon_loss = adjacency_reconstruction_loss(adj_original, adj_reconstructed, mask)
    kl_loss = kl_divergence(mu, logvar, free_bits=free_bits)

    feat_loss = torch.zeros((), device=recon_loss.device)
    if feature_loss_weight > 0.0 and x_reconstructed is not None:
        if x_original is None:
            raise ValueError(
                "feature_loss_weight>0 requires x_original to compute the feature "
                "reconstruction MSE."
            )
        feat_loss = F.mse_loss(x_reconstructed, x_original)

    total = recon_loss + feature_loss_weight * feat_loss + beta * kl_loss
    return total, recon_loss, kl_loss, feat_loss
