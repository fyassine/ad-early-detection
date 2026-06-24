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
from ..GAAE.utils import calculate_dense_adjacency


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor, free_bits: float = 0.0) -> torch.Tensor:
    """KL[ N(mu, σ²) || N(0, I) ], mean over nodes, summed over dims.

    With ``free_bits=0`` this is the mean per-node KL (the prior behaviour). With
    ``free_bits>0`` the per-dim loss is
    ``clamp(kl, min=free_bits) + clamp(free_bits - kl, min=0)**2``:

      * At/above the floor this is just ``kl`` (d/dkl = +1) — identical to the
        plain clamp, so using more than ``free_bits`` nats is exactly as costly
        as the standard, un-floored KL term (no extra discouragement).
      * Below the floor, the ``clamp(kl, min=free_bits)`` piece is constant
        (d/dkl = 0, same as the plain clamp), but the added squared-shortfall
        term has d/dkl = ``-2*(free_bits - kl) < 0`` — i.e. minimising the total
        loss actively pushes a collapsed dimension's KL **up** toward the floor,
        with more force the further below it sits.

    A plain ``torch.clamp(kl, min=free_bits)`` (no shortfall term) gives exactly
    zero gradient for any dimension already at/below the floor, so it can only
    stop the optimiser from squeezing KL down further than whatever
    floor-evading value it would otherwise reach — it cannot recruit a
    dimension that's already collapsed. ``raw_kl_per_dim`` starts near 0 at
    initialisation (mu≈0, logvar≈0), below every free-bits value worth
    sweeping, so every dimension starts inside that zero-gradient region
    simultaneously: this is the collapse observed across the whole
    ``tune_vgae_anticollapse.py`` search regardless of the swept value.

    A naive "soft" relaxation like ``free_bits + softplus(kl - free_bits)`` is
    NOT a fix either — its gradient w.r.t. ``kl`` stays *positive* below the
    floor (smaller than above it, but never negative), so it still pushes KL
    *down* there, just with less force than the unclamped term would. That
    makes collapse worse, not better (verified empirically: training with it
    converged to a *more* collapsed posterior, 100% of dims at the floor, than
    the plain hard clamp). The shortfall-penalty above is the one shape whose
    sign actually points the right way below the floor.
    """
    kl_dim_mean = raw_kl_per_dim(mu, logvar)
    if free_bits > 0.0:
        floored = torch.clamp(kl_dim_mean, min=free_bits)
        shortfall = torch.clamp(free_bits - kl_dim_mean, min=0.0)
        return (floored + shortfall.pow(2)).sum()
    return kl_dim_mean.sum()


def raw_kl_per_dim(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Unclamped batch-mean KL per latent dimension, shape ``[D]``.

    This is ``kl_divergence``'s ``free_bits=0`` computation exposed standalone so
    callers can see the pre-clamp per-dimension KL even when free-bits is active —
    ``torch.clamp(x, min=floor)`` zeroes the gradient for any dimension already at
    or below the floor, so without this the only visible number once free-bits is
    pinning the loss is the floor itself, not how far under it the encoder fell.
    """
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # [N, D]
    return kl_per_dim.mean(dim=0)  # [D] batch-mean per dim


def raw_kl_stats(mu: torch.Tensor, logvar: torch.Tensor, free_bits: float = 0.0) -> dict[str, float]:
    """Diagnostic summary of the unclamped per-dim KL: min/mean/max plus the
    fraction of dimensions sitting at or below ``free_bits`` (always 0.0 when
    ``free_bits<=0``, since there's no floor to sit at)."""
    kl_dim_mean = raw_kl_per_dim(mu, logvar)
    stats = {
        "raw_kl_min": kl_dim_mean.min().item(),
        "raw_kl_mean": kl_dim_mean.mean().item(),
        "raw_kl_max": kl_dim_mean.max().item(),
        "frac_dims_at_floor": 0.0,
    }
    if free_bits > 0.0:
        stats["frac_dims_at_floor"] = (kl_dim_mean <= free_bits).float().mean().item()
    return stats


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


def compute_sample_reconstruction_error(
    data,
    model,
    device,
    beta: float,
    *,
    free_bits: float = 0.0,
    feature_loss_weight: float = 0.0,
) -> tuple[float, float, float, float]:
    """
    Run one graph through a VGAE and return (recon_err, kl_err, feat_err, total_err).

    Mirrors ``model.GAAE.losses.compute_sample_reconstruction_error``'s call shape
    and no-grad/eval contract. Caller is responsible for setting model.eval() before
    a sweep (forward() then returns the deterministic ``z = mu`` rather than a
    reparameterised sample).
    """
    data = data.to(device)
    x, edge_index = data.x, data.edge_index
    edge_attr = getattr(data, "edge_attr", None)
    age = float(data.patient_age.item()) if torch.is_tensor(data.patient_age) else float(data.patient_age)
    sex = float(data.patient_sex.item()) if torch.is_tensor(data.patient_sex) else float(data.patient_sex)
    cond_vec = torch.tensor([[age, sex]], dtype=torch.float32, device=device)
    batch_mask = torch.zeros(x.size(0), dtype=torch.long, device=device)

    with torch.no_grad():
        _z, mu, logvar, adj_reconstructed, x_reconstructed = model(
            x, edge_index, edge_attr, cond_vec=cond_vec, batch_mask=batch_mask
        )
        adj_true = calculate_dense_adjacency(data).to(device)
        mask = torch.ones_like(adj_true, dtype=torch.bool)
        recon_error = adjacency_reconstruction_loss(adj_true, adj_reconstructed, mask).item()
        kl_error = kl_divergence(mu, logvar, free_bits=free_bits).item()

        feat_error = 0.0
        if feature_loss_weight > 0.0 and x_reconstructed is not None:
            feat_error = F.mse_loss(x_reconstructed, x).item()

    total_error = recon_error + feature_loss_weight * feat_error + beta * kl_error
    return float(recon_error), float(kl_error), float(feat_error), float(total_error)
