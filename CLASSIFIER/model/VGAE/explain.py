"""model/VGAE/explain.py — forward-trace + adjacency-reconstruction helpers.

The VGAE reconstructs the *adjacency* (not the node features), so its
explainability mirrors ``model/GAAE/explain.py`` but scores how well each ROI's
graph neighbourhood is recovered by ``sigmoid(z zᵀ)`` rather than feature MSE.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
from torch_geometric.utils import to_dense_adj


def _edge_attr(data) -> Optional[torch.Tensor]:
    return getattr(data, "edge_attr", None)


@torch.no_grad()
def per_node_adjacency_error(model, data, *, device="cpu") -> np.ndarray:
    """Per-ROI adjacency reconstruction error (BCE between true row and ``sigmoid(z zᵀ)``)."""
    model.eval()
    x = data.x.to(device)
    ei = data.edge_index.to(device)
    ea = _edge_attr(data)
    ea = ea.to(device) if ea is not None else None
    z = model.encode(x, ei, ea)
    adj_hat = model.decode_all(z).clamp(1e-6, 1 - 1e-6)
    adj_true = to_dense_adj(ei, max_num_nodes=x.shape[0]).squeeze(0).to(device)
    bce = -(adj_true * torch.log(adj_hat) + (1 - adj_true) * torch.log(1 - adj_hat))
    return bce.mean(dim=1).detach().cpu().numpy()  # (N,)


@torch.no_grad()
def reconstruct_adjacency(model, data, *, device="cpu"):
    """Return ``(adj_true, adj_hat)`` numpy arrays for one graph.

    ``adj_hat = sigmoid(z zᵀ)``; pairs with ``model.GAAE.explain.reconstruction_quality``
    to score adjacency-reconstruction fidelity (Pearson r / R²) the same way the GAAE
    scores feature reconstruction.
    """
    model.eval()
    x = data.x.to(device)
    ei = data.edge_index.to(device)
    ea = _edge_attr(data)
    ea = ea.to(device) if ea is not None else None
    z = model.encode(x, ei, ea)
    adj_hat = model.decode_all(z)
    adj_true = to_dense_adj(ei, max_num_nodes=x.shape[0]).squeeze(0).to(device)
    return adj_true.detach().cpu().numpy(), adj_hat.detach().cpu().numpy()


@torch.no_grad()
def trace_forward(model, data, *, device="cpu") -> Dict[str, Any]:
    """Capture intermediates of one VGAE forward pass for the data-journey plots.

    Returns ``x`` (N, F), ``latent`` = ``mu`` (N, latent), ``pooled`` (latent,),
    ``adj_true`` (N, N), ``adj_recon`` (N, N), and an ordered ``stages`` list.
    Deliberately omits the GAAE-only ``decoder_gat*`` / feature-``recon`` keys —
    the EXPLAIN notebook's VGAE branch compares ``adj_true`` against ``adj_recon``
    instead of feature reconstruction. Identical key set for both ``conv_type``
    variants (``gcn`` / ``gat``); only the encoder internals differ.
    """
    model.eval()
    x = data.x.to(device)
    ei = data.edge_index.to(device)
    ea = _edge_attr(data)
    ea = ea.to(device) if ea is not None else None

    mu = model.encode(x, ei, ea)
    pooled = mu.mean(dim=0)
    adj_recon = model.decode_all(mu)
    adj_true = to_dense_adj(ei, max_num_nodes=x.shape[0]).squeeze(0).to(device)

    def _np(t):
        return t.detach().cpu().numpy()

    out = {
        "x": _np(x),
        "latent": _np(mu),
        "pooled": _np(pooled),
        "adj_true": _np(adj_true),
        "adj_recon": _np(adj_recon),
    }
    out["stages"] = [
        ("input x (FC rows)", out["x"].shape),
        (f"latent mu ({model.conv_type})", out["latent"].shape),
        ("pooled graph embedding", out["pooled"].shape),
        ("true adjacency (kNN graph)", out["adj_true"].shape),
        ("reconstructed adjacency (sigmoid z zᵀ)", out["adj_recon"].shape),
    ]
    return out


__all__ = ["per_node_adjacency_error", "trace_forward"]
