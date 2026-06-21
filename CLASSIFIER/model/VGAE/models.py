"""VGAE model: Variational Graph Autoencoder with an InnerProduct decoder.

Mirrors the GAAE's public encode/decode contract so it is a drop-in feature
extractor for the downstream classifiers (``adapters/gep.py`` and friends call
``enc.encode(x, edge_index, edge_attr).mean(0)``), but:

  * the bottleneck is **variational** — two parallel conv heads emit ``mu`` and
    ``logvar``; ``encode`` returns ``mu`` (the deterministic latent used for
    inference / pooling), training uses the reparameterised ``z``;
  * the GAT *feature* decoder is dropped — reconstruction is **adjacency only**
    via ``torch_geometric.nn.InnerProductDecoder`` (``sigmoid(z zᵀ)``);
  * ``conv_type`` selects the encoder backbone: ``"gcn"`` (canonical VGAE) or
    ``"gat"`` (attention, ``return_attention`` supported for region maps).

No FiLM conditioning (the GAAE's age/sex modulation) — kept deliberately simple,
matching the textbook VGAE.
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, GATv2Conv, GCNConv, InnerProductDecoder


class VariationalGraphAutoencoder(nn.Module):
    """Variational graph autoencoder with an inner-product (adjacency) decoder.

    Parameters
    ----------
    in_features : int   Input node-feature dimension (number of ROIs).
    hidden_dim  : int   Hidden dim of the shared encoder layer.
    latent_dim  : int   Latent embedding dimension (the pooled-embedding width).
    conv_type   : str   ``"gcn"`` or ``"gat"`` encoder backbone.
    num_heads   : int   GAT attention heads (ignored for ``conv_type="gcn"``).
    dropout     : float Dropout probability.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        latent_dim: int,
        conv_type: str = "gcn",
        num_heads: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        conv_type = str(conv_type).lower()
        if conv_type not in ("gcn", "gat"):
            raise ValueError(f"conv_type must be 'gcn' or 'gat', got {conv_type!r}.")
        self.conv_type = conv_type
        self.dropout = dropout
        self.edge_dim = 1
        self.latent_dim = latent_dim
        self.adj_decoder = InnerProductDecoder()

        if conv_type == "gcn":
            self.shared = GCNConv(in_features, hidden_dim)
            self.bn = BatchNorm(hidden_dim)
            self.conv_mu = GCNConv(hidden_dim, latent_dim)
            self.conv_logvar = GCNConv(hidden_dim, latent_dim)
        else:  # gat
            self.shared = GATv2Conv(
                in_features, hidden_dim, heads=num_heads, concat=True,
                edge_dim=self.edge_dim, residual=True,
            )
            self.bn = BatchNorm(hidden_dim * num_heads)
            self.conv_mu = GATv2Conv(
                hidden_dim * num_heads, latent_dim, heads=num_heads, concat=False,
                edge_dim=self.edge_dim, residual=True,
            )
            self.conv_logvar = GATv2Conv(
                hidden_dim * num_heads, latent_dim, heads=num_heads, concat=False,
                edge_dim=self.edge_dim, residual=True,
            )

    @staticmethod
    def _normalize_edge_attr(edge_attr):
        if edge_attr is None:
            return None
        if edge_attr.dim() == 1:
            return edge_attr.unsqueeze(-1)
        return edge_attr

    def _shared(self, x, edge_index, edge_attr, return_attention=False):
        """Run the shared encoder layer; returns (h, attn_list)."""
        attn: List = []
        if self.conv_type == "gcn":
            ew = edge_attr.view(-1) if edge_attr is not None else None
            h = self.shared(x, edge_index, edge_weight=ew)
        else:
            ea = self._normalize_edge_attr(edge_attr)
            if return_attention:
                h, a = self.shared(x, edge_index, edge_attr=ea, return_attention_weights=True)
                attn.append(a)
            else:
                h = self.shared(x, edge_index, edge_attr=ea)
        h = self.bn(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return h, attn

    def _head(self, conv, h, edge_index, edge_attr, return_attention=False):
        if self.conv_type == "gcn":
            ew = edge_attr.view(-1) if edge_attr is not None else None
            return conv(h, edge_index, edge_weight=ew), None
        ea = self._normalize_edge_attr(edge_attr)
        if return_attention:
            out, a = conv(h, edge_index, edge_attr=ea, return_attention_weights=True)
            return out, a
        return conv(h, edge_index, edge_attr=ea), None

    def encode_dist(self, x, edge_index, edge_attr=None, return_attention=False):
        """Return ``(mu, logvar)`` (and an attention list when requested)."""
        h, attn = self._shared(x, edge_index, edge_attr, return_attention=return_attention)
        mu, a_mu = self._head(self.conv_mu, h, edge_index, edge_attr, return_attention)
        logvar, _ = self._head(self.conv_logvar, h, edge_index, edge_attr, return_attention=False)
        if return_attention:
            if a_mu is not None:
                attn.append(a_mu)
            return mu, logvar, attn
        return mu, logvar

    def encode(
        self, x, edge_index, edge_attr=None, return_attention=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List]]:
        """Deterministic latent ``mu`` (drop-in for the GAAE pooling path).

        Returns ``mu`` (``[N, latent_dim]``); with ``return_attention=True`` returns
        ``(mu, attention_weights)`` so ``region_importance`` works for the GAT variant
        (empty list for GCN, which has no attention).
        """
        if return_attention:
            mu, _logvar, attn = self.encode_dist(x, edge_index, edge_attr, return_attention=True)
            return mu, attn
        mu, _logvar = self.encode_dist(x, edge_index, edge_attr)
        return mu

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_adjacency(self, z, edge_index):
        """Edge-wise reconstruction probabilities for ``edge_index`` (sigmoid)."""
        return self.adj_decoder(z, edge_index)

    def decode_all(self, z):
        """Dense reconstructed adjacency ``sigmoid(z zᵀ)`` (``[N, N]``)."""
        return self.adj_decoder.forward_all(z)

    def forward(self, x, edge_index, edge_attr=None):
        """Return ``(z, mu, logvar, adj_reconstructed_dense)``.

        Training samples ``z`` via reparameterisation; ``eval`` collapses to ``mu``
        (``randn_like`` is a no-op variance source we skip when not training).
        """
        mu, logvar = self.encode_dist(x, edge_index, edge_attr)
        z = self.reparameterize(mu, logvar) if self.training else mu
        adj_reconstructed = self.decode_all(z)
        return z, mu, logvar, adj_reconstructed
