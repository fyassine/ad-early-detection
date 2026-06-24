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

Optional FiLM age/sex conditioning of ``mu`` (mirrors the GAAE's modulation) —
pass ``cond_vec``/``batch_mask`` to ``encode``/``encode_dist``/``forward`` to
enable it; omitting them leaves the plain textbook VGAE behavior unchanged.
"""
from __future__ import annotations

from typing import List, Tuple, Union

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
    feature_decoder : bool  When ``True``, add a node-feature decoder MLP
        (``latent_dim -> hidden_dim -> in_features``) so the latent is trained to
        reconstruct node features as well as adjacency. This is an anti-collapse
        signal the unit-Gaussian prior cannot satisfy by collapsing; it is
        consumed by ``losses.vgae_total_loss(feature_loss_weight=...)``. Off by
        default — the encode/pooling contract is unchanged either way.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        latent_dim: int,
        conv_type: str = "gcn",
        num_heads: int = 2,
        dropout: float = 0.3,
        feature_decoder: bool = False,
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

        # Optional node-feature decoder (anti-collapse). A plain MLP keeps it
        # backbone-agnostic (no edge_index needed) and node-wise, mirroring the
        # GAAE's feature-reconstruction role without its 3-layer GAT decoder.
        self.has_feature_decoder = bool(feature_decoder)
        if self.has_feature_decoder:
            self.feat_decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, in_features),
            )

        # FiLM age/sex conditioning of the latent (mirrors GAAE): two small MLPs map the
        # 2-d (age, sex) covariate to per-dimension scale/shift applied to ``mu``.
        self.cond_dim = 2
        self.film_gamma = nn.Sequential(
            nn.Linear(self.cond_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, latent_dim)
        )
        self.film_beta = nn.Sequential(
            nn.Linear(self.cond_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, latent_dim)
        )

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

    def encode_dist(self, x, edge_index, edge_attr=None, return_attention=False, cond_vec=None, batch_mask=None):
        h, attn = self._shared(x, edge_index, edge_attr, return_attention=return_attention)
        mu, a_mu = self._head(self.conv_mu, h, edge_index, edge_attr, return_attention)
        logvar, _ = self._head(self.conv_logvar, h, edge_index, edge_attr, return_attention=False)

        mu_raw = mu  # type: ignore[misc]
        if cond_vec is not None and batch_mask is not None:
            mu = self.condition_latent(mu, cond_vec, batch_mask)

        if return_attention:
            if a_mu is not None:
                attn.append(a_mu)
            return mu, logvar, attn, mu_raw   

        return mu, logvar, mu_raw 

    def encode(self, x, edge_index, edge_attr=None, return_attention=False, cond_vec=None, batch_mask=None):
        if return_attention:
            mu, _logvar, attn, _mu_raw = self.encode_dist(   # type: ignore[misc]
                x, edge_index, edge_attr, return_attention=True,
                cond_vec=cond_vec, batch_mask=batch_mask,
            )
            return mu, attn
        mu, _logvar, _mu_raw = self.encode_dist(              # type: ignore[misc]
            x, edge_index, edge_attr, cond_vec=cond_vec, batch_mask=batch_mask,
        )
        return mu

    def condition_latent(self, z, cond_vec, batch_mask):
        """FiLM-modulate ``z`` with per-graph ``(age, sex)`` scale/shift (mirrors GAAE)."""
        gamma = self.film_gamma(cond_vec)  # [batch_size, latent_dim]
        beta = self.film_beta(cond_vec)    # [batch_size, latent_dim]
        gamma_per_node = gamma[batch_mask]  # [num_nodes, latent_dim]
        beta_per_node = beta[batch_mask]    # [num_nodes, latent_dim]
        return gamma_per_node * z + beta_per_node

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

    def decode_features(self, z):
        """Reconstruct node features from the latent (``[N, in_features]``).

        Returns ``None`` when the model was built without a feature decoder.
        """
        if not self.has_feature_decoder:
            return None
        return self.feat_decoder(z)

    def forward(self, x, edge_index, edge_attr=None, cond_vec=None, batch_mask=None):
        mu, logvar, mu_raw = self.encode_dist(
            x, edge_index, edge_attr, cond_vec=cond_vec, batch_mask=batch_mask
        )
        z = self.reparameterize(mu, logvar)   # decoding uses FiLM-conditioned mu
        adj_reconstructed = self.decode_all(z)
        x_reconstructed = self.decode_features(z)
        return z, mu_raw, logvar, adj_reconstructed, x_reconstructed
