"""
TFGN/layers.py — Neural network layers and modules for the TFGN model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, GATv2Conv


def sparsemax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Exact sparsemax activation function (Martins & Astudillo 2016).
    Produces a sparse probability distribution with exact zeros.

    Parameters
    ----------
    z : torch.Tensor
        Input 1D or ND tensor.
    dim : int
        Dimension along which to apply sparsemax.

    Returns
    -------
    torch.Tensor
        Sparse probability distribution.
    """
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    z_cumsum = torch.cumsum(z_sorted, dim=dim)

    k_range = torch.arange(1, z.size(dim) + 1, device=z.device, dtype=z.dtype)
    shape = [1] * z.dim()
    shape[dim] = z.size(dim)
    k_range = k_range.view(shape)

    bound = 1 + k_range * z_sorted > z_cumsum
    k = (bound * k_range).max(dim=dim, keepdim=True).values

    z_cumsum_k = torch.gather(z_cumsum, dim, (k - 1).long())
    tau = (z_cumsum_k - 1) / k

    return torch.clamp(z - tau, min=0.0)


class NodeSharedLSTM(nn.Module):
    """
    Standard LSTM applied node-wise (shared weights across nodes).
    The same LSTM processes each node's time series independently.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, T, N, d_in).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, T, N, hidden_dim).
        """
        B, T, N, d_in = x.shape
        x_reshaped = x.permute(0, 2, 1, 3).reshape(B * N, T, d_in)
        out, _ = self.lstm(x_reshaped)
        out = out.view(B, N, T, self.hidden_dim).permute(0, 2, 1, 3)
        return out


class TemporalSaliencyGate(nn.Module):
    """
    Temporal Saliency Gate. Computes a gating scalar per node using a projection
    and LeakyReLU, followed by a sigmoid to bound values in (0, 1), and then
    scales the original features via a residual connection (1 + s_i) * h_i.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.W_s = nn.Linear(hidden_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        h : torch.Tensor
            Hidden state of shape (..., hidden_dim).

        Returns
        -------
        gated_h : torch.Tensor
            Gated hidden state of shape (..., hidden_dim).
        gate_scores : torch.Tensor
            Raw saliency scores (s_i) in (0, 1) of shape (..., 1).
        """
        h_proj = F.leaky_relu(self.W_s(h))
        gate_scores = torch.sigmoid(self.w(h_proj))
        gated_h = (1 + gate_scores) * h
        return gated_h, gate_scores


class GVAEEncoder(nn.Module):
    """
    GVAE Encoder with GATv2 μ/logσ² heads + FiLM conditioning on μ.
    Functions as a standalone encoder returning the variational latent z.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        latent_dim: int,
        num_heads: int = 2,
        dropout: float = 0.3,
        cond_dim: int = 2
    ) -> None:
        super().__init__()
        self.dropout = dropout

        self.shared = GATv2Conv(
            in_features,
            hidden_dim,
            heads=num_heads,
            concat=True,
            edge_dim=1,
            residual=True
        )
        self.bn = BatchNorm(hidden_dim * num_heads)

        self.conv_mu = GATv2Conv(
            hidden_dim * num_heads,
            latent_dim,
            heads=num_heads,
            concat=False,
            edge_dim=1,
            residual=True
        )
        self.conv_logvar = GATv2Conv(
            hidden_dim * num_heads,
            latent_dim,
            heads=num_heads,
            concat=False,
            edge_dim=1,
            residual=True
        )

        self.film_gamma = nn.Sequential(
            nn.Linear(cond_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, latent_dim)
        )
        self.film_beta = nn.Sequential(
            nn.Linear(cond_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, latent_dim)
        )

    @staticmethod
    def _normalize_edge_attr(edge_attr: torch.Tensor | None) -> torch.Tensor | None:
        if edge_attr is None:
            return None
        if edge_attr.dim() == 1:
            return edge_attr.unsqueeze(-1)
        return edge_attr

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample from the variational distribution if training, else return mu."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        cond_vec: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Node features.
        edge_index : torch.Tensor
            Edge indices.
        edge_attr : torch.Tensor, optional
            Edge weights/attributes.
        cond_vec : torch.Tensor, optional
            FiLM conditioning vector (e.g., age, sex).

        Returns
        -------
        z : torch.Tensor
            Sampled latent during training, or mu during evaluation.
        mu : torch.Tensor
            Mean of the latent distribution.
        logvar : torch.Tensor
            Log variance of the latent distribution.
        """
        ea = self._normalize_edge_attr(edge_attr)

        h = self.shared(x, edge_index, edge_attr=ea)
        h = self.bn(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        mu = self.conv_mu(h, edge_index, edge_attr=ea)
        logvar = self.conv_logvar(h, edge_index, edge_attr=ea)

        if cond_vec is not None:
            gamma = self.film_gamma(cond_vec)
            beta = self.film_beta(cond_vec)
            mu = gamma * mu + beta

        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class ConcatResidualFusion(nn.Module):
    """
    Concatenates hidden state and latent vectors, projects, and applies LayerNorm.
    """

    def __init__(self, h_dim: int, z_dim: int, out_dim: int | None = None) -> None:
        super().__init__()
        out_dim = out_dim if out_dim is not None else h_dim
        self.W_u = nn.Linear(h_dim + z_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        h : torch.Tensor
        z : torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        out = torch.cat([h, z], dim=-1)
        out = self.W_u(out)
        out = self.norm(out)
        return out


class AttentivePool(nn.Module):
    """
    Attention-weighted pooling over nodes, using exact sparsemax to generate
    sparse attention distributions.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attn_proj = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        h : torch.Tensor
            Input of shape (N, D) or (..., N, D)

        Returns
        -------
        torch.Tensor
            Weighted sum over nodes of shape (D,) or (..., D).
        """
        attn_scores = self.attn_proj(h).squeeze(-1)
        weights = sparsemax(attn_scores, dim=-1)
        out = (weights.unsqueeze(-1) * h).sum(dim=-2)
        return out


class DualScoreReadout(nn.Module):
    """
    Produces both a classification logit and an auxiliary reconstruction score.
    """

    def __init__(self, hidden_dim: int, use_dual: bool = True) -> None:
        super().__init__()
        self.use_dual = use_dual
        self.cls_head = nn.Linear(hidden_dim, 1)
        if use_dual:
            self.aux_head = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Parameters
        ----------
        h : torch.Tensor
            Pooled graph representation.

        Returns
        -------
        cls_logit : torch.Tensor
            Classification logit.
        aux_score : torch.Tensor or None
            Auxiliary score if use_dual is True, else None.
        """
        cls_logit = self.cls_head(h).squeeze(-1)
        if self.use_dual:
            aux_score = self.aux_head(h).squeeze(-1)
            return cls_logit, aux_score
        return cls_logit, None
