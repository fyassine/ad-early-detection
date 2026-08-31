"""TFGN/layers.py — Neural network layers and modules for the TFGN model."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, GATv2Conv


def sparsemax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Exact sparsemax activation function (Martins & Astudillo 2016).

    Produces a sparse probability distribution with exact zeros.
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
    """Standard LSTM applied node-wise (shared weights across nodes).

    The same LSTM processes each node's time series independently. When
    ``use_time_delta=True``, the inter-visit / cumulative time value ``log_dt``
    is concatenated to each node's feature vector across visits.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        use_time_delta: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_time_delta = use_time_delta
        effective_in = input_dim + (1 if use_time_delta else 0)
        self.lstm = nn.LSTM(
            input_size=effective_in,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor, log_dt: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, T, N, input_dim).
        log_dt : torch.Tensor, optional
            Time values of shape (B, T) or (T,).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, T, N, hidden_dim).
        """
        B, T, N, d_in = x.shape

        if self.use_time_delta and log_dt is not None:
            if log_dt.dim() == 1:
                log_dt = log_dt.unsqueeze(0)  # (1, T)
            # Expand log_dt: (B, T, 1, 1) -> (B, T, N, 1)
            dt_exp = log_dt.unsqueeze(-1).unsqueeze(-1).expand(B, T, N, 1)
            x_in = torch.cat([x, dt_exp], dim=-1)
        else:
            x_in = x

        eff_dim = x_in.shape[-1]
        x_reshaped = x_in.permute(0, 2, 1, 3).reshape(B * N, T, eff_dim)
        out, _ = self.lstm(x_reshaped)
        out = out.view(B, N, T, self.hidden_dim).permute(0, 2, 1, 3)
        return out


class TemporalSaliencyGate(nn.Module):
    """Temporal Saliency Gate.

    Computes a gating scalar per node using a projection and LeakyReLU,
    followed by a sigmoid to bound values in (0, 1), and scales features
    via residual connection: (1 + s_i) * h_i.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.W_s = nn.Linear(hidden_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Parameters
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
        gated_h = (1.0 + gate_scores) * h
        return gated_h, gate_scores


class GVAEEncoder(nn.Module):
    """GVAE Encoder with GATv2 μ/logσ² heads + FiLM conditioning on μ.

    Applied to the node-shared temporal representations over baseline graph A_0.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        latent_dim: int,
        num_heads: int = 2,
        dropout: float = 0.3,
        cond_dim: int = 2,
    ) -> None:
        super().__init__()
        self.dropout = dropout

        self.shared = GATv2Conv(
            in_features,
            hidden_dim,
            heads=num_heads,
            concat=True,
            edge_dim=1,
            residual=True,
        )
        self.bn = BatchNorm(hidden_dim * num_heads)

        self.conv_mu = GATv2Conv(
            hidden_dim * num_heads,
            latent_dim,
            heads=num_heads,
            concat=False,
            edge_dim=1,
            residual=True,
        )
        self.conv_logvar = GATv2Conv(
            hidden_dim * num_heads,
            latent_dim,
            heads=num_heads,
            concat=False,
            edge_dim=1,
            residual=True,
        )

        self.film_gamma = nn.Sequential(
            nn.Linear(cond_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.film_beta = nn.Sequential(
            nn.Linear(cond_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    @staticmethod
    def _normalize_edge_attr(
        edge_attr: torch.Tensor | None,
    ) -> torch.Tensor | None:
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
        cond_vec: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Parameters
        ----------
        x : torch.Tensor
            Node features (e.g. gated temporal states h_T).
        edge_index : torch.Tensor
            Baseline graph edge indices.
        edge_attr : torch.Tensor, optional
            Baseline graph edge weights.
        cond_vec : torch.Tensor, optional
            FiLM conditioning vector (e.g., age, sex).

        Returns
        -------
        z : torch.Tensor
            Sampled latent during training, or conditioned mu during evaluation.
        mu : torch.Tensor
            Conditioned latent mean.
        logvar : torch.Tensor
            Latent log variance.
        mu_raw : torch.Tensor
            Unconditioned (pre-FiLM) latent mean for unpenalized KL computation.
        """
        ea = self._normalize_edge_attr(edge_attr)

        h = self.shared(x, edge_index, edge_attr=ea)
        h = self.bn(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        mu_raw = self.conv_mu(h, edge_index, edge_attr=ea)
        logvar = self.conv_logvar(h, edge_index, edge_attr=ea)

        if cond_vec is not None:
            gamma = self.film_gamma(cond_vec)
            beta = self.film_beta(cond_vec)
            mu = gamma * mu_raw + beta
        else:
            mu = mu_raw

        z = self.reparameterize(mu, logvar)
        return z, mu, logvar, mu_raw


class ConcatResidualFusion(nn.Module):
    """Concatenates hidden state and latent vectors, projects, and applies LayerNorm."""

    def __init__(self, h_dim: int, z_dim: int, out_dim: int | None = None) -> None:
        super().__init__()
        out_dim = out_dim if out_dim is not None else h_dim
        self.W_u = nn.Linear(h_dim + z_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h, z], dim=-1)
        out = self.W_u(out)
        out = self.norm(out)
        return out


class AttentivePool(nn.Module):
    """Attention-weighted pooling over nodes, using exact sparsemax."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attn_proj = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Parameters
        ----------
        h : torch.Tensor
            Input of shape (N, D) or (..., N, D).

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
    """Produces classification logit and topological saliency scores s_topo."""

    def __init__(self, hidden_dim: int, use_dual: bool = True) -> None:
        super().__init__()
        self.use_dual = use_dual
        self.cls_head = nn.Linear(hidden_dim, 1)
        self.topo_head = nn.Linear(hidden_dim, 1) if use_dual else None

    def forward(
        self, h_pooled: torch.Tensor, h_nodes: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Parameters
        ----------
        h_pooled : torch.Tensor
            Pooled graph representation of shape (..., hidden_dim).
        h_nodes : torch.Tensor, optional
            Per-node representations of shape (..., N, hidden_dim).

        Returns
        -------
        cls_logit : torch.Tensor
            Classification logit of shape (...).
        s_topo : torch.Tensor or None
            Topological saliency scores of shape (..., N) if use_dual is True.
        """
        cls_logit = self.cls_head(h_pooled).squeeze(-1)
        s_topo = None
        if self.use_dual and self.topo_head is not None and h_nodes is not None:
            s_topo = self.topo_head(h_nodes).squeeze(-1)
        return cls_logit, s_topo


class _GradientReversal(torch.autograd.Function):
    """Identity on the forward pass, negates and scales the gradient on backward.

    Standard DANN (Ganin & Lempitsev 2015) trick: placing this between a
    representation and a domain/cohort classifier turns "minimize cohort CE"
    into "maximize cohort CE from the representation's perspective" without
    needing a separate min-max optimizer loop -- the classifier's own gradient
    descent step, run through this function, does the adversarial update.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    """Apply the gradient-reversal layer with reversal strength ``lambda_``."""
    return _GradientReversal.apply(x, lambda_)


class CohortAdversaryHead(nn.Module):
    """Binary cohort (ADNI vs. DELCODE) classifier for adversarial conditioning.

    Consumes the patient-pooled embedding through a gradient-reversal layer
    (``grad_reverse`` — applied by the caller, not here, so the reversal
    strength can be warmed up per-epoch like ``beta_kl``) and predicts cohort
    identity. Trained normally (minimize CE); the reversed gradient is what
    pushes the *upstream* encoder toward cohort-invariant representations.
    Two cohorts only — OASIS-3 is never in the pooled training set
    (`DOCS/flipped/PLAN.md` "Decisions already fixed": kept fully external),
    so this is a binary head, not a 3-way one.
    """

    def __init__(self, hidden_dim: int, adv_hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, adv_hidden),
            nn.ReLU(),
            nn.Linear(adv_hidden, 1),
        )

    def forward(self, h_pooled: torch.Tensor) -> torch.Tensor:
        return self.net(h_pooled).squeeze(-1)
