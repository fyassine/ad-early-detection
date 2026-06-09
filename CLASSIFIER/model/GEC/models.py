"""GEC graph encoder-classifier models with optional FiLM conditioning.

Two variants:
- GraphEncoderClassifier       — mean-pool graph embedding
- GraphEncoderClassifierAttention — learned attention pool
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, BatchNorm, global_mean_pool, GlobalAttention


def _set_requires_grad(modules: list[nn.Module], value: bool) -> None:
    """Set requires_grad for all parameters of the given modules."""
    for mod in modules:
        for p in mod.parameters():
            p.requires_grad_(value)


class GraphEncoderClassifier(nn.Module):
    """3-layer GAT encoder with FiLM conditioning → mean-pool → MLP classifier.

    Parameters
    ----------
    in_features : int     Node feature dimension (number of ROIs).
    hidden_dim  : int     Hidden dim per GAT layer.
    latent_dim  : int     Output dim of the encoder (graph embedding size).
    cond_dim    : int     Conditioning vector size (e.g. 2 for [age, sex]).
    num_heads   : int     Number of GAT attention heads.
    dropout     : float   Dropout probability in encoder and classifier.
    classifier_hidden : int  Hidden size of classifier MLP (0 = direct linear).
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        latent_dim: int,
        cond_dim: int,
        num_heads: int = 1,
        dropout: float = 0.0,
        classifier_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.dropout = dropout

        self.encoder_gat1 = GATv2Conv(in_features, hidden_dim, heads=num_heads, concat=True)
        self.encoder_bn1  = BatchNorm(hidden_dim * num_heads)
        self.encoder_gat2 = GATv2Conv(hidden_dim * num_heads, hidden_dim, heads=num_heads, concat=True)
        self.encoder_bn2  = BatchNorm(hidden_dim * num_heads)
        self.encoder_gat3 = GATv2Conv(hidden_dim * num_heads, latent_dim, heads=num_heads, concat=False)

        self.film_gamma = nn.Sequential(
            nn.Linear(cond_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, latent_dim)
        )
        self.film_beta = nn.Sequential(
            nn.Linear(cond_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, latent_dim)
        )

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, 1),
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return per-node latent embeddings (N, latent_dim)."""
        x = F.relu(self.encoder_bn1(self.encoder_gat1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.encoder_bn2(self.encoder_gat2(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.encoder_gat3(x, edge_index)

    def condition_latent(
        self, z: torch.Tensor, cond_vec: torch.Tensor, batch_mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply FiLM conditioning: z ← γ(cond)·z + β(cond)."""
        gamma = self.film_gamma(cond_vec)[batch_mask]
        beta  = self.film_beta(cond_vec)[batch_mask]
        return gamma * z + beta

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        cond_vec: torch.Tensor,
        batch_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (logits (B,), graph_embedding (B, latent_dim))."""
        z = self.encode(x, edge_index)
        z = self.condition_latent(z, cond_vec, batch_mask)
        graph_embedding = global_mean_pool(z, batch_mask)
        logits = self.classifier(graph_embedding).squeeze(-1)
        return logits, graph_embedding

    def _encoder_modules(self) -> list[nn.Module]:
        return [
            self.encoder_gat1, self.encoder_bn1,
            self.encoder_gat2, self.encoder_bn2,
            self.encoder_gat3,
            self.film_gamma, self.film_beta,
        ]

    def freeze_encoder(self) -> None:
        """Freeze all encoder + FiLM parameters."""
        _set_requires_grad(self._encoder_modules(), False)

    def unfreeze_encoder(self) -> None:
        """Unfreeze all encoder + FiLM parameters."""
        _set_requires_grad(self._encoder_modules(), True)

    def get_trainable_params(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]


class GraphEncoderClassifierAttention(nn.Module):
    """Same as GraphEncoderClassifier but uses a learned attention pool instead of mean-pool."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        latent_dim: int,
        cond_dim: int,
        num_heads: int = 1,
        dropout: float = 0.0,
        classifier_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.dropout = dropout

        self.encoder_gat1 = GATv2Conv(in_features, hidden_dim, heads=num_heads, concat=True)
        self.encoder_bn1  = BatchNorm(hidden_dim * num_heads)
        self.encoder_gat2 = GATv2Conv(hidden_dim * num_heads, hidden_dim, heads=num_heads, concat=True)
        self.encoder_bn2  = BatchNorm(hidden_dim * num_heads)
        self.encoder_gat3 = GATv2Conv(hidden_dim * num_heads, latent_dim, heads=num_heads, concat=False)

        self.film_gamma = nn.Sequential(
            nn.Linear(cond_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, latent_dim)
        )
        self.film_beta = nn.Sequential(
            nn.Linear(cond_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, latent_dim)
        )

        gate_nn = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, 1)
        )
        self.attention_pool = GlobalAttention(gate_nn)

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, 1),
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.encoder_bn1(self.encoder_gat1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.encoder_bn2(self.encoder_gat2(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.encoder_gat3(x, edge_index)

    def condition_latent(
        self, z: torch.Tensor, cond_vec: torch.Tensor, batch_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gamma = self.film_gamma(cond_vec)[batch_mask]
        beta  = self.film_beta(cond_vec)[batch_mask]
        return gamma * z + beta

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        cond_vec: torch.Tensor,
        batch_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x, edge_index)
        z = self.condition_latent(z, cond_vec, batch_mask)
        graph_embedding = self.attention_pool(z, batch_mask)
        logits = self.classifier(graph_embedding).squeeze(-1)
        return logits, graph_embedding

    def _encoder_modules(self) -> list[nn.Module]:
        return [
            self.encoder_gat1, self.encoder_bn1,
            self.encoder_gat2, self.encoder_bn2,
            self.encoder_gat3,
            self.film_gamma, self.film_beta,
        ]

    def freeze_encoder(self) -> None:
        _set_requires_grad(self._encoder_modules(), False)

    def unfreeze_encoder(self) -> None:
        _set_requires_grad(self._encoder_modules(), True)

    def get_trainable_params(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]
