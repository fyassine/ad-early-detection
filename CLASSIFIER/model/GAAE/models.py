import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, GATv2Conv, InnerProductDecoder

class GraphAttentionAutoencoderConditioned(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features, cond_dim, num_heads=1, dropout=0.0):
        super(GraphAttentionAutoencoderConditioned, self).__init__()
        self.dropout = dropout
        self.edge_dim = 1
        self.adj_decoder = InnerProductDecoder()

        # Encoder
        self.encoder_gat1 = GATv2Conv(
            in_features,
            hidden_dim,
            heads=num_heads,
            concat=True,
            edge_dim=self.edge_dim,
            residual=True,
        )
        self.encoder_bn1 = BatchNorm(hidden_dim * num_heads)
        self.encoder_gat2 = GATv2Conv(
            hidden_dim * num_heads,
            hidden_dim,
            heads=num_heads,
            concat=True,
            edge_dim=self.edge_dim,
            residual=True,
        )
        self.encoder_bn2 = BatchNorm(hidden_dim * num_heads)
        self.encoder_gat3 = GATv2Conv(
            hidden_dim * num_heads,
            out_features,
            heads=num_heads,
            concat=False,
            edge_dim=self.edge_dim,
            residual=True,
        )

        # FiLM modulator
        self.film_gamma = nn.Sequential(
            nn.Linear(cond_dim, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features)
        )
        self.film_beta = nn.Sequential(
            nn.Linear(cond_dim, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features)
        )

        # Decoder
        self.decoder_gat1 = GATv2Conv(
            out_features,
            hidden_dim,
            heads=num_heads,
            concat=True,
            edge_dim=self.edge_dim,
            residual=True,
        )
        self.decoder_bn1 = BatchNorm(hidden_dim * num_heads)
        self.decoder_gat2 = GATv2Conv(
            hidden_dim * num_heads,
            hidden_dim,
            heads=num_heads,
            concat=True,
            edge_dim=self.edge_dim,
            residual=True,
        )
        self.decoder_bn2 = BatchNorm(hidden_dim * num_heads)
        self.decoder_gat3 = GATv2Conv(
            hidden_dim * num_heads,
            in_features,
            heads=num_heads,
            concat=False,
            edge_dim=self.edge_dim,
            residual=True,
        )

    @staticmethod
    def _normalize_edge_attr(edge_attr):
        if edge_attr is None:
            return None
        if edge_attr.dim() == 1:
            return edge_attr.unsqueeze(-1)
        return edge_attr

    def encode(self, x, edge_index, edge_attr, return_attention=False):
        edge_attr = self._normalize_edge_attr(edge_attr)
        attention_weights = []

        if return_attention:
            x, attn = self.encoder_gat1(
                x, edge_index, edge_attr=edge_attr, return_attention_weights=True
            )
            attention_weights.append(attn)
        else:
            x = self.encoder_gat1(x, edge_index, edge_attr=edge_attr)
        x = self.encoder_bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if return_attention:
            x, attn = self.encoder_gat2(
                x, edge_index, edge_attr=edge_attr, return_attention_weights=True
            )
            attention_weights.append(attn)
        else:
            x = self.encoder_gat2(x, edge_index, edge_attr=edge_attr)
        x = self.encoder_bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if return_attention:
            z, attn = self.encoder_gat3(
                x, edge_index, edge_attr=edge_attr, return_attention_weights=True
            )
            attention_weights.append(attn)
            return z, attention_weights

        z = self.encoder_gat3(x, edge_index, edge_attr=edge_attr)
        return z

    def condition_latent(self, z, cond_vec, batch_mask):
        gamma = self.film_gamma(cond_vec)  # shape: [batch_size, F]
        beta = self.film_beta(cond_vec)    # shape: [batch_size, F]
        gamma_per_node = gamma[batch_mask]  # [num_nodes, F]
        beta_per_node = beta[batch_mask]    # [num_nodes, F]
        z = gamma_per_node * z + beta_per_node
        return z

    def decode_features(self, z, edge_index, edge_attr, return_attention=False):
        edge_attr = self._normalize_edge_attr(edge_attr)
        attention_weights = []

        if return_attention:
            x, attn = self.decoder_gat1(
                z, edge_index, edge_attr=edge_attr, return_attention_weights=True
            )
            attention_weights.append(attn)
        else:
            x = self.decoder_gat1(z, edge_index, edge_attr=edge_attr)
        x = self.decoder_bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if return_attention:
            x, attn = self.decoder_gat2(
                x, edge_index, edge_attr=edge_attr, return_attention_weights=True
            )
            attention_weights.append(attn)
        else:
            x = self.decoder_gat2(x, edge_index, edge_attr=edge_attr)
        x = self.decoder_bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if return_attention:
            x, attn = self.decoder_gat3(
                x, edge_index, edge_attr=edge_attr, return_attention_weights=True
            )
            attention_weights.append(attn)
            return x, attention_weights

        x = self.decoder_gat3(x, edge_index, edge_attr=edge_attr)
        return x

    def decode_adjacency(self, z, edge_index):
        return self.adj_decoder(z, edge_index)

    def forward(self, x, edge_index, edge_attr, cond_vec, batch_mask):
        z, encoder_attention = self.encode(x, edge_index, edge_attr, return_attention=True)
        z = self.condition_latent(z, cond_vec, batch_mask)
        x_reconstructed, decoder_attention = self.decode_features(
            z, edge_index, edge_attr, return_attention=True
        )
        adj_reconstructed = self.decode_adjacency(z, edge_index)
        attention_weights = {
            "encoder": encoder_attention,
            "decoder": decoder_attention,
        }
        return z, x_reconstructed, adj_reconstructed, attention_weights
