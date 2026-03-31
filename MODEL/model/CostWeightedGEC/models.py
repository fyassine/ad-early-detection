import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, BatchNorm, global_mean_pool, GlobalAttention


class GraphEncoderClassifier(nn.Module):
    def __init__(self, in_features, hidden_dim, latent_dim, cond_dim, num_heads=1, dropout=0.0, classifier_hidden=64):
        super(GraphEncoderClassifier, self).__init__()
        self.dropout = dropout

        self.encoder_gat1 = GATv2Conv(in_features, hidden_dim, heads=num_heads, concat=True)
        self.encoder_bn1 = BatchNorm(hidden_dim * num_heads)
        self.encoder_gat2 = GATv2Conv(hidden_dim * num_heads, hidden_dim, heads=num_heads, concat=True)
        self.encoder_bn2 = BatchNorm(hidden_dim * num_heads)
        self.encoder_gat3 = GATv2Conv(hidden_dim * num_heads, latent_dim, heads=num_heads, concat=False)

        self.film_gamma = nn.Sequential(
            nn.Linear(cond_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        self.film_beta = nn.Sequential(
            nn.Linear(cond_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, 1)
        )

    def encode(self, x, edge_index):
        x = self.encoder_gat1(x, edge_index)
        x = self.encoder_bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.encoder_gat2(x, edge_index)
        x = self.encoder_bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        z = self.encoder_gat3(x, edge_index)
        return z

    def condition_latent(self, z, cond_vec, batch_mask):
        gamma = self.film_gamma(cond_vec)
        beta = self.film_beta(cond_vec)
        gamma_per_node = gamma[batch_mask]
        beta_per_node = beta[batch_mask]
        z = gamma_per_node * z + beta_per_node
        return z

    def forward(self, x, edge_index, cond_vec, batch_mask):
        z = self.encode(x, edge_index)
        z = self.condition_latent(z, cond_vec, batch_mask)
        graph_embedding = global_mean_pool(z, batch_mask)
        logits = self.classifier(graph_embedding)
        return logits.squeeze(-1), graph_embedding

    def freeze_encoder(self):
        for param in self.encoder_gat1.parameters():
            param.requires_grad = False
        for param in self.encoder_bn1.parameters():
            param.requires_grad = False
        for param in self.encoder_gat2.parameters():
            param.requires_grad = False
        for param in self.encoder_bn2.parameters():
            param.requires_grad = False
        for param in self.encoder_gat3.parameters():
            param.requires_grad = False
        for param in self.film_gamma.parameters():
            param.requires_grad = False
        for param in self.film_beta.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder_gat1.parameters():
            param.requires_grad = True
        for param in self.encoder_bn1.parameters():
            param.requires_grad = True
        for param in self.encoder_gat2.parameters():
            param.requires_grad = True
        for param in self.encoder_bn2.parameters():
            param.requires_grad = True
        for param in self.encoder_gat3.parameters():
            param.requires_grad = True
        for param in self.film_gamma.parameters():
            param.requires_grad = True
        for param in self.film_beta.parameters():
            param.requires_grad = True

    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]


class GraphEncoderClassifierAttention(nn.Module):
    def __init__(self, in_features, hidden_dim, latent_dim, cond_dim, num_heads=1, dropout=0.0, classifier_hidden=64):
        super(GraphEncoderClassifierAttention, self).__init__()
        self.dropout = dropout

        self.encoder_gat1 = GATv2Conv(in_features, hidden_dim, heads=num_heads, concat=True)
        self.encoder_bn1 = BatchNorm(hidden_dim * num_heads)
        self.encoder_gat2 = GATv2Conv(hidden_dim * num_heads, hidden_dim, heads=num_heads, concat=True)
        self.encoder_bn2 = BatchNorm(hidden_dim * num_heads)
        self.encoder_gat3 = GATv2Conv(hidden_dim * num_heads, latent_dim, heads=num_heads, concat=False)

        self.film_gamma = nn.Sequential(
            nn.Linear(cond_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        self.film_beta = nn.Sequential(
            nn.Linear(cond_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

        gate_nn = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1)
        )
        self.attention_pool = GlobalAttention(gate_nn)

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, 1)
        )

    def encode(self, x, edge_index):
        x = self.encoder_gat1(x, edge_index)
        x = self.encoder_bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.encoder_gat2(x, edge_index)
        x = self.encoder_bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        z = self.encoder_gat3(x, edge_index)
        return z

    def condition_latent(self, z, cond_vec, batch_mask):
        gamma = self.film_gamma(cond_vec)
        beta = self.film_beta(cond_vec)
        gamma_per_node = gamma[batch_mask]
        beta_per_node = beta[batch_mask]
        z = gamma_per_node * z + beta_per_node
        return z

    def forward(self, x, edge_index, cond_vec, batch_mask):
        z = self.encode(x, edge_index)
        z = self.condition_latent(z, cond_vec, batch_mask)
        graph_embedding = self.attention_pool(z, batch_mask)
        logits = self.classifier(graph_embedding)
        return logits.squeeze(-1), graph_embedding

    def freeze_encoder(self):
        for param in self.encoder_gat1.parameters():
            param.requires_grad = False
        for param in self.encoder_bn1.parameters():
            param.requires_grad = False
        for param in self.encoder_gat2.parameters():
            param.requires_grad = False
        for param in self.encoder_bn2.parameters():
            param.requires_grad = False
        for param in self.encoder_gat3.parameters():
            param.requires_grad = False
        for param in self.film_gamma.parameters():
            param.requires_grad = False
        for param in self.film_beta.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder_gat1.parameters():
            param.requires_grad = True
        for param in self.encoder_bn1.parameters():
            param.requires_grad = True
        for param in self.encoder_gat2.parameters():
            param.requires_grad = True
        for param in self.encoder_bn2.parameters():
            param.requires_grad = True
        for param in self.encoder_gat3.parameters():
            param.requires_grad = True
        for param in self.film_gamma.parameters():
            param.requires_grad = True
        for param in self.film_beta.parameters():
            param.requires_grad = True

    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]
