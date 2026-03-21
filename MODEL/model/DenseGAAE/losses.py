import torch
import torch.nn.functional as F


def feature_reconstruction_loss(x, x_reconstructed):
    return F.mse_loss(x_reconstructed, x)


def adjacency_reconstruction_loss(adj_original, adj_reconstructed, mask):
    adj_original_flat = adj_original.view(-1)
    adj_reconstructed_flat = adj_reconstructed.view(-1)
    mask_flat = mask.view(-1)

    adj_original_selected = adj_original_flat[mask_flat]
    adj_reconstructed_selected = adj_reconstructed_flat[mask_flat]

    return F.mse_loss(adj_reconstructed_selected, adj_original_selected)


def total_loss_fn(x, x_reconstructed, adj_original, adj_reconstructed, mask, adj_loss_weight=1.0):
    feature_loss = feature_reconstruction_loss(x, x_reconstructed)
    adjacency_loss = adjacency_reconstruction_loss(adj_original, adj_reconstructed, mask)
    total_loss = feature_loss + adj_loss_weight * adjacency_loss
    return total_loss, feature_loss, adjacency_loss
