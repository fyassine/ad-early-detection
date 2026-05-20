import torch
import torch.nn.functional as F
import numpy as np

from .utils import calculate_dense_adjacency

def feature_reconstruction_loss(x, x_reconstructed):
    """
    Calculates the mean squared error (MSE) loss for feature reconstruction.

    Args:
        x (torch.Tensor): Original node feature matrix of shape (N, F),
                          where N is the number of nodes, and F is the feature dimension.
        x_reconstructed (torch.Tensor): Reconstructed node feature matrix of shape (N, F).

    Returns:
        torch.Tensor: The reconstruction loss for node features.
    """
    return F.mse_loss(x_reconstructed, x)

def adjacency_reconstruction_loss(precomputed_adj, adj_reconstructed, mask):
    """
    Calculates the binary cross-entropy (BCE) loss for adjacency reconstruction
    while ignoring padded areas.

    Args:
        precomputed_adj (torch.Tensor): Precomputed dense adjacency matrix of shape [N, N].
        adj_reconstructed (torch.Tensor): Reconstructed adjacency matrix of shape [N, N].
        mask (torch.Tensor): Boolean mask of shape [N, N] where `True` marks valid entries.

    Returns:
        torch.Tensor: The reconstruction loss for the adjacency matrix.
    """
    # Flatten matrices and mask
    precomputed_adj_flat = precomputed_adj.view(-1)
    adj_reconstructed_flat = adj_reconstructed.view(-1)
    mask_flat = mask.view(-1)  # Flatten the mask

    # Apply the mask to filter out padded areas
    precomputed_adj_selected = precomputed_adj_flat[mask_flat]
    adj_reconstructed_selected = adj_reconstructed_flat[mask_flat]

    # Compute loss only for the valid (non-padded) elements
    return F.binary_cross_entropy(adj_reconstructed_selected, precomputed_adj_selected)

def total_loss_fn(x, x_reconstructed, adj_original, adj_reconstructed, mask, adj_loss_weight=1.0):
    """
    Combines the feature reconstruction loss and adjacency reconstruction loss.
    
    Args:
        x (Tensor): Original node features [N, F] where N=nodes, F=features
        x_reconstructed (Tensor): Reconstructed node features [N, F]
        adj_original (Tensor): Ground truth adjacency matrix [N, N]
        adj_reconstructed (Tensor): Reconstructed adjacency matrix [N, N]
        mask (Tensor): Mask for valid adjacency regions [N, N]
        adj_loss_weight (float): Weighting factor for the adjacency loss term
    
    Returns:
        total_loss (Tensor): Combined weighted loss
        feature_loss (Tensor): Feature reconstruction loss (MSE)
        adjacency_loss (Tensor): Adjacency reconstruction loss (Binary Cross-Entropy)
    """
    # Feature reconstruction loss (MSE)
    feature_loss = feature_reconstruction_loss(x, x_reconstructed)
    
    # Adjacency reconstruction loss (BCE)
    adjacency_loss = adjacency_reconstruction_loss(adj_original, adj_reconstructed, mask)
    
    # Total loss with weighted adjacency term
    total_loss = feature_loss + adj_loss_weight * adjacency_loss
    
    return total_loss, feature_loss, adjacency_loss

def evaluate_reconstruction_errors_with_ids(dataset, model, device, adj_loss_weight=1.0):
    """
    Evaluate reconstruction errors and return patient IDs.
    
    Args:
        dataset: PyTorch Geometric dataset
        model: Trained autoencoder model
        device: torch device (cpu or cuda)
        adj_loss_weight: Weight for adjacency loss component
    
    Returns:
        x_errors: List of feature reconstruction errors
        adj_errors: List of adjacency reconstruction errors
        total_errors: List of total weighted errors
        patient_ids: List of patient IDs
    """
    x_errors = []
    adj_errors = []
    total_errors = []
    patient_ids = []
    
    model.eval()
    for data in dataset:
        data = data.to(device)
        x, edge_index = data.x, data.edge_index
        
        # 1. Conditioning vector: [age, sex]
        cond_vec = torch.tensor([[data.patient_age.item(), float(data.patient_sex.item())]], device=device)
        
        # 2. Batch mask (all nodes belong to the same graph → batch = 0)
        batch_mask = torch.zeros(x.size(0), dtype=torch.long, device=device)
        
        with torch.no_grad():
            z, x_reconstructed, adj_reconstructed = model(x, edge_index, cond_vec, batch_mask)
        
        # 3. Compute reconstruction errors
        x_error = feature_reconstruction_loss(x, x_reconstructed).item()
        adj_original = calculate_dense_adjacency(data)
        adj_error = adjacency_reconstruction_loss_single_instance(adj_original, adj_reconstructed).item()
        total_error = x_error + adj_error * adj_loss_weight
        
        x_errors.append(x_error)
        adj_errors.append(adj_error)
        total_errors.append(total_error)
        
        # Check if patient_id exists, otherwise use a placeholder or skip
        if hasattr(data, 'patient_id'):
            patient_ids.append(data.patient_id)
        else:
            patient_ids.append(None)
    
    return x_errors, adj_errors, total_errors, patient_ids
