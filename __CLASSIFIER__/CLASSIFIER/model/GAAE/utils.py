import json
import os
from datetime import datetime

import numpy as np
import torch
from torch_geometric.utils import to_dense_adj

def knn_binary_adjacency_matrix_no_diag(corr_matrix, k):
    """
    Generate a k-nearest neighbor binary adjacency matrix from a correlation matrix,
    excluding self-connections (diagonal elements) from consideration.
    """
    N = corr_matrix.shape[0]
    adjacency_matrix = np.zeros_like(corr_matrix)

    for i in range(N):
        corr_row = np.copy(corr_matrix[i, :])
        corr_row[i] = -np.inf  # Ensure self-connections are not considered

        nearest_indices = np.argsort(-corr_row)[:k] 
        adjacency_matrix[i, nearest_indices] = 1

    binary_adjacency_matrix = np.maximum(adjacency_matrix, adjacency_matrix.T)

    return binary_adjacency_matrix

def calculate_dense_adjacency(data):
    """
    Converts sparse edge_index to dense adjacency matrix.
    
    Args:
        data (Data): PyTorch Geometric Data object with edge_index.
    
    Returns:
        torch.Tensor: Dense adjacency matrix of shape [N, N].
    """
    dense_adj = to_dense_adj(data.edge_index, max_num_nodes=data.x.shape[0]).squeeze(0)
    return dense_adj

def create_mask(batch):
    """
    Creates a mask for adjacency matrices in batched graph data.
    
    This is useful when processing multiple graphs of different sizes in a batch.
    The mask identifies which regions of the batched adjacency matrix correspond
    to actual graph connections vs. padding.
    
    Args:
        batch (torch.Tensor): A tensor where each node is assigned a graph index.
    
    Returns:
        torch.Tensor: A mask of shape [N, N] where valid graph regions are 1 
                      and padded regions are 0.
    """
    num_nodes_per_graph = torch.bincount(batch)
    N = batch.size(0)  # Total number of nodes
    mask = torch.zeros((N, N), device=batch.device, dtype=torch.bool)
    
    start_idx = 0
    for num_nodes in num_nodes_per_graph:
        if num_nodes > 0:
            mask[start_idx:start_idx + num_nodes, start_idx:start_idx + num_nodes] = True
            start_idx += num_nodes
    
    return mask

def save_run_config(run_name, timestamp, dataset_info, model_config, training_config, run_artifact_dir):
    """
    Saves the run configuration to a JSON file.
    """
    config_to_save = {
        "run_name": run_name,
        "timestamp": timestamp,
        "dataset_info": dataset_info,
        "model_config": model_config,
        "training_config": training_config
    }

    # Helper to convert non-serializable objects (like device) to string
    def json_serial(obj):
        if isinstance(obj, (datetime, torch.device)):
            return str(obj)
        raise TypeError (f"Type {type(obj)} not serializable")

    config_filename = "run_config.json"
    config_file = os.path.join(run_artifact_dir, config_filename)
    with open(config_file, "w") as f:
        json.dump(config_to_save, f, indent=4, default=json_serial)
    print(f"Saved run configuration to {config_file}")


