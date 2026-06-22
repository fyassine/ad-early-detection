import json
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch_geometric.utils import to_dense_adj

if TYPE_CHECKING:
    from CLASSIFIER.model.GAAE.models import GraphAttentionAutoencoderConditioned

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


def load_gaae_for_inference(
    ckpt_path: Path | str,
    in_features: int,
    config: dict,
    device: torch.device | str = "cpu",
) -> "GraphAttentionAutoencoderConditioned":
    """
    Instantiate and load a frozen GAAE model for notebook inference.

    config keys used: latent_dim, hidden_dim, num_heads, cond_dim, dropout.
    in_features must be probed by the caller from a dataset sample so this
    function has no dataset dependency.
    """
    from CLASSIFIER.model.GAAE.models import GraphAttentionAutoencoderConditioned

    model = GraphAttentionAutoencoderConditioned(
        in_features=in_features,
        hidden_dim=config.get("hidden_dim", in_features),
        out_features=config.get("latent_dim", 64),
        cond_dim=config.get("cond_dim", 2),
        num_heads=config.get("num_heads", 2),
        dropout=config.get("dropout", 0.3),
    )
    ckpt_obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt_obj, torch.nn.Module):
        model = ckpt_obj
    elif isinstance(ckpt_obj, dict):
        state = ckpt_obj.get("model_state_dict", ckpt_obj)
        model.load_state_dict(state)
    else:
        raise TypeError(
            f"Unsupported checkpoint type: {type(ckpt_obj)}. "
            "Expected torch.nn.Module or state_dict."
        )
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model

