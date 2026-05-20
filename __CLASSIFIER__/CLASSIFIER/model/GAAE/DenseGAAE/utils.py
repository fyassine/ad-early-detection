import json
import os
from datetime import datetime

import numpy as np
import torch
from torch_geometric.utils import to_dense_adj


def dense_adjacency_from_corr(corr_matrix, use_abs=False, zero_diag=True):
    if use_abs:
        corr_matrix = np.abs(corr_matrix)
    if zero_diag:
        corr_matrix = corr_matrix.copy()
        np.fill_diagonal(corr_matrix, 0.0)
    return corr_matrix


def build_complete_edge_index(num_nodes, device=None):
    idx = torch.arange(num_nodes, device=device)
    row = idx.repeat_interleave(num_nodes)
    col = idx.repeat(num_nodes)
    mask = row != col
    return torch.stack([row[mask], col[mask]], dim=0)


def calculate_dense_adjacency(edge_index, batch, edge_attr=None, max_num_nodes=None):
    dense_adj = to_dense_adj(
        edge_index,
        batch=batch,
        edge_attr=edge_attr,
        max_num_nodes=max_num_nodes,
    ).squeeze(0)
    return dense_adj


def create_mask(batch):
    num_nodes_per_graph = torch.bincount(batch)
    num_nodes = batch.size(0)
    mask = torch.zeros((num_nodes, num_nodes), device=batch.device, dtype=torch.bool)

    start_idx = 0
    for nodes_in_graph in num_nodes_per_graph:
        if nodes_in_graph > 0:
            end_idx = start_idx + nodes_in_graph
            mask[start_idx:end_idx, start_idx:end_idx] = True
            start_idx = end_idx

    return mask


def save_run_config(run_name, timestamp, dataset_info, model_config, training_config, run_artifact_dir):
    config_to_save = {
        "run_name": run_name,
        "timestamp": timestamp,
        "dataset_info": dataset_info,
        "model_config": model_config,
        "training_config": training_config,
    }

    def json_serial(obj):
        if isinstance(obj, (datetime, torch.device)):
            return str(obj)
        raise TypeError(f"Type {type(obj)} not serializable")

    config_filename = "run_config.json"
    config_file = os.path.join(run_artifact_dir, config_filename)
    with open(config_file, "w") as handle:
        json.dump(config_to_save, handle, indent=4, default=json_serial)
    print(f"Saved run configuration to {config_file}")
