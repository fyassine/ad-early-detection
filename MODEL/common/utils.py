import torch
import numpy as np
from torch_geometric.utils import to_dense_adj


def load_frozen_encoder_from_gaae(gec_model, gaae_checkpoint_path, device='cpu'):
    checkpoint = torch.load(gaae_checkpoint_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        gaae_state_dict = checkpoint['model_state_dict']
    else:
        gaae_state_dict = checkpoint
    
    encoder_keys = [
        'encoder_gat1', 'encoder_bn1',
        'encoder_gat2', 'encoder_bn2',
        'encoder_gat3',
        'film_gamma', 'film_beta'
    ]
    
    gec_state_dict = gec_model.state_dict()
    keys_to_load = {}
    mismatched_keys = []
    
    for key in gaae_state_dict:
        if any(key.startswith(enc_key) for enc_key in encoder_keys):
            if key in gec_state_dict:
                if gaae_state_dict[key].shape != gec_state_dict[key].shape:
                    mismatched_keys.append(
                        (key, tuple(gaae_state_dict[key].shape), tuple(gec_state_dict[key].shape))
                    )
                    continue
                keys_to_load[key] = gaae_state_dict[key]

    if mismatched_keys:
        mismatch_lines = [
            f"- {k}: checkpoint{src_shape} != model{dst_shape}"
            for k, src_shape, dst_shape in mismatched_keys
        ]
        raise ValueError(
            "Checkpoint encoder is incompatible with current GEC model dimensions.\n"
            "Mismatched keys:\n" + "\n".join(mismatch_lines)
        )

    if not keys_to_load:
        raise ValueError(
            "No encoder keys from the checkpoint matched the current GEC model. "
            "Check checkpoint format and model architecture compatibility."
        )

    gec_state_dict.update(keys_to_load)
    
    gec_model.load_state_dict(gec_state_dict)
    gec_model.freeze_encoder()
    print(f"Loaded {len(keys_to_load)} pretrained encoder parameters from {gaae_checkpoint_path}")
    
    return gec_model


def compute_class_weights(labels, device='cpu'):
    labels = np.array(labels)
    n_samples = len(labels)
    n_positive = labels.sum()
    n_negative = n_samples - n_positive
    
    if n_positive == 0 or n_negative == 0:
        return torch.tensor(1.0, device=device)
    
    pos_weight = n_negative / n_positive
    
    return torch.tensor(pos_weight, dtype=torch.float, device=device)


def compute_class_cost_weights(labels, device='cpu', normalize=True):
    labels = np.array(labels).astype(int)
    n_samples = len(labels)
    n_positive = int(labels.sum())
    n_negative = int(n_samples - n_positive)

    if n_samples == 0:
        return torch.tensor([1.0, 1.0], dtype=torch.float, device=device)

    if n_positive == 0 or n_negative == 0:
        return torch.tensor([1.0, 1.0], dtype=torch.float, device=device)

    w0 = n_samples / (2.0 * n_negative)
    w1 = n_samples / (2.0 * n_positive)

    weights = torch.tensor([w0, w1], dtype=torch.float, device=device)

    if normalize:
        weights = weights / weights.mean().clamp_min(1e-12)

    return weights


def knn_binary_adjacency_matrix_no_diag(corr_matrix, k):
    N = corr_matrix.shape[0]
    adjacency_matrix = np.zeros_like(corr_matrix)

    for i in range(N):
        corr_row = np.copy(corr_matrix[i, :])
        corr_row[i] = -np.inf
        nearest_indices = np.argsort(-corr_row)[:k]
        adjacency_matrix[i, nearest_indices] = 1

    binary_adjacency_matrix = np.maximum(adjacency_matrix, adjacency_matrix.T)
    return binary_adjacency_matrix


def create_mask(batch_mask):
    num_nodes = batch_mask.size(0)
    mask = batch_mask.unsqueeze(0) == batch_mask.unsqueeze(1)
    return mask
