"""
GELSTM/utils.py — Utilities for the GELSTM model.

Key function:
    encode_batch_sequences() — encodes a batch of subject dicts into a PackedSequence
    ready for GELSTMClassifier.forward().
"""
from __future__ import annotations

import os
import random
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_class_weights(labels: List[int], device="cpu") -> torch.Tensor:
    """BCEWithLogitsLoss pos_weight: n_neg / n_pos."""
    labels  = np.array(labels)
    n_pos   = int(labels.sum())
    n_neg   = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return torch.tensor(1.0, device=device)
    return torch.tensor(n_neg / n_pos, dtype=torch.float, device=device)


def compute_class_cost_weights(
    labels: List[int], device="cpu", normalize: bool = True
) -> torch.Tensor:
    """Two-element cost-weight tensor [w_neg, w_pos]."""
    labels    = np.array(labels).astype(int)
    n_samples = len(labels)
    n_pos     = int(labels.sum())
    n_neg     = n_samples - n_pos
    if n_samples == 0 or n_pos == 0 or n_neg == 0:
        return torch.tensor([1.0, 1.0], dtype=torch.float, device=device)
    w0 = n_samples / (2.0 * n_neg)
    w1 = n_samples / (2.0 * n_pos)
    w  = torch.tensor([w0, w1], dtype=torch.float, device=device)
    if normalize:
        w = w / w.mean().clamp_min(1e-12)
    return w


# ── Batch encoding ────────────────────────────────────────────────────────────

def encode_batch_sequences(
    batch: List[Dict],
    encoder_model: "torch.nn.Module",
    device: torch.device,
    use_time_delta: bool = True,
    graph_pool: str = "mean",
    dim_filter: "np.ndarray | None" = None,
) -> Tuple[PackedSequence, torch.Tensor, torch.Tensor]:
    """
    Encode a list of subject dicts (from LongitudinalSubjectDataset) into a
    PackedSequence suitable for GELSTMClassifier.forward().

    Steps per subject:
        1. For each visit graph: run encoder → mean/max/sum pool → z_t ∈ R^d
        2. Optionally concatenate normalised Δt: z_t' = [z_t ‖ Δt_t]
        3. Stack into (T, d+1) tensor

    Batch is sorted by sequence length (descending) as required by PyTorch's
    pack_padded_sequence with enforce_sorted=True.

    Parameters
    ----------
    batch : list of dicts from LongitudinalSubjectDataset.__getitem__
    encoder_model : GELSTMClassifier or GraphAttentionAutoencoderConditioned
        Must expose encode_visit(x, edge_index, edge_attr) → (F,)
        OR encode(x, edge_index, edge_attr) → (N, F) with pooling applied here.
    device : torch.device
    use_time_delta : bool
    graph_pool : 'mean' | 'max' | 'sum'
    dim_filter : np.ndarray of int indices, or None
        If provided, z_t is projected to z_t[dim_filter] *before* Δt concatenation.
        This is the FDR-based dimension selection: pass top_dims[:TOP_K] from FDR
        analysis to reduce LSTM input from gaae_latent → TOP_K dimensions.
        Must be None (default) when using the standard GELSTM notebook.

    Returns
    -------
    packed_seqs : PackedSequence
    labels      : torch.Tensor  shape (B,)
    lengths     : torch.Tensor  shape (B,) — original sequence lengths (sorted desc)
    """
    # Sort batch by n_scans descending
    batch = sorted(batch, key=lambda b: len(b["graphs"]), reverse=True)

    seq_list: List[torch.Tensor] = []
    labels_list: List[int]       = []
    lengths_list: List[int]      = []

    encoder_model.eval()
    with torch.no_grad():
        for item in batch:
            graphs  = item["graphs"]
            deltas  = item["delta_t"]   # list of floats, len == len(graphs)
            T       = len(graphs)

            step_embs = []
            for t, g in enumerate(graphs):
                x  = g.x.to(device)
                ei = g.edge_index.to(device)
                ea = g.edge_attr.to(device) if g.edge_attr is not None else None

                # Support both GELSTMClassifier and bare GAAE encoder
                if hasattr(encoder_model, "encode_visit"):
                    z_t = encoder_model.encode_visit(x, ei, ea, pool=graph_pool)
                else:
                    z_nodes = encoder_model.encode(x, ei, ea)
                    if graph_pool == "mean":
                        z_t = z_nodes.mean(dim=0)
                    elif graph_pool == "max":
                        z_t = z_nodes.max(dim=0).values
                    else:
                        z_t = z_nodes.sum(dim=0)

                # FDR dimension filter: project to top-K discriminative dims
                # Note: dim_filter may have negative strides (np.argsort()[::-1][:K]),
                # so .copy() is called to ensure a contiguous C-order array before
                # converting to tensor (PyTorch does not support negative-stride arrays).
                if dim_filter is not None:
                    _idx = torch.tensor(
                        np.asarray(dim_filter).copy(), dtype=torch.long, device=device
                    )
                    z_t  = z_t[_idx]   # (TOP_K,)

                if use_time_delta:
                    dt  = torch.tensor([deltas[t]], dtype=torch.float, device=device)
                    z_t = torch.cat([z_t, dt], dim=0)

                step_embs.append(z_t)

            seq_tensor = torch.stack(step_embs, dim=0)   # (T, d+1)
            seq_list.append(seq_tensor)
            labels_list.append(item["label"])
            lengths_list.append(T)

    # Pad to max length
    max_T = max(lengths_list)
    feat_dim = seq_list[0].size(1)
    padded = torch.zeros(len(batch), max_T, feat_dim, device=device)
    for i, seq in enumerate(seq_list):
        padded[i, :seq.size(0)] = seq

    lengths = torch.tensor(lengths_list, dtype=torch.long)
    labels  = torch.tensor(labels_list,  dtype=torch.float, device=device)

    packed  = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=True)
    return packed, labels, lengths
