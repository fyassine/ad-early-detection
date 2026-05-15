"""
CLASSIFIER_v2 GELSTM/utils.py — Utilities for the GELSTM model.

v2 additions (vs CLASSIFIER/model/GELSTM/utils.py):
    * shuffle_order kwarg in encode_batch_sequences — randomly permutes the
      visit order (graphs + Δt) per subject. Used by SANITY_LSTM_CHECKS to test
      whether the LSTM is exploiting temporal order or only the visit content.
    * drop_time_delta is unchanged but documented as a Δt-ablation switch.
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


def encode_batch_sequences(
    batch: List[Dict],
    encoder_model: "torch.nn.Module",
    device: torch.device,
    use_time_delta: bool = True,
    graph_pool: str = "mean",
    dim_filter: "np.ndarray | None" = None,
    shuffle_order: bool = False,
    shuffle_rng: "np.random.Generator | None" = None,
) -> Tuple[PackedSequence, torch.Tensor, torch.Tensor]:
    """
    Encode a list of subject dicts into a PackedSequence for GELSTMClassifier.

    Parameters
    ----------
    batch : list of dicts from LongitudinalSubjectDataset.__getitem__
    encoder_model : exposes encode_visit() or encode() returning node embeddings.
    device : torch.device
    use_time_delta : bool
        If False, the [z_t ‖ Δt_t] concat step is skipped. Used for the
        SANITY_LSTM_CHECKS "Δt removed" ablation.
    graph_pool : 'mean' | 'max' | 'sum'
    dim_filter : np.ndarray of int indices or None
        FDR-based latent-dimension selection.
    shuffle_order : bool
        If True, randomly permute (graphs, delta_t) per subject before encoding.
        Used by SANITY_LSTM_CHECKS to test whether the LSTM relies on temporal
        order. Δt is permuted alongside the visits so the per-step value follows
        its original visit; this preserves Δt's marginal distribution while
        destroying ordering. *Use only at evaluation time.*
    shuffle_rng : np.random.Generator, optional
        Pass to make the shuffle deterministic.

    Returns
    -------
    packed_seqs : PackedSequence
    labels      : torch.Tensor (B,)
    lengths     : torch.Tensor (B,)  — sorted descending
    """
    rng = shuffle_rng if shuffle_rng is not None else np.random.default_rng()

    # Optional visit-order shuffling per subject (sanity-check switch).
    if shuffle_order:
        shuffled = []
        for item in batch:
            T = len(item["graphs"])
            perm = rng.permutation(T)
            new_item = dict(item)
            new_item["graphs"]  = [item["graphs"][i] for i in perm]
            new_item["delta_t"] = [item["delta_t"][i] for i in perm]
            new_item["visit_months"] = [item["visit_months"][i] for i in perm]
            shuffled.append(new_item)
        batch = shuffled

    batch = sorted(batch, key=lambda b: len(b["graphs"]), reverse=True)

    seq_list:    List[torch.Tensor] = []
    labels_list: List[int]          = []
    lengths_list: List[int]         = []

    encoder_model.eval()
    with torch.no_grad():
        for item in batch:
            graphs  = item["graphs"]
            deltas  = item["delta_t"]
            T       = len(graphs)

            step_embs = []
            for t, g in enumerate(graphs):
                x  = g.x.to(device)
                ei = g.edge_index.to(device)
                ea = g.edge_attr.to(device) if g.edge_attr is not None else None

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

                if dim_filter is not None:
                    _idx = torch.tensor(
                        np.asarray(dim_filter).copy(), dtype=torch.long, device=device
                    )
                    z_t  = z_t[_idx]

                if use_time_delta:
                    dt  = torch.tensor([deltas[t]], dtype=torch.float, device=device)
                    z_t = torch.cat([z_t, dt], dim=0)

                step_embs.append(z_t)

            seq_tensor = torch.stack(step_embs, dim=0)
            seq_list.append(seq_tensor)
            labels_list.append(item["label"])
            lengths_list.append(T)

    max_T = max(lengths_list)
    feat_dim = seq_list[0].size(1)
    padded = torch.zeros(len(batch), max_T, feat_dim, device=device)
    for i, seq in enumerate(seq_list):
        padded[i, :seq.size(0)] = seq

    lengths = torch.tensor(lengths_list, dtype=torch.long)
    labels  = torch.tensor(labels_list,  dtype=torch.float, device=device)

    packed  = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=True)
    return packed, labels, lengths
