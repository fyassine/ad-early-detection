"""
GELSTM/train.py — Training and evaluation loops for GELSTMClassifier.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, confusion_matrix

from .utils import encode_batch_sequences


def train_epoch(
    model: "torch.nn.Module",
    batch_list: List[List[dict]],
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    use_time_delta: bool = True,
    graph_pool: str = "mean",
    grad_clip: float = 1.0,
    dim_filter=None,
) -> float:
    """
    Run one training epoch over pre-batched subject lists.

    Parameters
    ----------
    batch_list : list of mini-batches, each mini-batch is a list of subject dicts.
    criterion : BCEWithLogitsLoss (or similar).

    Returns
    -------
    mean_loss : float
    """
    model.train()
    total_loss = 0.0

    for batch in batch_list:
        packed, labels, _ = encode_batch_sequences(
            batch, model, device,
            use_time_delta=use_time_delta,
            graph_pool=graph_pool,
            dim_filter=dim_filter,
        )
        # encode_batch_sequences calls model.eval() internally for encoder;
        # re-enable training mode for the full model after encoding
        model.train()

        logits = model(packed)            # (B,)
        loss   = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), grad_clip)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(batch_list), 1)


@torch.no_grad()
def evaluate(
    model: "torch.nn.Module",
    batch_list: List[List[dict]],
    device: torch.device,
    use_time_delta: bool = True,
    graph_pool: str = "mean",
    threshold: float = 0.5,
    shuffle_order: bool = False,
    shuffle_rng=None,
    dim_filter=None,
) -> Dict:
    """
    Evaluate model on a list of mini-batches.

    v2 kwargs (used by SANITY_LSTM_CHECKS):
        shuffle_order : permute visits per subject at eval-time. Δt rides along
            with its original visit so the Δt marginal distribution is
            preserved while temporal order is destroyed.
        shuffle_rng   : optional np.random.Generator for determinism.
        dim_filter    : FDR-based latent-dim selection (forwarded).

    Returns
    -------
    dict with keys: auc, sensitivity, specificity, f1, probs, targets,
                    subject_ids, n_scans
    """
    model.eval()
    all_probs:   List[float] = []
    all_targets: List[int]   = []
    all_sids:    List[str]   = []
    all_nscans:  List[int]   = []

    for batch in batch_list:
        # encode_batch_sequences sorts the batch internally; mirror that here
        # so that returned subject_ids align row-by-row with probs.
        sorted_batch = sorted(batch, key=lambda b: len(b["graphs"]), reverse=True)
        all_sids.extend([b.get("subject_id", "") for b in sorted_batch])
        all_nscans.extend([len(b["graphs"]) for b in sorted_batch])

        packed, labels, _ = encode_batch_sequences(
            batch, model, device,
            use_time_delta=use_time_delta,
            graph_pool=graph_pool,
            dim_filter=dim_filter,
            shuffle_order=shuffle_order,
            shuffle_rng=shuffle_rng,
        )
        logits = model(packed)
        probs  = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_targets.extend(labels.cpu().numpy().astype(int).tolist())

    probs_arr   = np.array(all_probs)
    targets_arr = np.array(all_targets)
    preds_arr   = (probs_arr >= threshold).astype(int)

    auc = roc_auc_score(targets_arr, probs_arr) if len(np.unique(targets_arr)) > 1 else 0.0

    if len(np.unique(targets_arr)) > 1:
        tn, fp, fn, tp = confusion_matrix(targets_arr, preds_arr).ravel()
    else:
        tn = fp = fn = tp = 0

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1   = f1_score(targets_arr, preds_arr, zero_division=0)

    # Youden threshold
    if len(np.unique(targets_arr)) > 1:
        fpr, tpr, thrs = roc_curve(targets_arr, probs_arr)
        j_idx = np.argmax(tpr - fpr)
        best_thr = float(thrs[j_idx])
    else:
        best_thr = threshold

    return {
        "auc":         float(auc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "f1":          float(f1),
        "best_threshold": best_thr,
        "probs":       probs_arr,
        "targets":     targets_arr,
        "subject_ids": np.array(all_sids),
        "n_scans":     np.array(all_nscans),
    }


def make_batches(items: List[dict], batch_size: int, shuffle: bool = True) -> List[List[dict]]:
    """Split a list of subject dicts into mini-batches."""
    if shuffle:
        idx = np.random.permutation(len(items))
        items = [items[i] for i in idx]
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
