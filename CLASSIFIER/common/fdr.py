"""
Fisher's Discriminant Ratio (FDR) based latent-dimension selection.

Used to project a high-dimensional embedding (e.g. the 64-D GAAE latent) onto
the top-K most class-discriminative dimensions before feeding a downstream
classifier.

LEAKAGE CONTRACT
----------------
This function is *supervised* feature selection: it reads `labels`. To keep a
cross-validation estimate honest, it MUST be called with embeddings/labels from
the TRAINING partition only — never the validation or test rows. Selecting
dimensions on the full pool (or with test data included) inflates the estimate.
The caller is responsible for slicing to the training fold before calling.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def compute_fdr_scores(embs: np.ndarray, labels: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Per-dimension Fisher's Discriminant Ratio for a binary label.

        FDR_j = (mu1_j - mu0_j)^2 / (var1_j + var0_j + eps)

    Parameters
    ----------
    embs : (N, D) array of embeddings.
    labels : (N,) binary array (0/1).
    eps : variance floor to avoid divide-by-zero.

    Returns
    -------
    scores : (D,) array of FDR scores, one per dimension.
    """
    embs = np.asarray(embs, dtype=float)
    labels = np.asarray(labels).astype(int)
    if embs.ndim != 2:
        raise ValueError(f"embs must be 2-D (N, D); got shape {embs.shape}")
    if labels.shape[0] != embs.shape[0]:
        raise ValueError(
            f"labels length {labels.shape[0]} != n_rows {embs.shape[0]}"
        )
    classes = np.unique(labels)
    if not np.array_equal(classes, np.array([0, 1])):
        raise ValueError(
            f"FDR selection requires both classes 0 and 1 present; got {classes.tolist()}"
        )

    pos = embs[labels == 1]
    neg = embs[labels == 0]
    mu_diff_sq = (pos.mean(axis=0) - neg.mean(axis=0)) ** 2
    var_sum = pos.var(axis=0) + neg.var(axis=0) + eps
    return mu_diff_sq / var_sum


def compute_fdr_filter(
    embs: np.ndarray, labels: np.ndarray, top_k: int, eps: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rank dimensions by FDR and return the top-K indices plus the full scores.

    Parameters
    ----------
    embs : (N, D) training-fold embeddings (see leakage contract above).
    labels : (N,) binary labels for the same rows.
    top_k : number of dimensions to keep. Must satisfy 1 <= top_k <= D.

    Returns
    -------
    top_dims : (top_k,) int array of dimension indices, highest FDR first.
        Contiguous, positive-stride (safe to pass to torch indexing).
    scores : (D,) FDR score per dimension.
    """
    scores = compute_fdr_scores(embs, labels, eps=eps)
    n_dims = scores.shape[0]
    if not (1 <= top_k <= n_dims):
        raise ValueError(f"top_k must be in [1, {n_dims}]; got {top_k}")
    # argsort ascending → reverse for descending → copy() makes it contiguous
    # (torch rejects negative-stride numpy views).
    top_dims = np.argsort(scores)[::-1][:top_k].copy()
    return top_dims, scores
