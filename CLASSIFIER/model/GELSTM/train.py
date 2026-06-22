"""
GELSTM/train.py — Training and evaluation loops for GELSTMClassifier.

Public entry points:
    * ``make_batches(items, batch_size, shuffle=True, rng=None)`` — pre-batches subject dicts.
    * ``train_epoch(model, batch_list, optimizer, criterion, device, *, eval_cfg=None, grad_clip=1.0)`` — one epoch.
    * ``evaluate(model, batch_list, device, *, eval_cfg=None)`` — eval with optional Youden's J thresholding.
    * ``train_model(model, train_batches, val_batches, cfg, eval_cfg, device, *, rng=None, criterion=None, save_path=None)`` — full training loop with early stopping and full-state checkpointing.

Reproducibility contract: callers should pass an explicit ``rng`` (e.g.
``make_rng(SEED)`` from ``CLASSIFIER.common.seeding``) into ``make_batches``
and ``train_model``. Falling back to global ``np.random`` is deprecated.
"""
from __future__ import annotations

import copy
import warnings
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve

from CLASSIFIER.configs.gelstm import EvalConfig, GELSTMTrainConfig

from .utils import encode_batch_sequences


def _eval_cfg(eval_cfg: Optional[EvalConfig]) -> EvalConfig:
    return eval_cfg if eval_cfg is not None else EvalConfig()


def _eval_cfg_to_dict(eval_cfg: EvalConfig) -> Dict:
    """Serialize EvalConfig for checkpoint storage, skipping the live RNG object."""
    return {
        "use_time_delta":   eval_cfg.use_time_delta,
        "zero_time_delta":  eval_cfg.zero_time_delta,
        "graph_pool":       eval_cfg.graph_pool,
        "dim_filter":       eval_cfg.dim_filter,
        "shuffle_order":    eval_cfg.shuffle_order,
        "threshold_mode":   eval_cfg.threshold_mode,
        "fixed_threshold":  eval_cfg.fixed_threshold,
    }


def train_epoch(
    model: "torch.nn.Module",
    batch_list: List[List[dict]],
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    use_time_delta: Optional[bool] = None,
    graph_pool: Optional[str] = None,
    grad_clip: float = 1.0,
    dim_filter=None,
    *,
    eval_cfg: Optional[EvalConfig] = None,
) -> float:
    """Run one training epoch over pre-batched subject lists.

    Notes
    -----
    Legacy positional kwargs (``use_time_delta``, ``graph_pool``, ``dim_filter``)
    are still accepted for back-compat with notebooks that pre-date the
    ``EvalConfig`` introduction. Prefer passing ``eval_cfg=EvalConfig(...)``.

    ``encode_batch_sequences`` uses an internal ``eval_mode`` context manager
    that restores the caller's training state, so no defensive ``model.train()``
    is needed after each batch.
    """
    cfg = _eval_cfg(eval_cfg)
    if use_time_delta is not None:
        cfg = EvalConfig(**{**_eval_cfg_to_dict(cfg), "use_time_delta": use_time_delta})
    if graph_pool is not None:
        cfg = EvalConfig(**{**_eval_cfg_to_dict(cfg), "graph_pool": graph_pool})
    if dim_filter is not None:
        cfg = EvalConfig(**{**_eval_cfg_to_dict(cfg), "dim_filter": dim_filter})
    model.train()
    total_loss = 0.0

    for batch in batch_list:
        packed, labels, _ = encode_batch_sequences(
            batch, model, device,
            use_time_delta=cfg.use_time_delta,
            zero_time_delta=cfg.zero_time_delta,
            graph_pool=cfg.graph_pool,
            dim_filter=cfg.dim_filter,
        )

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
    use_time_delta: Optional[bool] = None,
    zero_time_delta: Optional[bool] = None,
    graph_pool: Optional[str] = None,
    threshold: Optional[float] = None,
    shuffle_order: Optional[bool] = None,
    shuffle_rng=None,
    dim_filter=None,
    *,
    eval_cfg: Optional[EvalConfig] = None,
) -> Dict:
    """Evaluate model on a list of mini-batches.

    Thresholding
    ------------
    If ``eval_cfg.threshold_mode == "youden"`` (default), the Youden's J
    threshold is computed from the ROC curve and used to derive
    ``preds_arr``, ``sensitivity``, ``specificity``, and ``f1``. Otherwise
    ``eval_cfg.fixed_threshold`` is used.

    Legacy positional kwargs are accepted for back-compat with pre-EvalConfig
    notebooks; passing ``threshold=`` switches to fixed-threshold mode with the
    given value. Prefer ``eval_cfg=EvalConfig(...)``.
    """
    cfg = _eval_cfg(eval_cfg)
    if any(x is not None for x in (use_time_delta, zero_time_delta, graph_pool,
                                   threshold, shuffle_order, shuffle_rng, dim_filter)):
        overrides = {}
        if use_time_delta is not None:  overrides["use_time_delta"]  = use_time_delta
        if zero_time_delta is not None: overrides["zero_time_delta"] = zero_time_delta
        if graph_pool is not None:      overrides["graph_pool"]      = graph_pool
        if shuffle_order is not None:   overrides["shuffle_order"]   = shuffle_order
        if shuffle_rng is not None:     overrides["shuffle_rng"]     = shuffle_rng
        if dim_filter is not None:      overrides["dim_filter"]      = dim_filter
        if threshold is not None:
            overrides["threshold_mode"]  = "fixed"
            overrides["fixed_threshold"] = threshold
        cfg = EvalConfig(**{**_eval_cfg_to_dict(cfg), **overrides,
                            "shuffle_rng": overrides.get("shuffle_rng", cfg.shuffle_rng)})
    model.eval()

    all_probs:   List[float] = []
    all_targets: List[int]   = []
    all_sids:    List[str]   = []
    all_nscans:  List[int]   = []

    for batch in batch_list:
        sorted_batch = sorted(batch, key=lambda b: len(b["graphs"]), reverse=True)
        all_sids.extend([b.get("subject_id", "") for b in sorted_batch])
        all_nscans.extend([len(b["graphs"]) for b in sorted_batch])

        packed, labels, _ = encode_batch_sequences(
            batch, model, device,
            use_time_delta=cfg.use_time_delta,
            zero_time_delta=cfg.zero_time_delta,
            graph_pool=cfg.graph_pool,
            dim_filter=cfg.dim_filter,
            shuffle_order=cfg.shuffle_order,
            shuffle_rng=cfg.shuffle_rng,
        )
        logits = model(packed)
        probs  = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_targets.extend(labels.cpu().numpy().astype(int).tolist())

    probs_arr   = np.array(all_probs)
    targets_arr = np.array(all_targets)
    has_both_classes = len(np.unique(targets_arr)) > 1

    if has_both_classes:
        fpr, tpr, thrs = roc_curve(targets_arr, probs_arr)
        j_idx = int(np.argmax(tpr - fpr))
        best_thr = float(thrs[j_idx])
        auc = float(roc_auc_score(targets_arr, probs_arr))
    else:
        best_thr = cfg.fixed_threshold
        auc = 0.0

    if cfg.threshold_mode == "youden":
        threshold_used = best_thr
    else:
        threshold_used = cfg.fixed_threshold

    preds_arr = (probs_arr >= threshold_used).astype(int)

    if has_both_classes:
        tn, fp, fn, tp = confusion_matrix(targets_arr, preds_arr).ravel()
    else:
        tn = fp = fn = tp = 0

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1   = f1_score(targets_arr, preds_arr, zero_division=0)

    return {
        "auc":             float(auc),
        "sensitivity":     float(sens),
        "specificity":     float(spec),
        "f1":              float(f1),
        "best_threshold":  float(best_thr),
        "threshold_used":  float(threshold_used),
        "probs":           probs_arr,
        "targets":         targets_arr,
        "preds":           preds_arr,
        "subject_ids":     np.array(all_sids),
        "n_scans":         np.array(all_nscans),
    }


def make_batches(
    items: List[dict],
    batch_size: int,
    shuffle: bool = True,
    rng: "np.random.Generator | None" = None,
) -> List[List[dict]]:
    """Split a list of subject dicts into mini-batches.

    Pass an explicit ``rng`` (e.g. ``np.random.default_rng(SEED)``) for
    reproducibility. Calling without ``rng`` while ``shuffle=True`` emits a
    ``DeprecationWarning`` and falls back to global ``np.random`` for
    back-compat with old call sites.
    """
    if shuffle:
        if rng is None:
            warnings.warn(
                "make_batches called with shuffle=True and no rng; falling back to "
                "global np.random.permutation. Pass rng=np.random.default_rng(SEED) "
                "for reproducible shuffles.",
                DeprecationWarning,
                stacklevel=2,
            )
            idx = np.random.permutation(len(items))
        else:
            idx = rng.permutation(len(items))
        items = [items[i] for i in idx]
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def train_model(
    model: "torch.nn.Module",
    train_batches: List[List[dict]],
    val_batches: List[List[dict]],
    cfg: GELSTMTrainConfig,
    eval_cfg: EvalConfig,
    device: torch.device,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    criterion: Optional[torch.nn.Module] = None,
    rng: "np.random.Generator | None" = None,
    save_path: Optional[str] = None,
    log_fn=None,
) -> Tuple[Dict, Dict]:
    """Full training loop with early stopping and full-state checkpointing.

    Returns
    -------
    (best_checkpoint_dict, history) — the checkpoint dict mirrors what was
    written to ``save_path``.
    """
    if optimizer is None:
        optimizer = torch.optim.Adam(
            model.get_trainable_params(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
    if criterion is None:
        criterion = torch.nn.BCEWithLogitsLoss()

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=cfg.lr_factor, patience=cfg.lr_patience, min_lr=cfg.lr_min
        )
        if cfg.use_scheduler
        else None
    )

    best_val_auc = -1.0
    best_state = None
    best_epoch = -1
    best_threshold = cfg.fixed_threshold
    epochs_no_improve = 0
    history = {"train_loss": [], "val_auc": [], "val_f1": [], "learning_rate": [], "best_threshold": []}

    for epoch in range(cfg.epochs):
        train_loss = train_epoch(
            model, train_batches, optimizer, criterion, device,
            eval_cfg=eval_cfg, grad_clip=cfg.grad_clip,
        )
        val_metrics = evaluate(model, val_batches, device, eval_cfg=eval_cfg)

        history["train_loss"].append(train_loss)
        history["val_auc"].append(val_metrics["auc"])
        history["val_f1"].append(val_metrics["f1"])
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])
        history["best_threshold"].append(val_metrics["best_threshold"])

        if log_fn is not None:
            log_fn({"epoch": epoch, "train_loss": train_loss, **{
                k: v for k, v in val_metrics.items()
                if k in ("auc", "f1", "sensitivity", "specificity", "best_threshold", "threshold_used")
            }})

        if scheduler is not None:
            scheduler.step(val_metrics["auc"])

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_threshold = val_metrics["best_threshold"]
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= cfg.early_stopping_patience:
            break

    checkpoint = {
        "model_state_dict": best_state if best_state is not None else copy.deepcopy(model.state_dict()),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": best_epoch,
        "val_auc": float(best_val_auc),
        "best_threshold": float(best_threshold),
        "rng_state": rng.bit_generator.state if rng is not None else None,
        "torch_rng_state": torch.get_rng_state(),
        "config": asdict(cfg),
        "eval_config": _eval_cfg_to_dict(eval_cfg),
    }

    if save_path is not None:
        torch.save(checkpoint, save_path)

    return checkpoint, history
