"""
GEC/train.py — Training and evaluation loops for the GEC classifier.

Reproducibility contract
------------------------
Callers should build DataLoaders with a seeded ``generator`` and
``worker_init_fn`` (use :func:`build_loader` below or import the helpers from
``CLASSIFIER.common.seeding``)::

    from CLASSIFIER.common.seeding import make_torch_generator, seed_worker
    loader = DataLoader(ds, batch_size=32, shuffle=True,
                        generator=make_torch_generator(SEED),
                        worker_init_fn=seed_worker)

Batch contract
--------------
Each batch passed to :func:`train_classifier` must expose: ``x``,
``edge_index``, ``batch``, ``is_converter``, ``patient_age``, ``patient_sex``
(see :class:`CLASSIFIER.configs.gec.GECBatch`).
"""
from __future__ import annotations

import copy
from dataclasses import asdict
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from CLASSIFIER.configs.gec import GECTrainConfig
from CLASSIFIER.common.seeding import make_torch_generator, seed_worker


def build_loader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
    **loader_kwargs,
) -> DataLoader:
    """Build a DataLoader wired for reproducibility (seeded generator + worker init)."""
    generator = make_torch_generator(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        **loader_kwargs,
    )


def _forward_batch(model, batch, device):
    batch = batch.to(device)
    cond_vec = torch.stack(
        [batch.patient_age, batch.patient_sex.float()], dim=1
    ).to(device)
    logits, _ = model(batch.x, batch.edge_index, cond_vec, batch.batch)
    return logits, batch.is_converter


def _youden_threshold(labels, probs) -> float:
    if len(set(labels)) < 2:
        return 0.5
    fpr, tpr, thrs = roc_curve(labels, probs)
    j_idx = int((tpr - fpr).argmax())
    return float(thrs[j_idx])


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pos_weight,
    cfg: GECTrainConfig,
    *,
    wandb_run=None,
    rng=None,
    model_save_path: Optional[str] = None,
) -> Tuple[Dict, Dict]:
    """Train a GEC classifier.

    Parameters
    ----------
    cfg : GECTrainConfig
        All training hyperparameters (epochs, grad_clip, scheduler, etc.).
    wandb_run : optional
        A pre-initialized W&B run. If ``None``, no W&B logging happens.
    rng : optional numpy Generator
        Stored in the checkpoint for replayability.

    Returns
    -------
    (best_checkpoint_dict, history)
    """
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
        )
        if cfg.use_scheduler
        else None
    )

    best_val_auc = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = -1
    best_threshold = cfg.fixed_threshold
    epochs_no_improve = 0
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "train_f1": [], "val_f1": [],
        "train_auc": [], "val_auc": [],
        "best_threshold": [],
        "learning_rate": [],
    }

    outer_bar = tqdm(range(cfg.epochs), desc="Training Progress")
    for epoch in outer_bar:
        # --- train ---
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []

        for batch in train_loader:
            logits, labels = _forward_batch(model, batch, device)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        avg_train_loss = train_loss / max(1, len(train_loader))

        # --- val ---
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                logits, labels = _forward_batch(model, batch, device)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / max(1, len(val_loader))

        # --- choose threshold (Youden on val) ---
        if cfg.threshold_mode == "youden":
            epoch_threshold = _youden_threshold(val_labels, val_preds)
        else:
            epoch_threshold = cfg.fixed_threshold

        train_preds_binary = [1 if p > epoch_threshold else 0 for p in train_preds]
        val_preds_binary   = [1 if p > epoch_threshold else 0 for p in val_preds]

        train_acc = accuracy_score(train_labels, train_preds_binary)
        train_f1  = f1_score(train_labels, train_preds_binary, zero_division=0)
        train_auc = roc_auc_score(train_labels, train_preds) if len(set(train_labels)) > 1 else 0.0
        val_acc   = accuracy_score(val_labels, val_preds_binary)
        val_f1    = f1_score(val_labels, val_preds_binary, zero_division=0)
        val_auc   = roc_auc_score(val_labels, val_preds) if len(set(val_labels)) > 1 else 0.0

        for k, v in [
            ("train_loss", avg_train_loss), ("val_loss", avg_val_loss),
            ("train_acc", train_acc),       ("val_acc", val_acc),
            ("train_f1",  train_f1),        ("val_f1",  val_f1),
            ("train_auc", train_auc),       ("val_auc", val_auc),
            ("best_threshold", epoch_threshold),
            ("learning_rate", optimizer.param_groups[0]["lr"]),
        ]:
            history[k].append(v)

        outer_bar.set_postfix({
            "Train Loss": f"{avg_train_loss:.4f}",
            "Val Loss":   f"{avg_val_loss:.4f}",
            "Val AUC":    f"{val_auc:.4f}",
            "Thr":        f"{epoch_threshold:.3f}",
            "LR":         f"{optimizer.param_groups[0]['lr']:.2e}",
        })

        if wandb_run is not None:
            wandb_run.log({
                "train_loss": avg_train_loss, "val_loss": avg_val_loss,
                "train_acc": train_acc, "val_acc": val_acc,
                "train_f1":  train_f1,  "val_f1":  val_f1,
                "train_auc": train_auc, "val_auc": val_auc,
                "best_threshold": epoch_threshold,
                "learning_rate": optimizer.param_groups[0]["lr"],
            })

        if scheduler is not None:
            scheduler.step(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_threshold = epoch_threshold
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= cfg.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    checkpoint = {
        "model_state_dict": best_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": best_epoch,
        "val_auc": float(best_val_auc),
        "best_threshold": float(best_threshold),
        "rng_state": rng.bit_generator.state if rng is not None else None,
        "torch_rng_state": torch.get_rng_state(),
        "config": asdict(cfg),
    }

    if model_save_path is not None:
        torch.save(checkpoint, model_save_path)

    return checkpoint, history


def evaluate_classifier(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    *,
    threshold: float,
) -> Dict:
    """Evaluate on a test loader using an externally chosen threshold.

    ``threshold`` is required (no default) to prevent silent test-set
    threshold tuning. Pass the validation-derived ``best_threshold`` from
    :func:`train_classifier`.
    """
    if threshold is None:
        raise ValueError(
            "evaluate_classifier requires an explicit threshold (typically the "
            "validation-derived best_threshold). Choosing a threshold from test "
            "metrics would leak."
        )

    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            logits, labels = _forward_batch(model, batch, device)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = [1 if p > threshold else 0 for p in probs]
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1":       f1_score(all_labels, all_preds, zero_division=0),
        "auc":      roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0,
        "confusion_matrix":      confusion_matrix(all_labels, all_preds),
        "classification_report": classification_report(all_labels, all_preds, zero_division=0),
        "threshold_used": float(threshold),
        "predictions":    all_preds,
        "probabilities":  all_probs,
        "labels":         all_labels,
    }
