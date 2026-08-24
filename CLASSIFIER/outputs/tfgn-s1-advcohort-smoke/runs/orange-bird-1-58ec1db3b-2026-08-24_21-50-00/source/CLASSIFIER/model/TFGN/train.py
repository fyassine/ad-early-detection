"""TFGN/train.py — Training and evaluation loops for TFGNClassifier.

Public entry points:
    * ``make_batches(items, batch_size, shuffle=True, rng=None)`` — pre-batches TFGNItems.
    * ``train_epoch(model, batch_list, optimizer, criterion, device, *, cfg, grad_clip=1.0, epoch=0)`` — one epoch; returns ``(loss, component_means)``.
    * ``evaluate(model, batch_list, device, *, eval_cfg=None)`` — eval returning the standard key bundle.

Reproducibility contract: callers should pass an explicit ``rng`` into ``make_batches``.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve

from CLASSIFIER.configs.tfgn import TFGNEvalConfig, TFGNTrainConfig
from CLASSIFIER.model.TFGN.dataset import TFGNItem
from CLASSIFIER.model.TFGN.losses import (
    centrality_anchor_mse,
    change_mask_bce,
    cohort_adversarial_bce,
    delta_a_mse_loss,
    drift_anchor_mse,
    free_bits_kl,
    gate_sparsity_kl,
)

# Binary cohort label for the adversarial head (model/TFGN/layers.py::CohortAdversaryHead).
# Only these two cohorts ever appear in pooled training -- OASIS-3 is excluded from both
# pretraining and downstream training by design (DOCS/flipped/PLAN.md "Decisions already
# fixed": kept fully external), so an item with any other cohort tag here is a leakage
# bug, not a case to handle gracefully.
COHORT_LABEL_MAP = {"adni": 0.0, "delcode": 1.0}


def _forward_item(
    model, item: TFGNItem, device: torch.device, *, cohort_adv_lambda: float = 1.0
) -> dict:
    """Move an item's tensors to device and run the model forward.

    ``cohort_adv_lambda`` only affects models built with
    ``cohort_conditioning="adversarial"`` (the model ignores it otherwise); see
    ``train_epoch``'s warmup computation for where the per-epoch value comes from.
    """
    X = item.X.to(device)
    log_dt = item.log_dt.to(device)
    A0_ei = item.A0_edge_index.to(device)
    A0_ea = item.A0_edge_attr.to(device) if item.A0_edge_attr is not None else None
    cond = torch.tensor(
        [[item.age, float(item.sex)]], dtype=torch.float32, device=device
    )
    return model(X, log_dt, A0_ei, A0_ea, cond, cohort_adv_lambda=cohort_adv_lambda)


def make_batches(
    items: List[TFGNItem],
    batch_size: int,
    shuffle: bool = True,
    rng: "np.random.Generator | None" = None,
) -> List[List[TFGNItem]]:
    """Split a list of TFGNItems into mini-batches.

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
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def train_epoch(
    model: "torch.nn.Module",
    batch_list: List[List[TFGNItem]],
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    *,
    cfg: TFGNTrainConfig,
    grad_clip: float = 1.0,
    epoch: int = 0,
) -> tuple[float, Dict[str, float]]:
    """Run one training epoch over pre-batched TFGNItems.

    Returns ``(epoch_mean_total_loss, component_means)`` where
    ``component_means`` is the per-item mean of each loss term that
    contributed this epoch (``bce``, ``recon``, ``kl``, ``gate_sparsity``,
    ``drift``, ``cent``, ``cohort_adv``) — additive instrumentation only, the
    optimized objective is unchanged. A term a given config never activates
    (e.g. ``recon`` when ``cfg.recon_target == "none"``) is reported as ``0.0``.
    """
    model.train()
    total_loss = 0.0
    num_items = 0
    component_totals: Dict[str, float] = {
        "bce": 0.0,
        "recon": 0.0,
        "kl": 0.0,
        "gate_sparsity": 0.0,
        "drift": 0.0,
        "cent": 0.0,
        "cohort_adv": 0.0,
    }

    # Beta-KL warmup: linearly ramp from 0 to beta_kl over beta_warmup_epochs
    if cfg.beta_warmup_epochs > 0 and epoch < cfg.beta_warmup_epochs:
        beta_kl_eff = cfg.beta_kl * (epoch / cfg.beta_warmup_epochs)
    else:
        beta_kl_eff = cfg.beta_kl

    # Cohort-adversary gradient-reversal warmup: same ramp shape as beta-KL,
    # over its own schedule -- starting the reversal at full strength before
    # the classifier head has learned anything useful to reverse just adds
    # noise (standard DANN practice).
    if cfg.cohort_adv_warmup_epochs > 0 and epoch < cfg.cohort_adv_warmup_epochs:
        cohort_adv_lambda_eff = cfg.cohort_adv_lambda * (epoch / cfg.cohort_adv_warmup_epochs)
    else:
        cohort_adv_lambda_eff = cfg.cohort_adv_lambda

    for batch in batch_list:
        optimizer.zero_grad()
        batch_loss = torch.tensor(0.0, device=device, requires_grad=True)

        for item in batch:
            out = _forward_item(
                model, item, device, cohort_adv_lambda=cohort_adv_lambda_eff
            )
            logits = out["logits"]
            s_topo = out["s_topo"]
            gate_scores = out["gate_scores"]
            logvar = out["logvar"]
            mu_raw = out["mu_raw"]
            recon_logits = out["recon_logits"]
            cohort_logit = out["cohort_logit"]

            label_t = torch.tensor(
                [float(item.label)], dtype=torch.float32, device=device
            )
            loss = criterion(logits, label_t)
            component_totals["bce"] += loss.item()

            # Cohort-adversary loss (gradient-reversal head, PLAN.md section L).
            # OASIS-3 must never reach this path -- fail loud rather than
            # silently mis-labeling or skipping an unmapped cohort.
            if cfg.cohort_conditioning == "adversarial" and cohort_logit is not None:
                if item.cohort not in COHORT_LABEL_MAP:
                    raise ValueError(
                        f"cohort_conditioning='adversarial' but item {item.subject_id!r} "
                        f"has cohort={item.cohort!r}, not one of {sorted(COHORT_LABEL_MAP)}. "
                        "OASIS-3 (or any unmapped cohort) must never train through the "
                        "adversarial head -- it must stay fully external."
                    )
                cohort_label_t = torch.tensor(
                    [COHORT_LABEL_MAP[item.cohort]], dtype=torch.float32, device=device
                )
                cohort_term = cohort_adversarial_bce(cohort_logit, cohort_label_t)
                loss = loss + cohort_term
                component_totals["cohort_adv"] += cohort_term.item()

            # Gate sparsity + node-level drift anchor losses
            if cfg.use_gate and gate_scores is not None:
                sparse_term = cfg.lambda_sparse * gate_sparsity_kl(
                    gate_scores, cfg.gate_rho
                )
                loss = loss + sparse_term
                component_totals["gate_sparsity"] += sparse_term.item()

                da_target = item.drift_anchor.to(device)
                drift_term = cfg.lambda_drift * drift_anchor_mse(
                    gate_scores.squeeze(-1), da_target
                )
                loss = loss + drift_term
                component_totals["drift"] += drift_term.item()

            # Centrality anchor regularizer on topological saliency s_topo
            if cfg.lambda_cent > 0.0 and s_topo is not None:
                cent = item.strength_centrality.to(device)
                cent_term = cfg.lambda_cent * centrality_anchor_mse(
                    s_topo, cent
                )
                loss = loss + cent_term
                component_totals["cent"] += cent_term.item()

            # Reconstruction loss
            if cfg.recon_target != "none" and recon_logits is not None:
                if cfg.recon_target == "delta_a_topk":
                    cm = item.change_mask.to(device)
                    pw = (1.0 - cfg.change_mask_kappa) / cfg.change_mask_kappa
                    recon_term = cfg.lambda_recon * change_mask_bce(
                        recon_logits, cm, pw
                    )
                elif cfg.recon_target == "delta_a_mse":
                    delta_A = (item.X[-1] - item.X[0]).to(device)
                    recon_term = cfg.lambda_recon * delta_a_mse_loss(
                        recon_logits, delta_A / 2.0
                    )
                elif cfg.recon_target == "a_last":
                    A_last = item.X[-1].to(device)
                    recon_term = cfg.lambda_recon * change_mask_bce(
                        recon_logits, (A_last + 1.0) / 2.0, 1.0
                    )
                loss = loss + recon_term
                component_totals["recon"] += recon_term.item()

            # Variational KL loss (evaluated on mu_raw before FiLM conditioning)
            if mu_raw is not None and logvar is not None:
                kl_term = beta_kl_eff * free_bits_kl(
                    mu_raw, logvar, cfg.free_bits
                )
                loss = loss + kl_term
                component_totals["kl"] += kl_term.item()

            batch_loss = batch_loss + loss
            num_items += 1

        batch_loss = batch_loss / len(batch)
        batch_loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.get_trainable_params(), grad_clip
            )
        optimizer.step()

        total_loss += batch_loss.item() * len(batch)

    n = max(num_items, 1)
    component_means = {k: v / n for k, v in component_totals.items()}
    return total_loss / n, component_means


@torch.no_grad()
def evaluate(
    model: "torch.nn.Module",
    batch_list: List[List[TFGNItem]],
    device: torch.device,
    *,
    eval_cfg: Optional[TFGNEvalConfig] = None,
) -> Dict:
    """Evaluate model on a list of mini-batches.

    Returns the standard key bundle matching ``model/GELSTM/train.py::evaluate``:
    auc, sensitivity, specificity, f1, best_threshold, threshold_used,
    probs, targets, preds, subject_ids, n_scans.
    """
    if eval_cfg is None:
        eval_cfg = TFGNEvalConfig()

    model.eval()

    all_probs: List[float] = []
    all_targets: List[int] = []
    all_sids: List[str] = []
    all_nscans: List[int] = []

    for batch in batch_list:
        for item in batch:
            all_sids.append(item.subject_id)
            all_nscans.append(item.n_visits)
            all_targets.append(item.label)

            out = _forward_item(model, item, device)
            prob = torch.sigmoid(out["logits"]).cpu().numpy()
            all_probs.append(float(prob.item()))

    probs_arr = np.array(all_probs)
    targets_arr = np.array(all_targets)
    has_both_classes = len(np.unique(targets_arr)) > 1

    if has_both_classes:
        fpr, tpr, thrs = roc_curve(targets_arr, probs_arr)
        j_idx = int(np.argmax(tpr - fpr))
        best_thr = float(thrs[j_idx])
        auc = float(roc_auc_score(targets_arr, probs_arr))
    else:
        best_thr = eval_cfg.fixed_threshold
        auc = 0.0

    if eval_cfg.threshold_mode == "youden":
        threshold_used = best_thr
    else:
        threshold_used = eval_cfg.fixed_threshold

    preds_arr = (probs_arr >= threshold_used).astype(int)

    if has_both_classes:
        tn, fp, fn, tp = confusion_matrix(targets_arr, preds_arr).ravel()
    else:
        tn = fp = fn = tp = 0

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(targets_arr, preds_arr, zero_division=0)

    return {
        "auc": float(auc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "f1": float(f1),
        "best_threshold": float(best_thr),
        "threshold_used": float(threshold_used),
        "probs": probs_arr,
        "targets": targets_arr,
        "preds": preds_arr,
        "subject_ids": np.array(all_sids),
        "n_scans": np.array(all_nscans),
    }
