"""
common/crossval.py — model-agnostic subject-level cross-validation.

This is the shared replacement for the StratifiedGroupKFold loop that used to be
copy-pasted into every longitudinal notebook (GEC-MLP, GELSTM, …). It is driven
entirely by a model's ``train_fold`` hook, so the loop itself contains no
model-specific or W&B-specific code:

    * ``Bundle``        — the duck-typed per-split container the loop iterates.
    * ``run_kfold_cv``  — the CV loop; calls ``train_fold`` once per fold.
    * ``summarize_cv``  — the mean/std/min/max table print.

The deleted ``common/validation.py::run_kfold_cv`` was GEC-specific (it built the
model, loaders and class weights itself and imported ``wandb`` directly). This
version inverts that: the caller supplies a ``train_fold`` callable and an
optional ``log_fn`` sink, keeping the loop reusable and W&B-free.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_curve, f1_score


@dataclass
class Bundle:
    """One split's encoded data, indexable by subject.

    The shared cells rely only on this interface, so each model's ``prepare_data``
    hook can return a ``Bundle`` regardless of how it encodes a subject:

      labels : per-subject labels {0,1}, aligned with ``items``.
      groups : per-subject ids (the StratifiedGroupKFold groups).
      items  : per-subject records (model-specific payload; also used by the
               trajectory / visit-truncation hooks). Each item should expose at
               least ``subject_id``, ``label`` and ``n_scans``.
    """

    labels: List[int]
    groups: List[Any]
    items: List[Any]

    def subset(self, idx: Sequence[int]) -> "Bundle":
        """Return a new Bundle restricted to the positions in ``idx``."""
        idx = list(idx)
        return Bundle(
            [self.labels[i] for i in idx],
            [self.groups[i] for i in idx],
            [self.items[i] for i in idx],
        )

    def __len__(self) -> int:
        return len(self.items)


@dataclass
class CVResult:
    """Everything a notebook needs after cross-validation.

    ``best_threshold`` is the validation-derived (Youden) threshold of the best
    fold; ``best_f1_threshold`` is the F1-optimal threshold over the pooled
    out-of-fold predictions. Both are validation-side — see
    ``common.thresholds.select_oof_threshold`` for choosing between them.
    """

    cv_results: dict
    oof_probs: np.ndarray
    oof_targets: np.ndarray
    oof_sids: list
    best_fold: int
    best_val_auc: float
    best_model_state: Any
    best_threshold: float
    best_f1_threshold: float


def run_kfold_cv(
    bundle: Bundle,
    train_fold: Callable[..., dict],
    cfg: Any,
    *,
    n_folds: int,
    rng: "np.random.Generator | None",
    device: Any,
    log_fn: Optional[Callable[[dict], None]] = None,
) -> CVResult:
    """Run StratifiedGroupKFold subject-level CV driven by a ``train_fold`` hook.

    Parameters
    ----------
    bundle : Bundle
        The CV pool (train+val). Split is stratified on ``bundle.labels`` and
        grouped on ``bundle.groups`` so no subject crosses the fold boundary.
    train_fold : callable
        ``train_fold(bundle_tr, bundle_va, cfg, *, rng, device) -> dict`` with keys:
        ``state_dict``, ``val_metrics`` ({'auc','sensitivity','specificity','f1'}),
        ``best_threshold``, ``oof_probs``, ``oof_targets``, ``oof_sids``.
    cfg : Any
        Opaque per-model training config, forwarded to ``train_fold`` unchanged.
    n_folds : int
        Number of folds.
    rng : np.random.Generator or None
        Forwarded to ``train_fold`` for reproducible within-fold shuffling
        (per ``.claude/rules/seeding.md``). The split itself is deterministic.
    device : Any
        Forwarded to ``train_fold``.
    log_fn : callable, optional
        Per-fold metric sink, e.g. ``lambda d: tracking.log_metrics(run, d)``.
        Keeps this module free of any W&B import.

    Returns
    -------
    CVResult
    """
    cv_results: dict = {
        "fold": [],
        "val_auc": [],
        "val_sensitivity": [],
        "val_specificity": [],
        "val_f1": [],
        "best_threshold": [],
    }
    oof_probs: List[float] = []
    oof_targets: List[int] = []
    oof_sids: List[Any] = []

    best_val_auc, best_fold, best_model_state = 0.0, -1, None
    best_threshold_overall = 0.5

    sgkf = StratifiedGroupKFold(n_splits=n_folds)
    for fold, (tr_idx, va_idx) in enumerate(
        sgkf.split(bundle.items, bundle.labels, groups=bundle.groups)
    ):
        print("=" * 55)
        print(f"Fold {fold + 1}/{n_folds}  train={len(tr_idx)}  val={len(va_idx)}")

        fold_out = train_fold(
            bundle.subset(tr_idx), bundle.subset(va_idx), cfg, rng=rng, device=device
        )
        vm = fold_out["val_metrics"]

        oof_probs.extend(list(fold_out["oof_probs"]))
        oof_targets.extend(list(fold_out["oof_targets"]))
        oof_sids.extend(list(fold_out["oof_sids"]))

        cv_results["fold"].append(fold + 1)
        cv_results["val_auc"].append(vm["auc"])
        cv_results["val_sensitivity"].append(vm["sensitivity"])
        cv_results["val_specificity"].append(vm["specificity"])
        cv_results["val_f1"].append(vm["f1"])
        cv_results["best_threshold"].append(fold_out["best_threshold"])

        if log_fn is not None:
            log_fn({"fold": fold + 1, "val_auc": vm["auc"], "val_f1": vm["f1"]})

        print(
            f"  AUC={vm['auc']:.4f}  sens={vm['sensitivity']:.3f}  "
            f"spec={vm['specificity']:.3f}  F1={vm['f1']:.3f}"
        )

        if vm["auc"] > best_val_auc:
            best_val_auc, best_fold = vm["auc"], fold + 1
            best_model_state = fold_out["state_dict"]
            best_threshold_overall = fold_out["best_threshold"]

    oof_arr = np.asarray(oof_probs, dtype=float)
    oof_tgt = np.asarray(oof_targets, dtype=int)

    if len(np.unique(oof_tgt)) > 1:
        _, _, thrs = roc_curve(oof_tgt, oof_arr)
        best_f1_thr = float(
            thrs[
                int(
                    np.argmax(
                        [
                            f1_score(oof_tgt, (oof_arr >= t).astype(int), zero_division=0)
                            for t in thrs
                        ]
                    )
                )
            ]
        )
    else:
        best_f1_thr = best_threshold_overall

    print(f"\nBest fold: {best_fold}  CV AUC={best_val_auc:.4f}")
    print(f"Youden thr={best_threshold_overall:.4f}  OOF-F1 thr={best_f1_thr:.4f}")

    return CVResult(
        cv_results=cv_results,
        oof_probs=oof_arr,
        oof_targets=oof_tgt,
        oof_sids=oof_sids,
        best_fold=best_fold,
        best_val_auc=float(best_val_auc),
        best_model_state=best_model_state,
        best_threshold=float(best_threshold_overall),
        best_f1_threshold=float(best_f1_thr),
    )


def summarize_cv(cv_results: dict) -> None:
    """Print the per-metric mean/std/min/max table across folds."""
    print("Cross-Validation Summary:")
    print("=" * 60)
    print(f"{'Metric':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 60)
    for m in ["val_auc", "val_sensitivity", "val_specificity", "val_f1"]:
        v = cv_results[m]
        print(
            f"{m:<20} {np.mean(v):>10.4f} {np.std(v):>10.4f} "
            f"{np.min(v):>10.4f} {np.max(v):>10.4f}"
        )
