from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


@dataclass
class LogRegCVResult:
    fold_aucs: list[float] = field(default_factory=list)
    fold_thresholds: list[float] = field(default_factory=list)  # Youden per fold
    fold_sensitivities: list[float] = field(default_factory=list)
    fold_specificities: list[float] = field(default_factory=list)
    fold_f1s: list[float] = field(default_factory=list)
    best_fold: int = 0                 # 0-indexed
    best_val_auc: float = 0.0
    best_scaler: StandardScaler | None = None
    best_clf: LogisticRegression | None = None
    youden_threshold: float = 0.5      # from best fold, val-derived — never test
    f1_oof_threshold: float = 0.5      # F1-optimal on OOF predictions
    oof_probs: np.ndarray = field(default_factory=lambda: np.array([]))
    oof_targets: np.ndarray = field(default_factory=lambda: np.array([]))


def train_logreg_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
    lr_C: float = 1.0,
    lr_max_iter: int = 2000,
    lr_class_weight: str | dict = "balanced",
) -> LogRegCVResult:
    """
    StratifiedGroupKFold CV for logistic regression on pre-computed embeddings.

    Thresholds (Youden and best-F1) are derived on validation folds only —
    never on test data. Use result.best_scaler / result.best_clf /
    result.youden_threshold when evaluating on the held-out test set.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_folds)
    result = LogRegCVResult()

    oof_probs_arr = np.full(len(y), np.nan)
    oof_targets_arr = np.full(len(y), np.nan)

    for fold, (tr_i, va_i) in enumerate(sgkf.split(X, y, groups)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_i])
        X_va = scaler.transform(X[va_i])

        clf = LogisticRegression(
            C=lr_C,
            max_iter=lr_max_iter,
            class_weight=lr_class_weight,
            random_state=seed,
        )
        clf.fit(X_tr, y[tr_i])

        va_prob = clf.predict_proba(X_va)[:, 1]
        oof_probs_arr[va_i] = va_prob
        oof_targets_arr[va_i] = y[va_i]

        if len(np.unique(y[va_i])) < 2:
            auc = threshold = sens = spec = f1 = float("nan")
        else:
            auc = float(roc_auc_score(y[va_i], va_prob))
            fpr, tpr, thrs = roc_curve(y[va_i], va_prob)
            threshold = float(thrs[int(np.argmax(tpr - fpr))])
            pred = (va_prob >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y[va_i], pred, labels=[0, 1]).ravel()
            sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            f1 = float(f1_score(y[va_i], pred, zero_division=0))

        result.fold_aucs.append(auc)
        result.fold_thresholds.append(threshold)
        result.fold_sensitivities.append(sens)
        result.fold_specificities.append(spec)
        result.fold_f1s.append(f1)

        if fold == 0 or auc >= result.best_val_auc:
            result.best_val_auc = auc
            result.best_fold = fold
            result.best_scaler = scaler
            result.best_clf = clf
            result.youden_threshold = threshold

    valid = ~np.isnan(oof_probs_arr)
    result.oof_probs = oof_probs_arr[valid]
    result.oof_targets = oof_targets_arr[valid]

    if len(np.unique(result.oof_targets)) >= 2:
        _, _, oof_thrs = roc_curve(result.oof_targets, result.oof_probs)
        oof_f1s = [
            float(f1_score(result.oof_targets, (result.oof_probs >= t).astype(int), zero_division=0))
            for t in oof_thrs
        ]
        result.f1_oof_threshold = float(oof_thrs[int(np.argmax(oof_f1s))])
    else:
        result.f1_oof_threshold = result.youden_threshold

    return result
