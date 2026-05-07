"""
metrics.py — Survival evaluation metrics: C-index, integrated Brier score,
time-dependent cumulative-dynamic AUC, log-rank tests.
Wraps scikit-survival and lifelines.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from sksurv.metrics import (
    concordance_index_censored,
    integrated_brier_score as _sksurv_ibs,
    cumulative_dynamic_auc,
)
from sksurv.util import Surv
from lifelines.statistics import multivariate_logrank_test


def to_struct_array(T: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Build a sksurv structured array Surv.from_arrays(event, time)."""
    return Surv.from_arrays(event=E.astype(bool), time=T.astype(float))


def c_index(T: np.ndarray, E: np.ndarray, risk: np.ndarray) -> float:
    """Harrell's concordance index. `risk` is higher=worse prognosis."""
    cidx, *_ = concordance_index_censored(E.astype(bool), T.astype(float), risk.astype(float))
    return float(cidx)


def integrated_brier_score(
    T_train: np.ndarray, E_train: np.ndarray,
    T_test: np.ndarray, E_test: np.ndarray,
    survival_test: np.ndarray,
    eval_times: Iterable[float],
) -> float:
    """
    Integrated Brier score over `eval_times`.
    `survival_test` shape: (n_test, n_eval_times) — survival probabilities.
    """
    times = np.asarray(list(eval_times), dtype=float)
    times = times[(times > T_test.min()) & (times < T_test.max())]
    if len(times) == 0:
        return float("nan")

    y_train = to_struct_array(T_train, E_train)
    y_test = to_struct_array(T_test, E_test)
    surv_subset = survival_test[:, : len(times)]
    return float(_sksurv_ibs(y_train, y_test, surv_subset, times))


def time_dependent_auc(
    T_train: np.ndarray, E_train: np.ndarray,
    T_test: np.ndarray, E_test: np.ndarray,
    risk: np.ndarray,
    times: Iterable[float] = (24, 36, 60),
) -> dict[int, float]:
    """
    sksurv cumulative-dynamic AUC at each requested time.
    Returns {t: auc} dict; missing/invalid times → NaN.
    """
    y_train = to_struct_array(T_train, E_train)
    y_test = to_struct_array(T_test, E_test)
    times_arr = np.asarray(list(times), dtype=float)

    valid_mask = (times_arr > T_test.min()) & (times_arr < T_test.max())
    out: dict[int, float] = {int(t): float("nan") for t in times_arr}
    if not valid_mask.any():
        return out

    valid_times = times_arr[valid_mask]
    try:
        auc, _ = cumulative_dynamic_auc(y_train, y_test, risk.astype(float), valid_times)
        for t, a in zip(valid_times, auc):
            out[int(t)] = float(a)
    except Exception as exc:
        print(f"WARNING: cumulative_dynamic_auc failed: {exc}")
    return out


def log_rank_strata(T: np.ndarray, E: np.ndarray, groups: np.ndarray) -> dict:
    """Multivariate log-rank test across `groups`. Returns dict with p-value & test stat."""
    result = multivariate_logrank_test(T, groups, E)
    return {
        "p_value": float(result.p_value),
        "test_statistic": float(result.test_statistic),
        "summary": str(result.summary),
    }


def evaluate_model(
    model,
    X_train: np.ndarray, T_train: np.ndarray, E_train: np.ndarray,
    X_test: np.ndarray, T_test: np.ndarray, E_test: np.ndarray,
    eval_times: Iterable[float] = (12, 24, 36, 48, 60, 72),
) -> dict:
    """
    Compute the standard metric bundle for a fitted SurvivalModel.
    Returns dict: {c_index, ibs, auc: {24:..., 36:..., 60:...}}
    """
    risk = model.predict_risk(X_test)
    eval_times_arr = np.asarray(list(eval_times), dtype=float)
    surv_test = model.predict_survival(X_test, eval_times_arr)

    try:
        ibs = integrated_brier_score(T_train, E_train, T_test, E_test, surv_test, eval_times_arr)
    except (ValueError, Exception):
        ibs = float("nan")

    out: dict = {
        "c_index": c_index(T_test, E_test, risk),
        "ibs": ibs,
        "auc": time_dependent_auc(T_train, E_train, T_test, E_test, risk, times=(24, 36, 60)),
    }
    return out
