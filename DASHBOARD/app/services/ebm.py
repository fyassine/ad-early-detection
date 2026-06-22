"""
ebm.py — Lightweight Event-Based Model (EBM) staging.

Implements a simplified version of the AD Event-Based Model
(Fonteijn 2012; Young 2018 SuStaIn; Aksman et al. 2023 SuStaIn-AI
update) for cohort-level disease staging. Avoids the heavy
``pySuStaIn`` dependency by using a direct EBM:

  * For each biomarker B with a 'normal' reference distribution
    (CN baselines) and an 'abnormal' reference distribution (AD
    baselines), fit Gaussian likelihoods.
  * Order biomarkers by their separation strength (Hedges' g
    between CN and AD distributions, descending).
  * For a patient at any visit, the *stage* = number of biomarkers
    whose value is more likely under the abnormal distribution
    than the normal one (P(abnormal | x) > P(normal | x)).

This is the canonical EBM stage-assignment rule (Fonteijn 2012)
and is robust enough to ship without iterative model fitting.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


def _gauss_logpdf(x: float, mu: float, sigma: float) -> float:
    if sigma is None or not math.isfinite(sigma) or sigma <= 1e-9:
        return -math.inf
    z = (x - mu) / sigma
    return -0.5 * (z * z + math.log(2 * math.pi * sigma * sigma))


def _hedges_g(x: list, y: list) -> Optional[float]:
    """Hedges' g (small-sample Cohen's d). Used purely to rank biomarkers."""
    x = [v for v in x if v is not None and math.isfinite(v)]
    y = [v for v in y if v is not None and math.isfinite(v)]
    if len(x) < 2 or len(y) < 2:
        return None
    mx, my = float(np.mean(x)), float(np.mean(y))
    vx, vy = float(np.var(x, ddof=1)), float(np.var(y, ddof=1))
    pooled = math.sqrt(((len(x) - 1) * vx + (len(y) - 1) * vy) / (len(x) + len(y) - 2))
    if pooled < 1e-12:
        return None
    j = 1.0 - (3.0 / (4.0 * (len(x) + len(y)) - 9.0))
    return (mx - my) / pooled * j


def fit_ebm(
    cohort_values: dict[str, dict[str, list[float]]],
    biomarker_keys: list[str],
    abnormal_cohort: str = "ad",
    normal_cohort: str = "healthy",
) -> dict:
    """
    Fit per-biomarker normal/abnormal Gaussian likelihoods + the EBM
    sequence ordering.

    ``cohort_values[cohort][biomarker]`` -> list of values.

    Returns a JSON-serialisable dict::

        {
            "sequence":       ["abeta42", "p_tau", ...],         # ordered
            "biomarkers": {
                "abeta42":   {"mu_normal": ..., "sigma_normal": ...,
                              "mu_abnormal": ..., "sigma_abnormal": ...,
                              "direction": "decrease",  # which way is abnormal
                              "abs_g": 1.42},
                ...
            }
        }
    """
    abn = cohort_values.get(abnormal_cohort, {}) or {}
    norm = cohort_values.get(normal_cohort, {}) or {}

    biomarkers: dict[str, dict] = {}
    ranked: list[tuple[str, float]] = []
    for key in biomarker_keys:
        x_norm = [v for v in (norm.get(key) or []) if v is not None and math.isfinite(v)]
        x_abn = [v for v in (abn.get(key) or []) if v is not None and math.isfinite(v)]
        if len(x_norm) < 5 or len(x_abn) < 5:
            continue
        mu_n, sd_n = float(np.mean(x_norm)), float(np.std(x_norm, ddof=1))
        mu_a, sd_a = float(np.mean(x_abn)), float(np.std(x_abn, ddof=1))
        g = _hedges_g(x_abn, x_norm)
        if g is None:
            continue
        biomarkers[key] = {
            "mu_normal": mu_n, "sigma_normal": sd_n,
            "mu_abnormal": mu_a, "sigma_abnormal": sd_a,
            "direction": "decrease" if mu_a < mu_n else "increase",
            "abs_g": float(abs(g)),
            "n_normal": int(len(x_norm)),
            "n_abnormal": int(len(x_abn)),
        }
        ranked.append((key, float(abs(g))))

    ranked.sort(key=lambda kv: kv[1], reverse=True)
    sequence = [k for k, _ in ranked]
    return {"sequence": sequence, "biomarkers": biomarkers}


def stage_visit(
    visit_values: dict[str, Optional[float]],
    ebm: dict,
) -> dict:
    """
    Assign a stage (number of abnormal biomarkers along the EBM sequence)
    to a single visit.

    Returns ``{stage, stage_max, abnormalities, posteriors}``.
      - posteriors[k] = P(abnormal | x_k) under the fitted Gaussians
                        (assumes equal prior over the two states)
      - abnormalities  = ordered subset of ``ebm['sequence']`` for which
                         the posterior > 0.5
    """
    sequence = ebm.get("sequence") or []
    biomarkers = ebm.get("biomarkers") or {}
    posteriors: dict[str, float] = {}
    abnormalities: list[str] = []

    for key in sequence:
        params = biomarkers.get(key)
        v = visit_values.get(key)
        if params is None or v is None or not math.isfinite(v):
            posteriors[key] = None
            continue
        log_n = _gauss_logpdf(float(v), params["mu_normal"], params["sigma_normal"])
        log_a = _gauss_logpdf(float(v), params["mu_abnormal"], params["sigma_abnormal"])
        # softmax of two log-likelihoods with equal prior
        if not math.isfinite(log_n) and not math.isfinite(log_a):
            posteriors[key] = None
            continue
        m = max(log_n, log_a)
        log_n_s = log_n - m
        log_a_s = log_a - m
        post_a = math.exp(log_a_s) / (math.exp(log_n_s) + math.exp(log_a_s))
        posteriors[key] = float(post_a)
        if post_a > 0.5:
            abnormalities.append(key)

    # Stage = count of abnormalities along the canonical sequence (in order).
    stage = 0
    for key in sequence:
        if key in abnormalities:
            stage += 1
        else:
            # Stop counting at first non-abnormal — preserves EBM monotonicity
            break
    return {
        "stage": int(stage),
        "stage_max": int(len(sequence)),
        "abnormalities": abnormalities,
        "posteriors": posteriors,
    }
