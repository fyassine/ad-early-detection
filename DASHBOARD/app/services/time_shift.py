"""
time_shift.py — Per-patient time-shift on a cohort-average disease curve.

Lightweight Disease Course Mapping (Couronné, Ortholand, Schiratti 2023-2024,
``leaspy``-style) without the full Riemannian geometry. We fit a logistic
curve to the cohort-mean trajectory of each biomarker (CN, MCI-NC, AD as
anchor groups + converter visits as the longitudinal track), then estimate
each patient's time-shift τ that best aligns their visits with the
cohort-mean curve.

A positive τ (in months) means "this patient is τ months ahead of the
average converter trajectory" — i.e. closer to the AD endpoint than their
chronology suggests. Useful as a single-number prognostic summary in the
patient overview.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def _safe_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if not math.isfinite(f) else f


def _visit_months(visit) -> Optional[int]:
    if visit is None:
        return None
    m = re.match(r"M(\d+)", str(visit).strip().upper())
    return int(m.group(1)) if m else None


def _logistic(t: np.ndarray, L: float, k: float, t0: float, b: float) -> np.ndarray:
    """4-parameter logistic. L=range, k=slope, t0=midpoint, b=baseline."""
    return b + L / (1.0 + np.exp(-k * (t - t0)))


@dataclass
class TimeShiftModel:
    """One logistic curve per biomarker over the longitudinal sample."""
    biomarkers: dict = field(default_factory=dict)
    # Each biomarker: {"L": ..., "k": ..., "t0": ..., "b": ..., "direction": "increase"/"decrease",
    #                  "t_min": ..., "t_max": ...}


def fit_time_shift_model(
    longitudinal: dict[str, list[tuple[int, float]]],
) -> TimeShiftModel:
    """
    ``longitudinal[biomarker]`` is a list of ``(month_from_M0, value)`` pairs
    aggregated across the converter cohort. We fit a 4-parameter logistic to
    each biomarker so that we can infer time-shift later.

    If scipy is unavailable, falls back to a simple linear fit + slope-based
    surrogate (still produces a usable trajectory).
    """
    out = TimeShiftModel()
    try:
        from scipy.optimize import curve_fit
        has_scipy = True
    except ImportError:
        has_scipy = False

    for key, samples in longitudinal.items():
        pts = [(t, v) for t, v in samples if t is not None and v is not None and math.isfinite(v)]
        if len(pts) < 6:
            continue
        ts = np.asarray([p[0] for p in pts], dtype=np.float64)
        ys = np.asarray([p[1] for p in pts], dtype=np.float64)
        b_init = float(np.percentile(ys, 5))
        L_init = float(np.percentile(ys, 95) - np.percentile(ys, 5))
        if abs(L_init) < 1e-6:
            continue
        direction = "increase" if (ys[ts.argmax()] - ys[ts.argmin()]) > 0 else "decrease"

        if has_scipy:
            try:
                popt, _ = curve_fit(
                    _logistic, ts, ys,
                    p0=[L_init, 0.05, float(np.median(ts)), b_init],
                    maxfev=2000,
                )
                L, k, t0, b = (float(x) for x in popt)
            except Exception:
                continue
        else:
            # Linear fallback
            slope, intercept = np.polyfit(ts, ys, 1)
            L = float(L_init); k = float(slope) / max(L_init, 1e-6); t0 = float(np.median(ts)); b = float(intercept)

        out.biomarkers[key] = {
            "L": L, "k": k, "t0": t0, "b": b,
            "direction": direction,
            "t_min": float(ts.min()), "t_max": float(ts.max()),
        }
    return out


def estimate_patient_time_shift(
    model: TimeShiftModel,
    patient_visits: list[dict],
) -> dict:
    """
    Find the τ that minimises (patient_value - logistic(t + τ))² across all
    biomarkers and visits.

    ``patient_visits`` is a list of ``{visit, <biomarker_key>: value, ...}``
    dicts (i.e. mergedVisits from the frontend backend pipe). The visit name
    is parsed for months.
    """
    if not model.biomarkers:
        return {"tau_months": None, "n_obs": 0}

    # Collect (biomarker, t_observed, y_observed)
    obs = []
    for v in patient_visits:
        t = _visit_months(v.get("visit"))
        if t is None:
            continue
        for key in model.biomarkers:
            y = v.get(key)
            if y is None or not math.isfinite(float(y)):
                continue
            obs.append((key, float(t), float(y)))
    if len(obs) < 3:
        return {"tau_months": None, "n_obs": len(obs)}

    # Per-biomarker normalisation: divide residuals by biomarker σ across
    # observed samples so all biomarkers contribute equally to τ.
    by_key: dict[str, list[float]] = {}
    for key, _, y in obs:
        by_key.setdefault(key, []).append(y)
    sigmas = {k: max(float(np.std(vs, ddof=1)) if len(vs) > 1 else 1.0, 1e-6) for k, vs in by_key.items()}

    def loss(tau: float) -> float:
        total = 0.0
        n = 0
        for key, t, y in obs:
            p = model.biomarkers[key]
            t_eff = t + tau
            y_pred = _logistic(np.asarray([t_eff]), p["L"], p["k"], p["t0"], p["b"])[0]
            total += ((y - float(y_pred)) / sigmas[key]) ** 2
            n += 1
        return total / max(n, 1)

    # Coarse grid search → local minimisation
    grid = np.linspace(-120.0, 120.0, 121)  # ±10 years in 2-month steps
    losses = np.asarray([loss(t) for t in grid])
    tau_init = float(grid[int(losses.argmin())])
    try:
        from scipy.optimize import minimize_scalar
        res = minimize_scalar(loss, bracket=(tau_init - 12, tau_init, tau_init + 12), method="brent")
        tau = float(res.x)
    except Exception:
        tau = tau_init

    return {
        "tau_months": _safe_float(tau),
        "n_obs": int(len(obs)),
        "biomarkers_used": sorted(by_key.keys()),
    }
