"""
survival.py — Kaplan-Meier time-to-conversion analysis (lifelines).

Cohort-level survival view: time-to-conversion (M0 -> first visit with
diagnosis "ad") stratified by APOE4 carrier status and biological stage.
Pure-Python wrapper around ``lifelines.KaplanMeierFitter`` so the route
just calls one function and serialises the result for the frontend.
"""

from __future__ import annotations

import math
import re
from typing import Optional

import numpy as np
import pandas as pd


def _visit_months(visit) -> Optional[int]:
    """Parse ``"M12"`` -> 12; returns None for unrecognised codes."""
    if visit is None:
        return None
    m = re.match(r"M(\d+)", str(visit).strip().upper())
    return int(m.group(1)) if m else None


def _is_apoe4_carrier(apoe) -> Optional[bool]:
    """ApoE4 carrier = at least one ε4 allele. Handles 'e3/e4', '3/4', '34', etc."""
    if apoe is None or (isinstance(apoe, float) and math.isnan(apoe)):
        return None
    s = str(apoe).strip().lower().replace("e", "").replace("ε", "")
    if not s:
        return None
    # Treat any digit '4' in the string as an ε4 allele
    return "4" in s


def time_to_conversion_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-subject (duration, event_observed) table for converters.

    duration = months from M0 to the *first* visit labelled 'ad'
    event_observed = 1 if ad seen, 0 if right-censored at last visit

    Only includes subjects whose baseline diagnosis is in {converter, mci}
    — these are the at-risk groups whose conversion we care about.
    """
    if not all(c in df.columns for c in ("subject_id", "diagnosis", "visit")):
        return pd.DataFrame(columns=["subject_id", "duration", "event_observed", "apoe4"])

    out_rows: list[dict] = []
    for sid, grp in df.dropna(subset=["subject_id"]).groupby("subject_id"):
        diags = grp["diagnosis"].astype(str).str.lower().str.strip()
        # Baseline diagnosis (first non-empty, non-NaN)
        baseline_diag = diags.iloc[0] if len(diags) else ""
        if baseline_diag not in ("converter", "mci"):
            continue

        months = grp["visit"].apply(_visit_months)
        ad_mask = diags == "ad"
        if ad_mask.any():
            duration = int(months[ad_mask].min() or 0)
            event = 1
        else:
            valid_months = months.dropna()
            if valid_months.empty:
                continue
            duration = int(valid_months.max())
            event = 0

        apoe4 = None
        if "apoe" in grp.columns:
            for v in grp["apoe"]:
                c = _is_apoe4_carrier(v)
                if c is not None:
                    apoe4 = c
                    break

        out_rows.append({
            "subject_id": str(sid),
            "duration": duration,
            "event_observed": event,
            "apoe4": apoe4,
        })
    return pd.DataFrame(out_rows)


def kaplan_meier(
    df: pd.DataFrame,
    stratify_by: Optional[str] = None,
) -> dict:
    """
    Fit a Kaplan-Meier curve. ``stratify_by`` may be 'apoe4' or None.

    Returns a JSON-friendly dict with one curve per stratum:
        {strata: [{label, timeline: [...], survival: [...], ci_lo, ci_hi, n, n_events}]}
    """
    table = time_to_conversion_table(df)
    if table.empty:
        return {"strata": []}

    try:
        from lifelines import KaplanMeierFitter
    except ImportError:
        return {"strata": [], "error": "lifelines not installed"}

    strata: list[dict] = []

    def _fit_one(sub: pd.DataFrame, label: str) -> Optional[dict]:
        sub = sub[sub["duration"].notna()]
        if sub.empty:
            return None
        kmf = KaplanMeierFitter()
        kmf.fit(durations=sub["duration"].values,
                event_observed=sub["event_observed"].values,
                label=label)
        sf = kmf.survival_function_
        ci = kmf.confidence_interval_
        timeline = sf.index.tolist()
        survival = sf.iloc[:, 0].tolist()
        ci_lo = ci.iloc[:, 0].tolist()
        ci_hi = ci.iloc[:, 1].tolist()
        return {
            "label": label,
            "n": int(len(sub)),
            "n_events": int(sub["event_observed"].sum()),
            "timeline": [float(t) for t in timeline],
            "survival": [float(s) for s in survival],
            "ci_lo": [float(s) for s in ci_lo],
            "ci_hi": [float(s) for s in ci_hi],
        }

    if stratify_by == "apoe4":
        for carrier_value, label in ((True, "APOE4+"), (False, "APOE4−")):
            sub = table[table["apoe4"] == carrier_value]
            curve = _fit_one(sub, label)
            if curve:
                strata.append(curve)
        # Unknown APOE -> separate stratum so we don't lose subjects
        unk = table[table["apoe4"].isna()]
        if len(unk) >= 3:
            curve = _fit_one(unk, "APOE unknown")
            if curve:
                strata.append(curve)
    else:
        curve = _fit_one(table, "All at-risk")
        if curve:
            strata.append(curve)

    return {"strata": strata, "n_total": int(len(table))}
