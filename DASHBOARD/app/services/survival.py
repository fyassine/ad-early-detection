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
    """Parse a visit code to months.

    Handles:
      DELCODE  M0, M12, M24 …
      ADNI     bl/sc/screen → 0; m06 → 6; m12 → 12 (VISCODE2 lowercase-m style)
    """
    if visit is None:
        return None
    s = str(visit).strip().upper()
    if s in ("BL", "SC", "SCMRI", "SCREEN"):
        return 0
    m = re.match(r"^M0*(\d+)$", s)   # M0, M06, M12, M024 all match
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
        # Sort by visit time so baseline = earliest visit regardless of CSV row order
        grp = grp.copy()
        grp["_months"] = grp["visit"].apply(_visit_months)
        grp = grp.sort_values("_months", na_position="last").reset_index(drop=True)

        diags = grp["diagnosis"].astype(str).str.lower().str.strip()
        # First non-empty, non-NaN diagnosis (handles CSVs where M0 row has no label)
        valid_diags = diags[~diags.isin(["nan", "", "none", "nat"])]
        baseline_diag = valid_diags.iloc[0] if len(valid_diags) else ""
        if baseline_diag not in ("converter", "mci"):
            continue

        months = grp["_months"]
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


def _attach_atn_stage(table: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Add an 'atn_stage' column to ``table`` from the baseline ATN classification."""
    try:
        from .atn import classify_atn
    except ImportError:
        table["atn_stage"] = None
        return table

    stage_map: dict[str, str] = {}
    for sid, grp in df.dropna(subset=["subject_id"]).groupby("subject_id"):
        grp2 = grp.copy()
        grp2["_m"] = grp2["visit"].apply(_visit_months)
        grp2 = grp2.sort_values("_m", na_position="last").reset_index(drop=True)
        # Compute ATN for each visit, take the baseline stage
        for _, row in grp2.iterrows():
            atn = classify_atn(
                abeta42=row.get("abeta42"),
                p_tau=row.get("p_tau"),
                total_tau=row.get("total_tau"),
            )
            if atn and atn.get("stage") is not None:
                stage_map[str(sid)] = f"Stage {atn['stage']}"
                break

    table = table.copy()
    table["atn_stage"] = table["subject_id"].map(stage_map)
    return table


def kaplan_meier(
    df: pd.DataFrame,
    stratify_by: Optional[str] = None,
) -> dict:
    """
    Fit a Kaplan-Meier curve. ``stratify_by`` may be 'apoe4', 'atn', or None.

    Returns a JSON-friendly dict with one curve per stratum:
        {strata: [{label, timeline: [...], survival: [...], ci_lo, ci_hi, n, n_events}]}
    """
    table = time_to_conversion_table(df)
    if table.empty:
        return {
            "strata": [],
            "reason": (
                "No subjects with baseline diagnosis 'mci' or 'converter' were found after "
                "sorting each subject's visits chronologically. Kaplan-Meier requires longitudinal "
                "MCI subjects so that time-to-conversion (first visit labelled 'ad') can be measured."
            ),
        }

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
        unk = table[table["apoe4"].isna()]
        if len(unk) >= 3:
            curve = _fit_one(unk, "APOE unknown")
            if curve:
                strata.append(curve)
        if not strata:
            curve = _fit_one(table, "All at-risk (APOE unknown)")
            if curve:
                strata.append(curve)

    elif stratify_by == "atn":
        table = _attach_atn_stage(table, df)
        atn_order = ["Stage 0", "Stage 1", "Stage 2", "Stage 3"]
        for stage in atn_order:
            sub = table[table["atn_stage"] == stage]
            if len(sub) >= 3:
                curve = _fit_one(sub, stage)
                if curve:
                    strata.append(curve)
        if not strata:
            curve = _fit_one(table, "All at-risk")
            if curve:
                strata.append(curve)

    else:
        curve = _fit_one(table, "All at-risk")
        if curve:
            strata.append(curve)

    if not strata:
        return {
            "strata": [],
            "n_total": int(len(table)),
            "reason": (
                f"Found {len(table)} at-risk subjects but could not fit a survival curve. "
                "Check that subjects have valid visit months (M0/M12/…) and at least one "
                "non-zero duration."
            ),
        }

    return {"strata": strata, "n_total": int(len(table))}
