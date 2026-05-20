"""
population.py — Population-tier aggregations.

Builds the data backing the Population top-tab:

  - ``cohort_demographic_summary``  totals, conversion rates and demographic
                                    mix across the loaded cohort.
  - ``fang_epidemiology_table``     static reference lifetime-risk table from
                                    Fang et al. 2025 (Nat. Med.) used as the
                                    epidemiology overlay.
  - ``network_disruption_atlas``    per-Schaefer-7-network effect-size matrix
                                    between diagnosis pairs (DMN, salience,
                                    FPN, limbic, visual, sensorimotor, dorsal
                                    attention). Used to render the population
                                    network heatmap.

`cohort_demographic_summary` runs from metadata only.
`network_disruption_atlas` consumes a CohortStats instance that is already
cached in memory. None of these helpers touch disk directly.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from ..cohort_stats import COHORTS, SCHAEFER_NETWORKS, CohortStats


# --------------------------------------------------------------------------- #
# Fang et al. 2025 lifetime-risk reference                                    #
# --------------------------------------------------------------------------- #

def fang_epidemiology_table() -> dict:
    """
    Lifetime AD-and-related-dementia risk reference values, adapted from
    Fang et al. (2025) "Lifetime risk and projected burden of dementia"
    (Nat. Med.).

    Returned as percentage points (0-1) so the frontend can format with the
    same axis as cohort observed rates.
    """
    return {
        "overall_age_55_plus": 0.42,
        "by_sex": {
            "Female": 0.48,
            "Male": 0.35,
        },
        "by_race": {
            "Black": 0.48,
            "Hispanic": 0.43,
            "White": 0.40,
            "Asian": 0.37,
        },
        "by_apoe4": {
            "Non-carrier": 0.30,
            "Heterozygote (e3/e4)": 0.42,
            "Homozygote (e4/e4)": 0.59,
        },
        "by_age_residual": [
            {"age": 55, "risk": 0.42},
            {"age": 65, "risk": 0.36},
            {"age": 75, "risk": 0.28},
            {"age": 85, "risk": 0.18},
        ],
        "citation": (
            "Fang, Pike, Coresh et al. (2025). Lifetime risk and projected burden "
            "of dementia. Nature Medicine."
        ),
        "notes": (
            "Residual lifetime risk decreases with attained age because shorter "
            "remaining lifespans give the disease less time to manifest. "
            "APOE-stratified estimates are pooled from the cited paper's "
            "primary analytic cohort; sex/race breakdowns reflect Table 2 of "
            "the same paper."
        ),
    }


# --------------------------------------------------------------------------- #
# Cohort demographic summary                                                   #
# --------------------------------------------------------------------------- #

def _safe_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if not math.isfinite(f) else f


def _pct(numerator: int, denominator: int) -> Optional[float]:
    if not denominator:
        return None
    return float(numerator) / float(denominator)


def cohort_demographic_summary(df: pd.DataFrame) -> dict:
    """
    Population-level demographic and conversion summary.

    Returns:
        {
            "cohorts": {
                cohort_name: {n_subjects, n_visits, age_mean, age_std,
                              sex_pct_F, apoe4_pct, conversion_rate}
            },
            "totals": {n_subjects, n_visits, n_cohorts, conversion_rate},
            "site": {site_name: {n_subjects, n_visits}}   # optional, when site col present
        }
    """
    if df is None or df.empty:
        return {"cohorts": {}, "totals": {}, "site": {}}

    diag = df["diagnosis"].astype(str).str.lower() if "diagnosis" in df.columns else None
    cohort_summary: dict = {}
    total_subjects = 0
    total_visits = 0

    for cohort in COHORTS:
        if diag is None:
            continue
        sub = df[diag == cohort]
        if sub.empty:
            cohort_summary[cohort] = {
                "n_subjects": 0, "n_visits": 0,
                "age_mean": None, "age_std": None,
                "sex_pct_F": None, "apoe4_pct": None,
                "conversion_rate": None,
            }
            continue

        n_visits = int(len(sub))
        ids = sub["subject_id"].astype(str).unique() if "subject_id" in sub.columns else []
        n_subjects = int(len(ids))
        total_subjects += n_subjects
        total_visits += n_visits

        age_mean = age_std = None
        if "age" in sub.columns:
            ages = pd.to_numeric(sub["age"], errors="coerce").dropna()
            if not ages.empty:
                age_mean = _safe_float(ages.mean())
                age_std = _safe_float(ages.std(ddof=0))

        sex_pct_F = None
        if "sex" in sub.columns:
            sex = sub["sex"].astype(str).str.upper().str[0]
            n_f = int((sex == "F").sum())
            sex_pct_F = _pct(n_f, n_f + int((sex == "M").sum()))

        apoe4_pct = None
        if "apoe4" in sub.columns:
            apoe = pd.to_numeric(sub["apoe4"], errors="coerce").dropna()
            if not apoe.empty:
                apoe4_pct = _pct(int((apoe >= 1).sum()), int(apoe.size))

        cohort_summary[cohort] = {
            "n_subjects": n_subjects,
            "n_visits": n_visits,
            "age_mean": age_mean,
            "age_std": age_std,
            "sex_pct_F": sex_pct_F,
            "apoe4_pct": apoe4_pct,
            # Conversion rate is only meaningful relative to the MCI / converter
            # cohorts. We surface it as the share of the cohort with a recorded
            # converter visit, which is 0 for non-MCI groups.
            "conversion_rate": 1.0 if cohort == "converter" else 0.0,
        }

    # Site / study breakdown — optional, present in multi-site CSVs.
    site_summary: dict = {}
    site_col = next((c for c in ("site", "study", "dataset") if c in df.columns), None)
    if site_col is not None:
        for site, sub in df.groupby(df[site_col].astype(str)):
            ids = sub["subject_id"].astype(str).unique() if "subject_id" in sub.columns else []
            site_summary[site] = {
                "n_subjects": int(len(ids)),
                "n_visits": int(len(sub)),
            }

    converters = cohort_summary.get("converter", {}).get("n_subjects", 0)
    mci_total = cohort_summary.get("mci", {}).get("n_subjects", 0) + converters
    return {
        "cohorts": cohort_summary,
        "totals": {
            "n_subjects": total_subjects,
            "n_visits": total_visits,
            "n_cohorts": sum(1 for c in cohort_summary.values() if c["n_subjects"] > 0),
            "mci_conversion_rate": _pct(converters, mci_total),
        },
        "site": site_summary,
    }


# --------------------------------------------------------------------------- #
# Network-level disruption atlas                                              #
# --------------------------------------------------------------------------- #

def _approx_cohens_d(m_a, s_a, n_a, m_b, s_b, n_b) -> Optional[float]:
    """
    Pooled Cohen's d from summary stats (used when raw per-subject values
    aren't cached). Equivalent to ``effect_sizes.cohens_d`` without the
    Hedges small-sample correction.
    """
    if None in (m_a, s_a, n_a, m_b, s_b, n_b):
        return None
    if n_a < 2 or n_b < 2:
        return None
    try:
        va, vb = float(s_a) ** 2, float(s_b) ** 2
        pooled = math.sqrt(((n_a - 1) * va + (n_b - 1) * vb) / (n_a + n_b - 2))
    except (TypeError, ValueError):
        return None
    if pooled < 1e-9:
        return None
    return float((float(m_a) - float(m_b)) / pooled)


def network_disruption_atlas(stats: CohortStats) -> dict:
    """
    Per-Schaefer-7-network effect-size matrix across cohort pairs.

    Uses ``stats.network_fc_stats`` (mean / std / n) which is already cached.
    Reports Cohen's d (without bootstrap CI — that needs raw values that the
    cache doesn't yet store; phase 3 extends ``CohortStats`` to add them).

    Returns:
        {
          "networks":  [...],          # network names with any data
          "cohorts":   [...],          # cohorts with at least one network
          "matrix":    {network: {(cohort_a__cohort_b): d, ...}},
          "summary":   {network: {mean, std, n}, ...},
          "global_fc_by_network": {cohort: {network: mean_fc}}
        }
    """
    net_stats = stats.network_fc_stats or {}
    if not net_stats:
        return {"networks": [], "cohorts": [], "matrix": {}, "summary": {},
                "global_fc_by_network": {}}

    networks: list[str] = []
    for cohort in COHORTS:
        for net in (net_stats.get(cohort) or {}).keys():
            if net not in networks:
                networks.append(net)

    cohorts_with_data = [c for c in COHORTS if any(net_stats.get(c, {}).values())]

    matrix: dict = {}
    summary: dict = {}
    for net in networks:
        matrix[net] = {}
        per_cohort_means = []
        for i, ca in enumerate(cohorts_with_data):
            for cb in cohorts_with_data[i + 1:]:
                a = (net_stats.get(ca) or {}).get(net) or {}
                b = (net_stats.get(cb) or {}).get(net) or {}
                d = _approx_cohens_d(
                    a.get("mean"), a.get("std"), a.get("n"),
                    b.get("mean"), b.get("std"), b.get("n"),
                )
                matrix[net][f"{ca}__{cb}"] = d
        for c in cohorts_with_data:
            stat = (net_stats.get(c) or {}).get(net) or {}
            m = stat.get("mean")
            if m is not None and math.isfinite(float(m)):
                per_cohort_means.append(float(m))
        if per_cohort_means:
            arr = np.asarray(per_cohort_means, dtype=np.float64)
            summary[net] = {
                "mean": _safe_float(arr.mean()),
                "std": _safe_float(arr.std(ddof=0)) if arr.size > 1 else None,
                "n": int(arr.size),
            }

    global_fc_by_network: dict = {}
    for c in cohorts_with_data:
        nets = net_stats.get(c) or {}
        if not nets:
            continue
        global_fc_by_network[c] = {
            n: _safe_float((nets.get(n) or {}).get("mean"))
            for n in networks
        }

    return {
        "networks": networks,
        "cohorts": cohorts_with_data,
        "matrix": matrix,
        "summary": summary,
        "global_fc_by_network": global_fc_by_network,
    }
