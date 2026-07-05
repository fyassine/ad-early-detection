"""
survival_table.py — Build per-subject (duration, event_observed, covariates)
tables for time-to-conversion (MCI → AD) survival analysis.

Censoring rules:
  - Keep subjects whose earliest non-NaN diagnosis ∈ {converter, mci}
  - event_observed = 1 if any visit has diagnosis == 'ad'
      duration = months from M0 to that first AD visit
  - event_observed = 0 (right-censored) at last visit if no AD seen

Logic ported and extended from DASHBOARD/app/services/survival.py:39-92.
Kept standalone (no DASHBOARD imports) so PROGNOSER is deployable on its own.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _visit_months(visit) -> int | None:
    """Parse 'M12' → 12; returns None for unrecognised codes."""
    if visit is None or (isinstance(visit, float) and math.isnan(visit)):
        return None
    m = re.match(r"M(\d+)", str(visit).strip().upper())
    return int(m.group(1)) if m else None


def _is_apoe4_carrier(apoe) -> bool | None:
    """ApoE4 carrier = at least one ε4 allele. Handles 'e3/e4', '3/4', '34', etc."""
    if apoe is None or (isinstance(apoe, float) and math.isnan(apoe)):
        return None
    s = str(apoe).strip().lower().replace("e", "").replace("ε", "")
    if not s:
        return None
    return "4" in s


def _first_non_null(series: pd.Series):
    """Return the first non-null, non-empty value in a Series, or None."""
    for v in series:
        if v is None:
            continue
        if isinstance(v, float) and math.isnan(v):
            continue
        if isinstance(v, str) and v.strip().lower() in ("", "nan", "none", "nat"):
            continue
        return v
    return None


def _coerce_float(v) -> float | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if not math.isfinite(f) else f


def build_survival_table(
    cohorts_csv: str | Path,
    splits_dir: str | Path | None = None,
    split: str | None = None,
    include_features: Iterable[str] = ("age", "sex", "mmstot", "cdrglobal", "apoe4"),
    longitudinal_features: Iterable[str] = (),
    longitudinal_aggs: Iterable[str] = ("baseline", "last", "slope", "delta"),
) -> pd.DataFrame:
    """
    Build a per-subject survival table from a longitudinal cohorts CSV.

    Parameters
    ----------
    cohorts_csv : path
        Path to DELCODE-style cohorts.csv with columns at least
        Pseudonym/visit/diagnosis/visdate (sex, age via brthdat, mmstot,
        cdrglobal, ApoE optional).
    splits_dir : path | None
        If provided with `split`, restrict to subjects in that split CSV.
    split : 'train' | 'val' | 'test' | None
        Which split file to filter on (looks for splits_dir/{split}.csv).
    include_features : iterable of str
        Which clinical covariates to extract per subject. Supported:
        age, sex, mmstot, cdrglobal, apoe4.

    Returns
    -------
    DataFrame with columns:
        subject_id, duration, event_observed, baseline_visit,
        baseline_filename (from `file` column), <feature columns>, split
    """
    cohorts_csv = Path(cohorts_csv)
    df = pd.read_csv(cohorts_csv, low_memory=False)

    id_col = next((c for c in ("Pseudonym", "subject_id") if c in df.columns), None)
    if id_col is None:
        raise ValueError(f"No subject ID column found in {cohorts_csv}")
    if "diagnosis" not in df.columns or "visit" not in df.columns:
        raise ValueError(f"cohorts CSV must have 'diagnosis' and 'visit' columns, got {list(df.columns)}")

    df = df.copy()
    df[id_col] = df[id_col].astype(str)
    df["_diagnosis_norm"] = df["diagnosis"].astype(str).str.lower().str.strip()
    df["_months"] = df["visit"].apply(_visit_months)

    allowed_subjects: set[str] | None = None
    if splits_dir is not None and split is not None:
        split_csv = Path(splits_dir) / f"{split}.csv"
        if not split_csv.exists():
            raise FileNotFoundError(f"Split CSV not found: {split_csv}")
        sdf = pd.read_csv(split_csv)
        sid_col = next((c for c in ("Pseudonym", "subject_id") if c in sdf.columns), None)
        if sid_col is None:
            raise ValueError(f"Split CSV {split_csv} missing subject ID column")
        allowed_subjects = set(sdf[sid_col].astype(str))

    long_features = list(longitudinal_features)
    long_aggs = list(longitudinal_aggs)

    # Build LongitudinalAggregator if longitudinal features requested
    agg = None
    if long_features:
        from PROGNOSER.common.longitudinal import LongitudinalAggregator
        agg = LongitudinalAggregator(df, id_col=id_col, visit_col="visit", diagnosis_col="diagnosis")

    rows: list[dict] = []
    for sid, grp in df.dropna(subset=[id_col]).groupby(id_col):
        if allowed_subjects is not None and sid not in allowed_subjects:
            continue

        grp = grp.sort_values("_months", na_position="last").reset_index(drop=True)
        diags = grp["_diagnosis_norm"]
        valid_diags = diags[~diags.isin(["nan", "", "none", "nat"])]
        baseline_diag = valid_diags.iloc[0] if len(valid_diags) else ""
        if baseline_diag not in ("converter", "mci"):
            continue

        months = grp["_months"]
        ad_mask = diags == "ad"
        if ad_mask.any():
            duration = int(months[ad_mask].dropna().min()) if not months[ad_mask].dropna().empty else 0
            event = 1
        else:
            valid_months = months.dropna()
            if valid_months.empty:
                continue
            duration = int(valid_months.max())
            event = 0

        baseline_row = grp.iloc[0]
        baseline_visit = baseline_row.get("visit", "M0")
        baseline_filename = baseline_row.get("file", None)

        record = {
            "subject_id": str(sid),
            "duration": float(duration),
            "event_observed": int(event),
            "baseline_diagnosis": baseline_diag,
            "baseline_visit": baseline_visit,
            "baseline_filename": baseline_filename,
            "n_visits_in_window": int((months < duration).sum()) if months.notna().any() else 0,
        }

        if "age" in include_features:
            record["age"] = _extract_age(grp)
        if "sex" in include_features:
            sex_val = _first_non_null(grp.get("sex", pd.Series(dtype=object)))
            record["sex"] = 1 if str(sex_val).lower().strip() == "m" else 0 if sex_val is not None else None
        if "mmstot" in include_features:
            record["mmstot"] = _coerce_float(_first_non_null(grp.get("mmstot", pd.Series(dtype=object))))
        if "cdrglobal" in include_features:
            record["cdrglobal"] = _coerce_float(_first_non_null(grp.get("cdrglobal", pd.Series(dtype=object))))
        if "apoe4" in include_features:
            apoe4_val = None
            for v in grp.get("ApoE", grp.get("apoe", pd.Series(dtype=object))):
                c = _is_apoe4_carrier(v)
                if c is not None:
                    apoe4_val = int(c)
                    break
            record["apoe4"] = apoe4_val

        # Longitudinal aggregate features — computed within the at-risk window,
        # same logic for converters and non-converters
        if agg is not None:
            for feat in long_features:
                for agg_name in long_aggs:
                    col_name = f"{feat}_{agg_name}"
                    fn = getattr(agg, agg_name, None)
                    if fn is None:
                        continue
                    try:
                        record[col_name] = fn(str(sid), feat, duration)
                    except Exception:
                        record[col_name] = None

        if split is not None:
            record["split"] = split

        rows.append(record)

    return pd.DataFrame(rows)


def _extract_age(grp: pd.DataFrame) -> float | None:
    """Compute age at baseline visit. Prefers explicit 'age' column;
    falls back to (visdate - brthdat) if both available."""
    if "age" in grp.columns:
        a = _coerce_float(_first_non_null(grp["age"]))
        if a is not None:
            return a
    if "brthdat" in grp.columns and "visdate" in grp.columns:
        baseline = grp.iloc[0]
        try:
            birth = pd.to_datetime(baseline["brthdat"], errors="coerce")
            visit = pd.to_datetime(baseline["visdate"], dayfirst=True, errors="coerce")
            if pd.notna(birth) and pd.notna(visit):
                return float((visit - birth).days / 365.25)
        except Exception:
            return None
    return None


def filter_to_split(table: pd.DataFrame, splits_dir: str | Path, split: str) -> pd.DataFrame:
    """Filter an already-built survival table to subjects in a split CSV."""
    split_csv = Path(splits_dir) / f"{split}.csv"
    sdf = pd.read_csv(split_csv)
    sid_col = next((c for c in ("Pseudonym", "subject_id") if c in sdf.columns), None)
    allowed = set(sdf[sid_col].astype(str))
    return table[table["subject_id"].astype(str).isin(allowed)].reset_index(drop=True)


def make_xte(
    table: pd.DataFrame,
    feature_cols: list[str],
    drop_na: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Convert a survival table to (X, T, E) numpy arrays for model fitting.

    Returns
    -------
    X : (n, n_features) float64
    T : (n,) float64 — durations
    E : (n,) int — event_observed (0/1)
    used_table : DataFrame — the subset of `table` actually used (after NaN drop)
    """
    missing = [c for c in feature_cols if c not in table.columns]
    if missing:
        raise KeyError(f"Missing feature columns in table: {missing}")

    sub = table[["subject_id", "duration", "event_observed", *feature_cols]].copy()
    if drop_na:
        sub = sub.dropna(subset=feature_cols + ["duration", "event_observed"]).reset_index(drop=True)

    X = sub[feature_cols].astype(float).to_numpy()
    T = sub["duration"].astype(float).to_numpy()
    E = sub["event_observed"].astype(int).to_numpy()
    return X, T, E, sub
