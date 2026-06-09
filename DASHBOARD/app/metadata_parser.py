"""
metadata_parser.py — Generic CSV metadata parser.

Dataset-agnostic: auto-detects delimiter, normalizes common column names,
computes derived fields (age), and returns structured metrics.
"""

import io
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


# Column name normalization map (lowercase source -> canonical name)
COLUMN_ALIASES = {
    # Subject ID
    "pseudonym": "subject_id",
    "repseudonym": "subject_id",
    "subjectid": "subject_id",
    "subject_id": "subject_id",
    # Sex
    "sex": "sex",
    # Birth date
    "brthdat": "birth_date",
    # Diagnosis
    "diagnosis": "diagnosis",
    "prmdiag": "diagnosis_code",
    # Visit
    "visit": "visit",
    "viscode": "visit",   # ADNI VISCODE / VISCODE2 alternate names
    "viscode2": "visit",
    "visnam": "visit_name",
    "visdate": "visit_date",
    "visdat": "visit_date",
    # Clinical scores
    "mmstot": "mmse_total",
    "moctot": "moca_total",
    "cdrtot": "cdr_sum",
    "cdrglobal": "cdr_global",
    "gdstot": "gds_total",
    "faqtot": "faq_total",
    # Biomarkers
    "apoe": "apoe",
    "abeta42": "abeta42",
    "totaltau": "total_tau",
    "phosphotau181": "p_tau",
    "ratio_abeta42_40": "abeta_ratio",
    "pacc5": "pacc5",
    # Split
    "split": "split",
    # File reference
    "file": "file_ref",
}

# DELCODE diagnosis code mapping
DIAGNOSIS_CODE_MAP = {
    0: "healthy",
    1: "scd",
    2: "mci",  # includes converters tagged as mci initially
    3: "relative",
    5: "ad",
}


def detect_delimiter(file_path: str) -> str:
    """Auto-detect CSV delimiter by reading first few lines."""
    with open(file_path, "r", encoding="utf-8-sig", errors="replace") as f:
        sample = f.read(8192)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        # Fallback: count occurrences
        for delim in [";", ",", "\t", "|"]:
            if delim in sample:
                return delim
        return ","


def load_metadata(file_path: str, cohort: Optional[str] = None) -> pd.DataFrame:
    """
    Load a metadata CSV file, auto-detecting delimiter.
    Returns a DataFrame with normalized column names.
    If cohort is provided, filters all rows to match that diagnosis.
    """
    delimiter = detect_delimiter(file_path)
    df = pd.read_csv(
        file_path,
        delimiter=delimiter,
        encoding="utf-8-sig",
        low_memory=False,
        on_bad_lines="skip",
    )

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Build a rename map from detected columns
    rename_map = {}
    for col in df.columns:
        lower = col.lower().strip()
        if lower in COLUMN_ALIASES:
            canonical = COLUMN_ALIASES[lower]
            if canonical not in rename_map.values():
                rename_map[col] = canonical

    df = df.rename(columns=rename_map)

    # --- Derived fields ---

    # Compute age from birth_date if available
    if "birth_date" in df.columns:
        df["age"] = df["birth_date"].apply(_compute_age)

    # Map diagnosis codes to labels if we have codes but no text diagnosis
    if "diagnosis_code" in df.columns and "diagnosis" not in df.columns:
        df["diagnosis"] = df["diagnosis_code"].apply(
            lambda x: DIAGNOSIS_CODE_MAP.get(_safe_int(x), f"unknown({x})")
        )

    # Normalize diagnosis labels
    if "diagnosis" in df.columns:
        df["diagnosis"] = df["diagnosis"].astype(str).str.strip().str.lower()

    # If the diagnosis column contains only ADNI-style numeric codes (1/2/3), map them to
    # text labels so the survival analysis and cohort stats work correctly.
    # ADNI: 1=CN, 2=MCI, 3=AD  (DELCODE numeric codes are handled separately via prmdiag)
    _ADNI_DX_MAP = {"1": "cn", "2": "mci", "3": "ad"}
    if "diagnosis" in df.columns:
        non_null = df["diagnosis"].dropna()
        non_null = non_null[~non_null.isin(["nan", "", "none"])]
        if len(non_null) > 0 and non_null.isin(_ADNI_DX_MAP).all():
            df["diagnosis"] = df["diagnosis"].map(_ADNI_DX_MAP).fillna(df["diagnosis"])

    # Normalize sex
    if "sex" in df.columns:
        df["sex"] = df["sex"].astype(str).str.strip().str.lower()
        df["sex"] = df["sex"].replace({"male": "m", "female": "f", "1": "m", "2": "f"})

    # Normalize visit
    if "visit" in df.columns:
        df["visit"] = df["visit"].astype(str).str.strip()
    elif "visit_name" in df.columns:
        df["visit"] = df["visit_name"].astype(str).str.strip()

    # --- Filter by Cohort if requested ---
    if cohort and "diagnosis" in df.columns:
        cohort_str = cohort.strip().lower()
        df = df[df["diagnosis"] == cohort_str]

    return df


def compute_metadata_metrics(df: pd.DataFrame, scan_subjects: Optional[list[str]] = None, scan_subject_counts: Optional[dict] = None) -> dict:
    """
    Compute aggregate metrics from a metadata DataFrame.
    If scan_subjects is provided, also compute metrics for the scan subset.
    If scan_subject_counts is provided, compute scans per diagnosis.
    """
    metrics = {}

    # Total rows and unique subjects
    metrics["total_rows"] = len(df)
    if "subject_id" in df.columns:
        metrics["unique_subjects"] = int(df["subject_id"].nunique())
    else:
        metrics["unique_subjects"] = metrics["total_rows"]

    # Columns available
    metrics["columns_available"] = list(df.columns)

    # --- Distributions ---

    # Diagnosis (patients at baseline + scans per diagnosis)
    if "diagnosis" in df.columns:
        baseline = _get_baseline(df)
        diag_counts = baseline["diagnosis"].value_counts().to_dict()
        metrics["diagnosis_distribution"] = {str(k): int(v) for k, v in diag_counts.items()}

        if scan_subject_counts and "subject_id" in baseline.columns:
            diag_scans: dict[str, int] = {}
            for _, row in baseline.iterrows():
                sid = str(row.get("subject_id", ""))
                diag = str(row.get("diagnosis", "unknown"))
                n_scans = scan_subject_counts.get(sid, 0)
                diag_scans[diag] = diag_scans.get(diag, 0) + n_scans
            metrics["diagnosis_scans"] = {k: int(diag_scans.get(k, 0)) for k in metrics["diagnosis_distribution"]}

        if "visit" in df.columns and "subject_id" in baseline.columns:
            visit_df = df[["subject_id", "visit"]].copy()
            visit_df["subject_id"] = visit_df["subject_id"].astype(str)
            visit_df["visit"] = visit_df["visit"].astype(str).str.strip()
            visit_df = visit_df[~visit_df["visit"].isin(["", ".", "nan", "NaN"])]
            unique_visits = visit_df.drop_duplicates()

            subj_diag = baseline.set_index("subject_id")["diagnosis"].astype(str).to_dict()
            diag_visits: dict[str, int] = {}
            for _, row in unique_visits.iterrows():
                diag = subj_diag.get(row["subject_id"], "unknown")
                diag_visits[diag] = diag_visits.get(diag, 0) + 1

            metrics["diagnosis_visits"] = {k: int(diag_visits.get(k, 0)) for k in metrics["diagnosis_distribution"]}

    # Sex
    if "sex" in df.columns:
        baseline = _get_baseline(df)
        sex_counts = baseline["sex"].value_counts().to_dict()
        metrics["sex_distribution"] = {str(k): int(v) for k, v in sex_counts.items()}

    # Age
    if "age" in df.columns:
        baseline = _get_baseline(df)
        ages = baseline["age"].dropna()
        if len(ages) > 0:
            metrics["age_stats"] = {
                "mean": round(float(ages.mean()), 1),
                "median": round(float(ages.median()), 1),
                "min": round(float(ages.min()), 1),
                "max": round(float(ages.max()), 1),
                "std": round(float(ages.std()), 1),
            }
            # Histogram bins
            bins = list(range(int(ages.min() // 5) * 5, int(ages.max() // 5 + 2) * 5, 5))
            hist = pd.cut(ages, bins=bins).value_counts().sort_index()
            metrics["age_histogram"] = {
                "labels": [str(interval) for interval in hist.index],
                "counts": [int(v) for v in hist.values],
            }

    # Visit distribution
    if "visit" in df.columns:
        visit_counts = df["visit"].value_counts().to_dict()
        metrics["visit_distribution"] = {str(k): int(v) for k, v in visit_counts.items()}

    # MMSE
    if "mmse_total" in df.columns:
        baseline = _get_baseline(df)
        mmse = pd.to_numeric(baseline["mmse_total"], errors="coerce").dropna()
        if len(mmse) > 0:
            metrics["mmse_stats"] = {
                "mean": round(float(mmse.mean()), 1),
                "median": round(float(mmse.median()), 1),
                "min": round(float(mmse.min()), 1),
                "max": round(float(mmse.max()), 1),
            }
            bins = list(range(0, 32, 2))
            hist = pd.cut(mmse, bins=bins).value_counts().sort_index()
            metrics["mmse_histogram"] = {
                "labels": [str(interval) for interval in hist.index],
                "counts": [int(v) for v in hist.values],
            }

    # CDR
    if "cdr_global" in df.columns:
        baseline = _get_baseline(df)
        cdr = pd.to_numeric(baseline["cdr_global"], errors="coerce").dropna()
        if len(cdr) > 0:
            cdr_counts = cdr.value_counts().sort_index().to_dict()
            metrics["cdr_distribution"] = {str(k): int(v) for k, v in cdr_counts.items()}

    # ApoE
    if "apoe" in df.columns:
        baseline = _get_baseline(df)
        apoe = baseline["apoe"].dropna().astype(str).str.strip()
        apoe = apoe[apoe != ""]
        if len(apoe) > 0:
            apoe_counts = apoe.value_counts().to_dict()
            metrics["apoe_distribution"] = {str(k): int(v) for k, v in apoe_counts.items()}
            # ε4 zygosity grouping (Yin 2023 JAMA Neurol: OR ≈3× het, 8–12× hom)
            zyg = apoe.apply(_classify_apoe4_zygosity)
            zyg_counts = zyg.value_counts().to_dict()
            metrics["apoe4_zygosity_distribution"] = {
                "non-carrier": int(zyg_counts.get("non-carrier", 0)),
                "heterozygous": int(zyg_counts.get("heterozygous", 0)),
                "homozygous":  int(zyg_counts.get("homozygous", 0)),
            }

    # Train/Val/Test split
    if "split" in df.columns:
        baseline = _get_baseline(df)
        split_counts = baseline["split"].dropna().value_counts().to_dict()
        metrics["split_distribution"] = {str(k): int(v) for k, v in split_counts.items()}

    # Scan coverage
    if scan_subjects is not None and "subject_id" in df.columns:
        meta_subjects = set(df["subject_id"].dropna().astype(str))
        scan_set = set(scan_subjects)
        matched = meta_subjects & scan_set
        metrics["scan_coverage"] = {
            "metadata_subjects": len(meta_subjects),
            "scan_subjects": len(scan_set),
            "matched": len(matched),
            "metadata_only": len(meta_subjects - scan_set),
            "scans_only": len(scan_set - meta_subjects),
        }

    # Visits per subject stats
    if "subject_id" in df.columns and "visit" in df.columns:
        visits_per_subject = df.groupby("subject_id")["visit"].nunique().to_dict()
        if visits_per_subject:
            metrics["visits_per_subject_stats"] = {
                "mean": round(float(pd.Series(list(visits_per_subject.values())).mean()), 1),
                "max": int(max(visits_per_subject.values())),
            }

    # Patient table: baseline row + longitudinal visit info
    if "subject_id" in df.columns:
        import re as _re
        baseline = _get_baseline(df)
        table_cols = [c for c in ["subject_id", "sex", "age", "diagnosis", "mmse_total",
                                   "cdr_global", "apoe", "split"] if c in baseline.columns]
        table_df = baseline[table_cols].copy()

        # Add per-patient visit list and count from full df
        if "visit" in df.columns:
            def _visit_list(sid):
                rows = df[df["subject_id"] == sid]["visit"].dropna().unique().tolist()
                def vkey(v):
                    m = _re.search(r"(\d+)", str(v))
                    return int(m.group(1)) if m else 9999
                rows.sort(key=vkey)
                return rows

            visit_map = {sid: _visit_list(sid) for sid in table_df["subject_id"].dropna().unique()}
            table_df["visits"] = table_df["subject_id"].map(lambda s: visit_map.get(s, []))
            table_df["n_visits"] = table_df["visits"].map(len)
            table_df["longitudinal"] = table_df["n_visits"] > 1

        # Scan count per subject
        if scan_subject_counts and "subject_id" in table_df.columns:
            table_df["n_scans"] = table_df["subject_id"].map(
                lambda s: scan_subject_counts.get(str(s), 0)
            )

        table_df = table_df.where(pd.notna(table_df), None)
        records = table_df.head(2000).to_dict(orient="records")
        # visits is a list — restore after the NaN-replacement
        if "visits" in table_df.columns:
            visit_col = table_df["visits"].head(2000).tolist()
            for i, rec in enumerate(records):
                rec["visits"] = visit_col[i] if isinstance(visit_col[i], list) else []
        metrics["patient_table"] = records

    return metrics


def _get_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Get baseline (first visit) rows. Try multiple heuristics."""
    if "file_ref" in df.columns:
        mask = df["file_ref"].astype(str).str.upper().str.contains("BASELINE", na=False)
        if mask.any():
            return df[mask]

    if "visit_name" in df.columns:
        mask = df["visit_name"].astype(str).str.lower().str.contains("baseline", na=False)
        if mask.any():
            return df[mask]

    if "visit" in df.columns:
        # Try M0 or Baseline
        mask = df["visit"].astype(str).str.upper().isin(["M0", "BASELINE", "M00"])
        if mask.any():
            return df[mask]

    # Fallback: first row per subject
    if "subject_id" in df.columns:
        return df.drop_duplicates(subset="subject_id", keep="first")

    return df


def _compute_age(birth_str) -> Optional[float]:
    """Compute age from birth date string (various formats)."""
    if pd.isna(birth_str) or str(birth_str).strip() in ["", ".", "nan"]:
        return None

    birth_str = str(birth_str).strip()
    now = datetime.now()

    # Try YYYY-MM format
    for fmt in ["%Y-%m", "%Y-%m-%d", "%m-%Y", "%d-%m-%Y"]:
        try:
            birth = datetime.strptime(birth_str, fmt)
            age = (now - birth).days / 365.25
            if 0 < age < 120:
                return round(age, 1)
        except ValueError:
            continue

    # Try just year
    try:
        year = int(birth_str[:4])
        age = now.year - year
        if 0 < age < 120:
            return float(age)
    except (ValueError, IndexError):
        pass

    return None


def _classify_apoe4_zygosity(apoe_str: str) -> str:
    """Return 'non-carrier', 'heterozygous', or 'homozygous' for an APOE genotype string."""
    s = str(apoe_str).strip().lower().replace("e", "").replace("ε", "").replace("/", "").replace("\\", "").replace(" ", "")
    count = s.count("4")
    if count >= 2:
        return "homozygous"
    if count == 1:
        return "heterozygous"
    return "non-carrier"


def _safe_int(val) -> Optional[int]:
    """Safely convert to int."""
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None

def get_patient_clinical_trajectory(df: pd.DataFrame, subject_id: str) -> dict:
    """
    Filter the dataframe for a specific subject_id and return a time-series
    dictionary of their clinical biomarkers across available visits.
    """
    if "subject_id" not in df.columns:
        return {}
    
    patient_df = df[df["subject_id"] == subject_id].copy()
    if patient_df.empty:
        return {}
    
    # Sort by visit logically (M0, M12, M24...). Returns a (priority, value)
    # tuple so visits that don't match the "M<n>" pattern sort *after* parsed
    # ones while keeping a stable secondary order, instead of all collapsing
    # onto a single sentinel.
    def visit_sort_key(v):
        v = str(v).upper()
        if v.startswith('M'):
            try:
                return (0, int(v[1:]))
            except ValueError:
                pass
        return (1, v)
        
    if "visit" in patient_df.columns:
        patient_df["sort_key"] = patient_df["visit"].apply(visit_sort_key)
        patient_df = patient_df.sort_values("sort_key")
        visits = patient_df["visit"].tolist()
    else:
        visits = [f"V{i}" for i in range(len(patient_df))]
        
    def _extract_col(col_name):
        if col_name in patient_df.columns:
            # Handle string conversions if necessary, keep None for NaNs
            res = []
            for x in patient_df[col_name]:
                try:
                    if pd.notna(x):
                        res.append(float(x))
                    else:
                        res.append(None)
                except ValueError:
                    res.append(None)
            return res
        return [None] * len(patient_df)
        
    def _extract_str_col(col_name):
        if col_name in patient_df.columns:
            res = []
            for x in patient_df[col_name]:
                if pd.notna(x):
                    res.append(str(x))
                else:
                    res.append(None)
            return res
        return [None] * len(patient_df)
        
    return {
        "visits": visits,
        "diagnosis": _extract_str_col("diagnosis"),
        "cognitive": {
            "mmse": _extract_col("mmse_total"),
            "cdr": _extract_col("cdr_global"),
            "pacc5": _extract_col("pacc5")
        },
        "csf": {
            "abeta42": _extract_col("abeta42"),
            "tau": _extract_col("total_tau"),
            "ptau": _extract_col("p_tau")
        }
    }
