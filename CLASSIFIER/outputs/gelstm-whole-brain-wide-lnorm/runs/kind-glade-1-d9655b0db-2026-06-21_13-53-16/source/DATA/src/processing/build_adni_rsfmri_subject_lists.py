"""
Build extended rsfMRI converter / non-converter subject lists for ADNI.

Expands the existing subject list (which only used "Resting State fMRI") to include:
  - Axial rsfMRI (Eyes Open) and all naming variants
  - Resting State fMRI and all naming variants
  - Extended Resting State fMRI and all naming variants
  - Axial fcMRI (Eyes Open) variants (same TR/TE=3000/30ms as Group A)

Grouped into three protocol groups by TR value:
  A: Standard rsfMRI  (TR > 2000 ms, TE ~30 ms)
  B: MB rsfMRI        (TR < 1000 ms, TE ~32-35 ms)
  C: HB rsfMRI        (1000 <= TR < 2000 ms, TE ~30 ms)
"""

import os
import pandas as pd
import numpy as np
from datetime import date

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE = "/mnt/e/fyassine/ad-early-detection/DATA"
DXSUM_PATH  = f"{BASE}/ADNI/All_Subjects_DXSUM_12May2026.csv"
FMRI_PATH   = f"{BASE}/ADNI/All_Subjects_Functional_MRI_Images_12May2026.csv"
OUT_DIR     = f"{BASE}/ADNI/"
OLD_ARTIFACT = f"{OUT_DIR}/RestingStatefMRI_MCI_Converters_NonConverters_12May2026.csv"

DATE_STR = date.today().strftime("%d%b%Y")   # e.g. "14May2026"
DATE_WINDOW = 90  # days to match an fMRI scan to a clinical visit

OUT_CONVERTERS     = f"{OUT_DIR}/Extended_rsfMRI_MCI_Converters_{DATE_STR}.csv"
OUT_NON_CONVERTERS = f"{OUT_DIR}/Extended_rsfMRI_MCI_NonConverters_{DATE_STR}.csv"


# ---------------------------------------------------------------------------
# Step 1: Load data
# ---------------------------------------------------------------------------
print("Loading data ...")
dxsum = pd.read_csv(DXSUM_PATH, low_memory=False)
fmri  = pd.read_csv(FMRI_PATH,  low_memory=False)
print(f"  DXSUM: {len(dxsum):,} rows, {dxsum['PTID'].nunique():,} subjects")
print(f"  fMRI:  {len(fmri):,} rows,  {fmri['subject_id'].nunique():,} subjects")


# ---------------------------------------------------------------------------
# Step 2: Filter fMRI to target sequence families
# ---------------------------------------------------------------------------
def _is_target(desc: str) -> bool:
    d = str(desc).lower().strip()
    return (
        "rsfmri" in d                          # rsfMRI / MB rsfMRI / HB rsfMRI
        or "resting state fmri" in d           # Resting State fMRI / Extended Resting State fMRI
        or ("extended" in d and "resting" in d)
        or "axial fcmri" in d                  # fcMRI naming variant (same TR/TE as Group A)
        or "axial resting fcmri" in d
    )

def _assign_group(tr) -> str:
    try:
        tr = float(tr)
    except (TypeError, ValueError):
        return "A"
    if tr < 1000:
        return "B"
    if tr < 2000:
        return "C"
    return "A"

fmri_cols = ["image_id", "subject_id", "fmri_visit", "fmri_date",
             "fmri_description", "fmri_tr", "fmri_te"]
fmri_f = fmri[fmri_cols].copy()
fmri_f = fmri_f[fmri_f["fmri_description"].apply(_is_target)].copy()
fmri_f["fmri_group"] = fmri_f["fmri_tr"].apply(_assign_group)
fmri_f["fmri_date"] = pd.to_datetime(fmri_f["fmri_date"], errors="coerce")

print(f"\nTarget fMRI sequences: {len(fmri_f):,} scans, "
      f"{fmri_f['subject_id'].nunique():,} subjects")
print("  Group A (Standard, TR>2000ms):", (fmri_f["fmri_group"] == "A").sum())
print("  Group B (MB,       TR<1000ms):", (fmri_f["fmri_group"] == "B").sum())
print("  Group C (HB,       TR 1-2s):",  (fmri_f["fmri_group"] == "C").sum())
print("  Unique descriptions:")
for desc, grp in (fmri_f.groupby("fmri_description")["fmri_group"]
                  .first().sort_index().items()):
    print(f"    [{grp}]  {desc}")


# ---------------------------------------------------------------------------
# Step 3: Prepare DXSUM — keep only rows with a valid numeric diagnosis 1/2/3
# ---------------------------------------------------------------------------
dx = dxsum[["PTID", "VISCODE", "EXAMDATE", "DIAGNOSIS"]].copy()
dx["DIAGNOSIS"] = pd.to_numeric(dx["DIAGNOSIS"], errors="coerce")
dx = dx[dx["DIAGNOSIS"].isin([1.0, 2.0, 3.0])].copy()
dx["DIAGNOSIS"] = dx["DIAGNOSIS"].astype(int)
dx["EXAMDATE"]  = pd.to_datetime(dx["EXAMDATE"], errors="coerce")
dx = dx.dropna(subset=["EXAMDATE"]).sort_values(["PTID", "EXAMDATE"]).reset_index(drop=True)
print(f"\nDXSUM after filtering: {len(dx):,} rows, {dx['PTID'].nunique():,} subjects")


# ---------------------------------------------------------------------------
# Step 4: Classify each subject as converter / non-converter / other
# ---------------------------------------------------------------------------
def _classify(group: pd.DataFrame):
    g = group.sort_values("EXAMDATE")
    diag = g["DIAGNOSIS"].values
    dates = g["EXAMDATE"].values

    if 2 not in diag:
        return None, None, None

    first_mci_date = g[g["DIAGNOSIS"] == 2]["EXAMDATE"].min()

    # Converter: DX=3 must appear strictly after first DX=2
    post_mci_dx3 = g[(g["DIAGNOSIS"] == 3) & (g["EXAMDATE"] > first_mci_date)]
    if len(post_mci_dx3) > 0:
        first_conversion_date = post_mci_dx3["EXAMDATE"].min()
        return "converter", first_mci_date, first_conversion_date

    # Non-converter: has DX=2, never reaches DX=3
    if 3 not in diag:
        return "non_converter_stable_mci", first_mci_date, pd.NaT

    return None, None, None

subject_meta = {}
for ptid, grp in dx.groupby("PTID"):
    label, first_mci, first_conv = _classify(grp)
    if label:
        subject_meta[ptid] = {"label": label,
                               "first_mci_date": first_mci,
                               "first_conversion_date": first_conv}

n_conv    = sum(1 for v in subject_meta.values() if v["label"] == "converter")
n_nonconv = sum(1 for v in subject_meta.values() if v["label"] == "non_converter_stable_mci")
print(f"\nClassified: {n_conv} converters, {n_nonconv} non-converters")


# ---------------------------------------------------------------------------
# Step 5: Extract relevant DXSUM visit rows per group
# ---------------------------------------------------------------------------
conv_visit_rows    = []
nonconv_visit_rows = []

for ptid, meta in subject_meta.items():
    subj = dx[dx["PTID"] == ptid].copy()
    label = meta["label"]

    if label == "converter":
        window = subj[
            (subj["EXAMDATE"] >= meta["first_mci_date"]) &
            (subj["EXAMDATE"] <= meta["first_conversion_date"])
        ].copy()
        window["label"] = label
        conv_visit_rows.append(window)

    else:  # non_converter_stable_mci
        dx2 = subj[subj["DIAGNOSIS"] == 2].copy()
        dx2["label"] = label
        nonconv_visit_rows.append(dx2)

conv_visits    = pd.concat(conv_visit_rows,    ignore_index=True) if conv_visit_rows    else pd.DataFrame(columns=["PTID","VISCODE","EXAMDATE","DIAGNOSIS","label"])
nonconv_visits = pd.concat(nonconv_visit_rows, ignore_index=True) if nonconv_visit_rows else pd.DataFrame(columns=["PTID","VISCODE","EXAMDATE","DIAGNOSIS","label"])

print(f"\nVisit rows — converters:     {len(conv_visits):,}  "
      f"({conv_visits['PTID'].nunique() if len(conv_visits) else 0} subjects)")
print(f"Visit rows — non-converters: {len(nonconv_visits):,}  "
      f"({nonconv_visits['PTID'].nunique() if len(nonconv_visits) else 0} subjects)")


# ---------------------------------------------------------------------------
# Step 6: Join fMRI scan availability onto each DXSUM visit row
# ---------------------------------------------------------------------------
FMRI_JOIN_COLS = ["image_id", "subject_id", "fmri_visit", "fmri_date",
                  "fmri_description", "fmri_group", "fmri_tr", "fmri_te"]

def attach_fmri(visits_df: pd.DataFrame, fmri_df: pd.DataFrame,
                date_window: int = DATE_WINDOW) -> pd.DataFrame:
    if visits_df.empty:
        return visits_df.copy()

    # Merge all fMRI records for matching subjects (left join keeps all visits)
    merged = visits_df.merge(
        fmri_df[FMRI_JOIN_COLS],
        left_on="PTID", right_on="subject_id",
        how="left"
    )
    merged["date_diff_days"] = (merged["fmri_date"] - merged["EXAMDATE"]).abs().dt.days

    # 1. Visits with at least one in-window fMRI scan — pick the closest
    in_window = merged[merged["date_diff_days"] <= date_window]
    if len(in_window) > 0:
        best = (in_window
                .sort_values("date_diff_days")
                .groupby(["PTID", "EXAMDATE"], as_index=False)
                .first())
        best["has_rsfmri_scan"] = True
    else:
        best = pd.DataFrame()

    # 2. Visits with no in-window scan — recover from original visit list
    matched_keys = (set(zip(best["PTID"], best["EXAMDATE"]))
                    if len(best) else set())

    no_scan = visits_df.merge(
        best[["PTID", "EXAMDATE"]] if len(best) else pd.DataFrame(columns=["PTID","EXAMDATE"]),
        on=["PTID", "EXAMDATE"],
        how="left",
        indicator=True
    )
    no_scan = no_scan[no_scan["_merge"] == "left_only"].drop(columns=["_merge"]).copy()
    no_scan["has_rsfmri_scan"] = False
    for col in ["image_id", "fmri_visit", "fmri_date", "fmri_description",
                "fmri_group", "fmri_tr", "fmri_te", "date_diff_days", "subject_id"]:
        no_scan[col] = np.nan

    result = pd.concat([best, no_scan], ignore_index=True)
    # Drop the redundant subject_id column from the fMRI side (PTID is the primary key)
    if "subject_id" in result.columns:
        result = result.drop(columns=["subject_id"])
    return result.sort_values(["PTID", "EXAMDATE"]).reset_index(drop=True)


print("\nAttaching fMRI availability to converter visits ...")
conv_out = attach_fmri(conv_visits, fmri_f)

print("Attaching fMRI availability to non-converter visits ...")
nonconv_out = attach_fmri(nonconv_visits, fmri_f)


# ---------------------------------------------------------------------------
# Step 7: Rename columns and select final output order
# ---------------------------------------------------------------------------
RENAME = {
    "PTID":    "subject_id",
    "VISCODE": "viscode",
    "EXAMDATE":"examdate",
    "DIAGNOSIS":"diagnosis",
}

OUT_COLS = [
    "subject_id", "label", "viscode", "examdate", "diagnosis",
    "has_rsfmri_scan",
    "image_id", "fmri_visit", "fmri_date", "fmri_description",
    "fmri_group", "fmri_tr", "fmri_te", "date_diff_days",
]

def finalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=RENAME)
    # image_id may be called "image_id" already (from fmri merge) or absent
    for col in OUT_COLS:
        if col not in df.columns:
            df[col] = np.nan
    return df[OUT_COLS].copy()

conv_out    = finalise(conv_out)
nonconv_out = finalise(nonconv_out)


# ---------------------------------------------------------------------------
# Step 8: Write outputs
# ---------------------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)
conv_out.to_csv(OUT_CONVERTERS,     index=False)
nonconv_out.to_csv(OUT_NON_CONVERTERS, index=False)
print(f"\nWrote: {OUT_CONVERTERS}")
print(f"Wrote: {OUT_NON_CONVERTERS}")


# ---------------------------------------------------------------------------
# Step 9: Summary statistics
# ---------------------------------------------------------------------------
def _group_summary(df: pd.DataFrame, label: str):
    n_subj = int(df["subject_id"].nunique())
    n_with_scan = int(df[df["has_rsfmri_scan"]]["subject_id"].nunique())
    n_scan_visits = int(df["has_rsfmri_scan"].sum())
    print(f"\n  {label}: {n_subj} subjects, "
          f"{n_with_scan} have ≥1 rsfMRI scan, "
          f"{n_scan_visits} scan-visits total")
    if n_scan_visits > 0:
        for grp, cnt in df[df["has_rsfmri_scan"]]["fmri_group"].value_counts().items():
            print(f"    Group {grp}: {cnt} scan-visits")

print("\n=== Summary ===")
_group_summary(conv_out,    "Converters")
_group_summary(nonconv_out, "Non-converters")

# Overlap with old artifact
try:
    old = pd.read_csv(OLD_ARTIFACT)
    old_conv    = set(old[old["label"] == "converter"]["subject_id"])
    old_nonconv = set(old[old["label"] == "non_converter_stable_mci"]["subject_id"])
    new_conv    = set(conv_out["subject_id"].dropna().unique())
    new_nonconv = set(nonconv_out["subject_id"].dropna().unique())
    print(f"\n  vs old artifact —")
    print(f"    Converters:     {len(old_conv)} → {len(new_conv)} "
          f"(+{len(new_conv - old_conv)} new, -{len(old_conv - new_conv)} dropped)")
    print(f"    Non-converters: {len(old_nonconv)} → {len(new_nonconv)} "
          f"(+{len(new_nonconv - old_nonconv)} new, -{len(old_nonconv - new_nonconv)} dropped)")
except FileNotFoundError:
    print(f"  (Old artifact not found for comparison)")

print("\nDone.")
