"""
Build a per-visit diagnosis CSV by matching each fMRI session (image_id)
to the closest clinical assessment (visit_id) from OASIS3_UDSb4_cdr.csv
within a ±365-day window.

Output columns:
  image_id        — MR session label     (e.g. OAS30001_MR_d0129)
  visit_id        — CDR visit label      (e.g. OAS30001_UDSb4_d0339)
  subject_id      — OASIS subject ID     (e.g. OAS30001)
  days_image      — MR days-from-entry
  days_visit      — CDR days-from-entry
  days_gap        — |days_image - days_visit|
  age_at_visit    — age at CDR assessment
  MMSE            — Mini-Mental State Exam score
  CDRTOT          — Global CDR score (0 / 0.5 / 1 / 2 / 3)
  CDRSUM          — CDR Sum of Boxes
  dx1_code        — Numeric diagnosis code
  dx1             — Diagnosis label
  dx_category     — Simplified category: HC / MCI / AD / Other

Run from the project root:
  python3 DATA/OASIS3/__metadata__/build_visit_diagnosis.py
"""

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]  # __metadata__ -> OASIS3 -> DATA -> project root

# ── Paths ──────────────────────────────────────────────────────────────────────
SESSIONS_CSV = REPO_ROOT / "DATA/OASIS3/sessions/oasis3_bold_sessions.csv"
CLINICAL_CSV = ROOT / "OASIS3_UDSb4_cdr.csv"
OUTPUT_CSV = ROOT / "visit_level_diagnoses.csv"

WINDOW_DAYS = 365  # maximum |days_image - days_visit| to accept a match

# ── Helpers ───────────────────────────────────────────────────────────────────


def parse_days(label: str) -> int | None:
    """Extract the integer number of days from a label like 'OAS30001_MR_d0129'."""
    m = re.search(r"d(\d+)$", str(label))
    return int(m.group(1)) if m else None


def parse_subject(label: str) -> str | None:
    """Extract subject ID (OAS3XXXX) from a session label."""
    m = re.match(r"(OAS3\d+)", str(label))
    return m.group(1) if m else None


def simplified_category(row) -> str:
    """Map the detailed diagnosis to one of: HC / MCI / AD / Other."""
    cdrtot = row.get("CDRTOT")
    dx1 = str(row.get("dx1", "")).upper()

    if cdrtot == 0.0:
        return "HC"
    if cdrtot == 0.5:
        if "AD" in dx1 or "DAT" in dx1:
            return "AD"
        return "MCI"
    if cdrtot >= 1.0:
        if "AD" in dx1 or "DAT" in dx1:
            return "AD"
        return "Other Dementia"
    return "Unknown"


# ── Load data ─────────────────────────────────────────────────────────────────

print("Loading sessions…")
sessions = pd.read_csv(SESSIONS_CSV)
sessions.columns = [c.strip() for c in sessions.columns]
# Rename the column to a consistent name regardless of original header
sessions = sessions.rename(columns={sessions.columns[0]: "experiment_id"})
sessions["subject_id"] = sessions["experiment_id"].apply(parse_subject)
sessions["days_image"] = sessions["experiment_id"].apply(parse_days)
sessions = sessions.dropna(subset=["subject_id", "days_image"])
sessions["days_image"] = sessions["days_image"].astype(int)
print(f"  {len(sessions)} MR sessions loaded.")

print("Loading clinical data…")
cdr = pd.read_csv(CLINICAL_CSV)
cdr["subject_id"] = cdr["OASISID"]
cdr["days_visit"] = pd.to_numeric(cdr["days_to_visit"], errors="coerce")
cdr = cdr.dropna(subset=["subject_id", "days_visit"])
cdr["days_visit"] = cdr["days_visit"].astype(int)
print(f"  {len(cdr)} clinical visits loaded.")

# ── Match each MR session to the closest CDR visit ────────────────────────────

print("Matching sessions to clinical visits…")
cdr_indexed = cdr.set_index("subject_id")
records = []

for _, mr in sessions.iterrows():
    subj = mr["subject_id"]
    d_img = mr["days_image"]
    img_label = mr["experiment_id"]

    if subj not in cdr_indexed.index:
        continue

    visits = cdr_indexed.loc[[subj]].copy()
    visits["gap"] = (visits["days_visit"] - d_img).abs()

    # Keep only visits within the ±WINDOW_DAYS window
    within = visits[visits["gap"] <= WINDOW_DAYS]
    if within.empty:
        continue

    # Pick the single closest visit (nsmallest avoids idxmin ambiguity)
    best = within.nsmallest(1, "gap").iloc[0]

    records.append(
        {
            "image_id": img_label,
            "visit_id": best["OASIS_session_label"],
            "subject_id": subj,
            "days_image": int(d_img),
            "days_visit": int(best["days_visit"]),
            "days_gap": int(best["gap"]),
            "age_at_visit": best["age at visit"],
            "MMSE": best["MMSE"],
            "CDRTOT": best["CDRTOT"],
            "CDRSUM": best["CDRSUM"],
            "dx1_code": best["dx1_code"],
            "dx1": best["dx1"],
            "dx_category": simplified_category(best),
        }
    )

out = pd.DataFrame(records)
print(f"  {len(out)} matches found out of {len(sessions)} MR sessions.")

# ── Diagnostics ───────────────────────────────────────────────────────────────

print("\ndx_category distribution:")
print(out["dx_category"].value_counts())

print("\nCDRTOT distribution:")
print(out["CDRTOT"].value_counts().sort_index())

# ── Save ──────────────────────────────────────────────────────────────────────

out.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved → {OUTPUT_CSV}")
