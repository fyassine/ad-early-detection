"""
Build OASIS3 converter / non-converter rsfMRI subject lists, matching the schema of
DATA/DELCODE/src/processing/build_adni_rsfmri_subject_lists.py so both cohorts can be
consumed downstream through one 14-column format.

Labels:
  converter                 — MCI at some visit, later diagnosed AD
  non_converter_stable_mci  — MCI at some visit, never diagnosed AD

MCI is "any of the 16 UDS Form D1 MCI-subtype columns == 1" (MCIAMEM, MCIAPLUS, MCINON1,
MCINON2 and their domain sub-flags). AD is "DEMENTED == 1 AND (PROBAD == 1 OR (alzdis == 1
AND alzdisif == 1))" — the OR is required because OASIS3 spans two UDS form eras: v1/v2
records the AD etiology in PROBAD, v3 replaced it with alzdis/alzdisif. The two fields are
near-perfectly complementary (only one is ever populated per row); using PROBAD alone
silently discards every v3-era AD diagnosis.

IMPORTANT — no calendar dates: OASIS3 is de-identified and tracks time only as
"days from entry" (days_to_visit / the d#### suffix in session labels). The examdate and
fmri_date columns below therefore hold integers, not dates, unlike the ADNI CSVs of the
same name. date_diff_days is directly comparable across both cohorts.

Run from the project root:
  source .venv/bin/activate && python DATA/OASIS3/__metadata__/build_oasis3_rsfmri_subject_lists.py
"""

import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent  # DATA/OASIS3/__metadata__
DIAGNOSES_PATH = ROOT / "OASIS3_UDSd1_diagnoses.csv"
MR_PATH = ROOT / "OASIS3_MR_json.csv"
OUT_DIR = ROOT

DATE_STR = date.today().strftime("%d%b%Y")  # e.g. "17Jul2026"
DATE_WINDOW = 365  # days to match an rsfMRI session to a clinical visit

OUT_CONVERTERS = OUT_DIR / f"Extended_rsfMRI_MCI_Converters_{DATE_STR}.csv"
OUT_NON_CONVERTERS = OUT_DIR / f"Extended_rsfMRI_MCI_NonConverters_{DATE_STR}.csv"
OUT_LONGITUDINAL = OUT_DIR / f"Extended_rsfMRI_MCI_Longitudinal_{DATE_STR}.csv"
OUT_UNANCHORED = OUT_DIR / f"Extended_rsfMRI_MCI_Unanchored_{DATE_STR}.csv"

MCI_COLS = [
    "MCIAMEM",
    "MCIAPLUS",
    "MCIAPLAN",
    "MCIAPATT",
    "MCIAPEX",
    "MCIAPVIS",
    "MCINON1",
    "MCIN1LAN",
    "MCIN1ATT",
    "MCIN1EX",
    "MCIN1VIS",
    "MCINON2",
    "MCIN2LAN",
    "MCIN2ATT",
    "MCIN2EX",
    "MCIN2VIS",
]
REQUIRED_D1_COLS = [
    "OASISID",
    "OASIS_session_label",
    "days_to_visit",
    "NORMCOG",
    "DEMENTED",
    "PROBAD",
    "alzdis",
    "alzdisif",
    *MCI_COLS,
]

# ---------------------------------------------------------------------------
# Step 1: Load diagnoses and derive per-visit diagnosis code
# ---------------------------------------------------------------------------
print("Loading diagnoses ...")
d1 = pd.read_csv(DIAGNOSES_PATH, low_memory=False)
missing = [c for c in REQUIRED_D1_COLS if c not in d1.columns]
if missing:
    raise ValueError(
        f"{DIAGNOSES_PATH} is missing required column(s): {missing}. "
        "This script hardcodes the OASIS3 UDS Form D1 schema and cannot "
        "silently fall back to a smaller MCI/AD definition."
    )
print(f"  D1: {len(d1):,} rows, {d1['OASISID'].nunique():,} subjects")

d1["days_to_visit"] = pd.to_numeric(d1["days_to_visit"], errors="coerce")
d1 = d1.dropna(subset=["days_to_visit"]).copy()
d1["days_to_visit"] = d1["days_to_visit"].astype(int)

is_cn = d1["NORMCOG"] == 1
is_mci = (d1[MCI_COLS] == 1).any(axis=1)  # IMPNOMCI==1 is deliberately NOT MCI
is_ad = (d1["DEMENTED"] == 1) & (
    (d1["PROBAD"] == 1) | ((d1["alzdis"] == 1) & (d1["alzdisif"] == 1))
)

d1["DIAGNOSIS"] = np.select([is_cn, is_mci, is_ad], [1, 2, 3], default=np.nan)
d1 = d1.dropna(subset=["DIAGNOSIS"]).copy()
d1["DIAGNOSIS"] = d1["DIAGNOSIS"].astype(int)
d1 = d1.sort_values(["OASISID", "days_to_visit"]).reset_index(drop=True)
print(f"  after CN/MCI/AD filtering: {len(d1):,} rows, {d1['OASISID'].nunique():,} subjects")
print(
    f"    CN={int((d1.DIAGNOSIS == 1).sum())}  MCI={int((d1.DIAGNOSIS == 2).sum())}  "
    f"AD={int((d1.DIAGNOSIS == 3).sum())}"
)


# ---------------------------------------------------------------------------
# Step 2: Classify each subject as converter / non-converter / other
# ---------------------------------------------------------------------------
def _classify(group: pd.DataFrame):
    g = group.sort_values("days_to_visit")
    diag = g["DIAGNOSIS"].values

    if 2 not in diag:
        return None, None, None

    first_mci_days = g[g["DIAGNOSIS"] == 2]["days_to_visit"].min()

    # Converter: DX=3 must appear strictly after first DX=2
    post_mci_dx3 = g[(g["DIAGNOSIS"] == 3) & (g["days_to_visit"] > first_mci_days)]
    if len(post_mci_dx3) > 0:
        first_conversion_days = post_mci_dx3["days_to_visit"].min()
        return "converter", first_mci_days, first_conversion_days

    # Non-converter: has DX=2, never reaches DX=3
    if 3 not in diag:
        return "non_converter_stable_mci", first_mci_days, None

    return None, None, None


subject_meta = {}
for oasisid, grp in d1.groupby("OASISID"):
    label, first_mci, first_conv = _classify(grp)
    if label:
        subject_meta[oasisid] = {
            "label": label,
            "first_mci_days": first_mci,
            "first_conversion_days": first_conv,
        }

n_conv = sum(1 for v in subject_meta.values() if v["label"] == "converter")
n_nonconv = sum(1 for v in subject_meta.values() if v["label"] == "non_converter_stable_mci")
print(f"\nClassified: {n_conv} converters, {n_nonconv} non-converters")


# ---------------------------------------------------------------------------
# Step 3: Extract relevant D1 visit rows per group
# ---------------------------------------------------------------------------
conv_visit_rows = []
nonconv_visit_rows = []

for oasisid, meta in subject_meta.items():
    subj = d1[d1["OASISID"] == oasisid].copy()
    label = meta["label"]

    if label == "converter":
        window = subj[
            (subj["days_to_visit"] >= meta["first_mci_days"])
            & (subj["days_to_visit"] <= meta["first_conversion_days"])
        ].copy()
        window["label"] = label
        conv_visit_rows.append(window)
    else:  # non_converter_stable_mci
        mci_rows = subj[subj["DIAGNOSIS"] == 2].copy()
        mci_rows["label"] = label
        nonconv_visit_rows.append(mci_rows)

conv_visits = (
    pd.concat(conv_visit_rows, ignore_index=True)
    if conv_visit_rows
    else pd.DataFrame(
        columns=["OASISID", "OASIS_session_label", "days_to_visit", "DIAGNOSIS", "label"]
    )
)
nonconv_visits = (
    pd.concat(nonconv_visit_rows, ignore_index=True)
    if nonconv_visit_rows
    else pd.DataFrame(
        columns=["OASISID", "OASIS_session_label", "days_to_visit", "DIAGNOSIS", "label"]
    )
)

print(
    f"\nVisit rows — converters:     {len(conv_visits):,}  "
    f"({conv_visits['OASISID'].nunique() if len(conv_visits) else 0} subjects)"
)
print(
    f"Visit rows — non-converters: {len(nonconv_visits):,}  "
    f"({nonconv_visits['OASISID'].nunique() if len(nonconv_visits) else 0} subjects)"
)


# ---------------------------------------------------------------------------
# Step 4: Load rsfMRI session inventory
# ---------------------------------------------------------------------------
print("\nLoading rsfMRI scan inventory ...")
mr = pd.read_csv(MR_PATH, low_memory=False)
mr.columns = [c.strip() for c in mr.columns]
required_mr_cols = ["label", "scan category", "SeriesDescription", "RepetitionTime", "EchoTime"]
missing_mr = [c for c in required_mr_cols if c not in mr.columns]
if missing_mr:
    raise ValueError(f"{MR_PATH} is missing required column(s): {missing_mr}")

rest = mr[mr["scan category"] == "bold-rest"].copy()
rest = rest[
    rest["SeriesDescription"] != "rsfmri_ref"
].copy()  # calibration scan, not usable rest data


def _parse_subject(label: str):
    m = re.match(r"(OAS3\d+)", str(label))
    return m.group(1) if m else None


def _parse_days(label: str):
    m = re.search(r"d(\d+)$", str(label))
    return int(m.group(1)) if m else None


rest["subject_id"] = rest["label"].apply(_parse_subject)
rest["days_image"] = rest["label"].apply(_parse_days)
rest = rest.dropna(subset=["subject_id", "days_image"]).copy()
rest["days_image"] = rest["days_image"].astype(int)

# OASIS3 BIDS JSON stores TR/TE in seconds; ADNI's CSVs store milliseconds.
rest["fmri_tr"] = rest["RepetitionTime"] * 1000.0
rest["fmri_te"] = rest["EchoTime"] * 1000.0


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


rest["fmri_group"] = rest["fmri_tr"].apply(_assign_group)

# One row per (subject, session) — most sessions have multiple rest runs, but the
# BIDS session directory is the matching unit.
sessions = (
    rest.sort_values("label")
    .drop_duplicates(subset=["subject_id", "days_image"], keep="first")
    .copy()
)
sessions = sessions.rename(columns={"label": "image_id", "SeriesDescription": "fmri_description"})
sessions["fmri_visit"] = sessions["image_id"]
sessions["fmri_date"] = sessions["days_image"]

print(
    f"  rsfMRI: {len(rest):,} scan rows -> {len(sessions):,} sessions, "
    f"{sessions['subject_id'].nunique():,} subjects"
)
print("  fmri_group distribution:", sessions["fmri_group"].value_counts().to_dict())


# ---------------------------------------------------------------------------
# Step 5: Attach rsfMRI availability to each visit row
# ---------------------------------------------------------------------------
SESSION_JOIN_COLS = [
    "image_id",
    "subject_id",
    "fmri_visit",
    "fmri_date",
    "fmri_description",
    "fmri_group",
    "fmri_tr",
    "fmri_te",
]


def attach_fmri(
    visits_df: pd.DataFrame, sessions_df: pd.DataFrame, date_window: int = DATE_WINDOW
) -> pd.DataFrame:
    if visits_df.empty:
        return visits_df.copy()

    merged = visits_df.merge(
        sessions_df[SESSION_JOIN_COLS],
        left_on="OASISID",
        right_on="subject_id",
        how="left",
    )
    merged["date_diff_days"] = (merged["fmri_date"] - merged["days_to_visit"]).abs()

    # 1. Visits with at least one in-window session — pick the closest
    in_window = merged[merged["date_diff_days"] <= date_window]
    if len(in_window) > 0:
        best = (
            in_window.sort_values("date_diff_days")
            .groupby(["OASISID", "days_to_visit"], as_index=False)
            .first()
        )
        best["has_rsfmri_scan"] = True
    else:
        best = pd.DataFrame()

    # 2. Visits with no in-window session — recover from original visit list
    no_scan = visits_df.merge(
        best[["OASISID", "days_to_visit"]]
        if len(best)
        else pd.DataFrame(columns=["OASISID", "days_to_visit"]),
        on=["OASISID", "days_to_visit"],
        how="left",
        indicator=True,
    )
    no_scan = no_scan[no_scan["_merge"] == "left_only"].drop(columns=["_merge"]).copy()
    no_scan["has_rsfmri_scan"] = False
    for col in [
        "image_id",
        "fmri_visit",
        "fmri_date",
        "fmri_description",
        "fmri_group",
        "fmri_tr",
        "fmri_te",
        "date_diff_days",
        "subject_id",
    ]:
        no_scan[col] = np.nan

    result = pd.concat([best, no_scan], ignore_index=True)
    if "subject_id" in result.columns:
        result = result.drop(columns=["subject_id"])
    return result.sort_values(["OASISID", "days_to_visit"]).reset_index(drop=True)


print("\nAttaching rsfMRI availability to converter visits ...")
conv_out = attach_fmri(conv_visits, sessions)

print("Attaching rsfMRI availability to non-converter visits ...")
nonconv_out = attach_fmri(nonconv_visits, sessions)


# ---------------------------------------------------------------------------
# Step 6: Rename columns and select final output order (ADNI-parity schema)
# ---------------------------------------------------------------------------
RENAME = {
    "OASISID": "subject_id",
    "OASIS_session_label": "viscode",
    "days_to_visit": "examdate",
    "DIAGNOSIS": "diagnosis",
}

OUT_COLS = [
    "subject_id",
    "label",
    "viscode",
    "examdate",
    "diagnosis",
    "has_rsfmri_scan",
    "image_id",
    "fmri_visit",
    "fmri_date",
    "fmri_description",
    "fmri_group",
    "fmri_tr",
    "fmri_te",
    "date_diff_days",
]


def finalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=RENAME)
    for col in OUT_COLS:
        if col not in df.columns:
            df[col] = np.nan
    return df[OUT_COLS].copy()


conv_out = finalise(conv_out)
nonconv_out = finalise(nonconv_out)


# ---------------------------------------------------------------------------
# Step 7: Build the Longitudinal CSV
# ---------------------------------------------------------------------------
# Criterion (OASIS3-specific, documented in README): subjects with >=2
# scan-matched visits (has_rsfmri_scan == True) across the converter/
# non-converter union. The ADNI Longitudinal CSV has no reproducible
# generator in the repo, so this is a fresh, explicit definition rather than
# a reverse-engineered match to ADNI's selection.
union_out = pd.concat([conv_out, nonconv_out], ignore_index=True)
scan_counts = union_out[union_out["has_rsfmri_scan"] == True].groupby("subject_id").size()  # noqa: E712
longitudinal_subjects = set(scan_counts[scan_counts >= 2].index)
longitudinal_out = union_out[union_out["subject_id"].isin(longitudinal_subjects)].copy()
longitudinal_out = longitudinal_out.sort_values(["subject_id", "examdate"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 7b: Build the "Unanchored" audit CSV
# ---------------------------------------------------------------------------
# attach_fmri() only ever emits a row for an rsfMRI session when it is the
# *closest* in-window (<=DATE_WINDOW days) session to some labeled visit. A
# downloaded session that never wins that comparison for any visit of its
# subject — either because every visit is >DATE_WINDOW days away, or because
# a different session was closer for all of them — never gets a row anywhere
# and silently disappears from the Converters/NonConverters/Longitudinal
# CSVs. This step surfaces exactly those sessions: same subject/scan
# identifiers, plus the closest labeled visit regardless of distance and the
# true date_diff_days to it, so "no metadata row" is visible and auditable
# instead of looking like missing data.
ALL_VISIT_COLS = ["OASISID", "OASIS_session_label", "days_to_visit", "DIAGNOSIS", "label"]
all_visits = (
    pd.concat([conv_visits[ALL_VISIT_COLS], nonconv_visits[ALL_VISIT_COLS]], ignore_index=True)
    if (len(conv_visits) or len(nonconv_visits))
    else pd.DataFrame(columns=ALL_VISIT_COLS)
)

matched_image_ids = set(union_out["image_id"].dropna())

classified_ids = set(subject_meta.keys())
sessions_classified = sessions[sessions["subject_id"].isin(classified_ids)].copy()

unanchored_rows = []
for oasisid, sess_grp in sessions_classified.groupby("subject_id"):
    subj_visits = all_visits[all_visits["OASISID"] == oasisid]
    if subj_visits.empty:
        continue  # cannot happen for a classified subject, but guard anyway
    label = subject_meta[oasisid]["label"]
    for _, srow in sess_grp.iterrows():
        if srow["image_id"] in matched_image_ids:
            continue  # already has a row in conv_out/nonconv_out
        diffs = (subj_visits["days_to_visit"] - srow["days_image"]).abs()
        nearest = subj_visits.loc[diffs.idxmin()]
        unanchored_rows.append(
            {
                "subject_id": oasisid,
                "label": label,
                "image_id": srow["image_id"],
                "fmri_visit": srow["fmri_visit"],
                "fmri_date": int(srow["fmri_date"]),
                "fmri_description": srow["fmri_description"],
                "fmri_group": srow["fmri_group"],
                "fmri_tr": srow["fmri_tr"],
                "fmri_te": srow["fmri_te"],
                "nearest_viscode": nearest["OASIS_session_label"],
                "nearest_examdate": int(nearest["days_to_visit"]),
                "nearest_diagnosis": int(nearest["DIAGNOSIS"]),
                "date_diff_days": int(diffs.min()),
            }
        )

UNANCHORED_COLS = [
    "subject_id",
    "label",
    "image_id",
    "fmri_visit",
    "fmri_date",
    "fmri_description",
    "fmri_group",
    "fmri_tr",
    "fmri_te",
    "nearest_viscode",
    "nearest_examdate",
    "nearest_diagnosis",
    "date_diff_days",
]
unanchored_out = pd.DataFrame(unanchored_rows, columns=UNANCHORED_COLS)
if len(unanchored_out):
    unanchored_out = unanchored_out.sort_values(["subject_id", "fmri_date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 8: Write outputs
# ---------------------------------------------------------------------------
conv_out.to_csv(OUT_CONVERTERS, index=False)
nonconv_out.to_csv(OUT_NON_CONVERTERS, index=False)
longitudinal_out.to_csv(OUT_LONGITUDINAL, index=False)
unanchored_out.to_csv(OUT_UNANCHORED, index=False)
print(f"\nWrote: {OUT_CONVERTERS}")
print(f"Wrote: {OUT_NON_CONVERTERS}")
print(f"Wrote: {OUT_LONGITUDINAL}")
print(f"Wrote: {OUT_UNANCHORED}")


# ---------------------------------------------------------------------------
# Step 9: Summary statistics
# ---------------------------------------------------------------------------
def _group_summary(df: pd.DataFrame, label: str):
    n_subj = int(df["subject_id"].nunique())
    n_with_scan = int(df[df["has_rsfmri_scan"] == True]["subject_id"].nunique())  # noqa: E712
    n_scan_visits = int((df["has_rsfmri_scan"] == True).sum())  # noqa: E712
    print(
        f"\n  {label}: {n_subj} subjects, "
        f"{n_with_scan} have >=1 rsfMRI scan, "
        f"{n_scan_visits} scan-visits total"
    )
    if n_scan_visits > 0:
        for grp, cnt in df[df["has_rsfmri_scan"] == True]["fmri_group"].value_counts().items():  # noqa: E712
            print(f"    Group {grp}: {cnt} scan-visits")


print("\n=== Summary ===")
_group_summary(conv_out, "Converters")
_group_summary(nonconv_out, "Non-converters")
print(
    f"\n  Longitudinal: {longitudinal_out['subject_id'].nunique()} subjects "
    f"(>=2 scan-matched visits), {len(longitudinal_out)} rows"
)
print(
    f"\n  Unanchored: {unanchored_out['subject_id'].nunique() if len(unanchored_out) else 0} subjects, "
    f"{len(unanchored_out)} downloaded rsfMRI sessions with no in-window ({DATE_WINDOW}d) "
    "labeled-visit match"
)

print("\nDone.")
