"""
Report the OASIS3 subjects whose on-disk rsfMRI sessions never fall within the
±365-day matching window (`DATE_WINDOW` in build_oasis3_rsfmri_subject_lists.py)
of any labeled clinical visit, for every session they have.

This reuses the exact same classification (converter / non_converter_stable_mci),
visit-window restriction, and session inventory as
build_oasis3_rsfmri_subject_lists.py::attach_fmri() -- it does not introduce a new
or looser definition of "match". A subject is "fully unmatched" here iff every one
of their on-disk rsfMRI sessions has date_diff_days > DATE_WINDOW against every
visit in their relevant window (the converter's MCI->AD span, or the
non-converter's MCI-only visits).

Run from the project root:
  source .venv/bin/activate && \
  python DATA/OASIS3/__metadata__/__extra__/build_unmatched_subjects_report.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

METADATA_DIR = Path(__file__).resolve().parent.parent  # DATA/OASIS3/__metadata__
OUT_DIR = Path(__file__).resolve().parent  # DATA/OASIS3/__metadata__/__extra__

DIAGNOSES_PATH = METADATA_DIR / "OASIS3_UDSd1_diagnoses.csv"
MR_PATH = METADATA_DIR / "OASIS3_MR_json.csv"

DATE_WINDOW = 365  # must match build_oasis3_rsfmri_subject_lists.py::DATE_WINDOW

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
# Step 1: Load diagnoses and derive per-visit diagnosis code (identical to
# build_oasis3_rsfmri_subject_lists.py)
# ---------------------------------------------------------------------------
print("Loading diagnoses ...")
d1 = pd.read_csv(DIAGNOSES_PATH, low_memory=False)
missing = [c for c in REQUIRED_D1_COLS if c not in d1.columns]
if missing:
    raise ValueError(f"{DIAGNOSES_PATH} is missing required column(s): {missing}")

d1["days_to_visit"] = pd.to_numeric(d1["days_to_visit"], errors="coerce")
d1 = d1.dropna(subset=["days_to_visit"]).copy()
d1["days_to_visit"] = d1["days_to_visit"].astype(int)

is_cn = d1["NORMCOG"] == 1
is_mci = (d1[MCI_COLS] == 1).any(axis=1)
is_ad = (d1["DEMENTED"] == 1) & (
    (d1["PROBAD"] == 1) | ((d1["alzdis"] == 1) & (d1["alzdisif"] == 1))
)

d1["DIAGNOSIS"] = np.select([is_cn, is_mci, is_ad], [1, 2, 3], default=np.nan)
d1 = d1.dropna(subset=["DIAGNOSIS"]).copy()
d1["DIAGNOSIS"] = d1["DIAGNOSIS"].astype(int)
d1 = d1.sort_values(["OASISID", "days_to_visit"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 2: Classify each subject as converter / non-converter / other
# (identical to build_oasis3_rsfmri_subject_lists.py::_classify)
# ---------------------------------------------------------------------------
def _classify(group: pd.DataFrame):
    g = group.sort_values("days_to_visit")
    diag = g["DIAGNOSIS"].values

    if 2 not in diag:
        return None, None, None

    first_mci_days = g[g["DIAGNOSIS"] == 2]["days_to_visit"].min()
    post_mci_dx3 = g[(g["DIAGNOSIS"] == 3) & (g["days_to_visit"] > first_mci_days)]
    if len(post_mci_dx3) > 0:
        first_conversion_days = post_mci_dx3["days_to_visit"].min()
        return "converter", first_mci_days, first_conversion_days

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

print(
    f"Classified: "
    f"{sum(1 for v in subject_meta.values() if v['label'] == 'converter')} converters, "
    f"{sum(1 for v in subject_meta.values() if v['label'] == 'non_converter_stable_mci')} non-converters"
)


# ---------------------------------------------------------------------------
# Step 3: Build the relevant visit window per classified subject (identical
# restriction to build_oasis3_rsfmri_subject_lists.py Step 3)
# ---------------------------------------------------------------------------
visit_rows = []
for oasisid, meta in subject_meta.items():
    subj = d1[d1["OASISID"] == oasisid].copy()
    label = meta["label"]

    if label == "converter":
        window = subj[
            (subj["days_to_visit"] >= meta["first_mci_days"])
            & (subj["days_to_visit"] <= meta["first_conversion_days"])
        ].copy()
    else:  # non_converter_stable_mci
        window = subj[subj["DIAGNOSIS"] == 2].copy()

    window["label"] = label
    visit_rows.append(window)

visits = pd.concat(visit_rows, ignore_index=True)
print(
    f"Visit-window rows: {len(visits):,} across {visits['OASISID'].nunique():,} classified subjects"
)


# ---------------------------------------------------------------------------
# Step 4: Load rsfMRI session inventory (identical to
# build_oasis3_rsfmri_subject_lists.py Step 4)
# ---------------------------------------------------------------------------
print("Loading rsfMRI scan inventory ...")
mr = pd.read_csv(MR_PATH, low_memory=False)
mr.columns = [c.strip() for c in mr.columns]

rest = mr[mr["scan category"] == "bold-rest"].copy()
rest = rest[rest["SeriesDescription"] != "rsfmri_ref"].copy()


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
rest["fmri_tr"] = rest["RepetitionTime"] * 1000.0

sessions = (
    rest.sort_values("label")
    .drop_duplicates(subset=["subject_id", "days_image"], keep="first")
    .copy()
)
sessions = sessions.rename(columns={"label": "image_id", "SeriesDescription": "fmri_description"})

print(
    f"rsfMRI: {len(rest):,} scan rows -> {len(sessions):,} sessions, "
    f"{sessions['subject_id'].nunique():,} subjects"
)


# ---------------------------------------------------------------------------
# Step 5: For every on-disk session of every classified subject, compute the
# gap to the *nearest* visit in that subject's relevant window. A subject is
# "fully unmatched" iff every one of their sessions has min-gap > DATE_WINDOW.
# ---------------------------------------------------------------------------
rows = []
for subject_id, sess_grp in sessions.groupby("subject_id"):
    if subject_id not in subject_meta:
        continue  # not classified converter/non-converter -- out of scope

    meta = subject_meta[subject_id]
    subj_visits = visits[visits["OASISID"] == subject_id]
    if subj_visits.empty:
        continue  # should not happen given Step 3, but guard anyway

    visit_days = subj_visits["days_to_visit"].to_numpy()

    for _, sess in sess_grp.iterrows():
        days_image = int(sess["days_image"])
        diffs = np.abs(visit_days - days_image)
        nearest_idx = int(np.argmin(diffs))
        nearest_gap = int(diffs[nearest_idx])
        nearest_visit_day = int(visit_days[nearest_idx])

        rows.append(
            {
                "subject_id": subject_id,
                "label": meta["label"],
                "first_mci_days": meta["first_mci_days"],
                "first_conversion_days": meta["first_conversion_days"],
                "image_id": sess["image_id"],
                "days_image": days_image,
                "fmri_description": sess["fmri_description"],
                "fmri_tr_ms": sess["fmri_tr"],
                "n_visits_in_window": len(subj_visits),
                "visit_window_days_span": f"{int(visit_days.min())}-{int(visit_days.max())}",
                "nearest_visit_days_to_visit": nearest_visit_day,
                "gap_to_nearest_visit_days": nearest_gap,
                "within_365_window": nearest_gap <= DATE_WINDOW,
            }
        )

report = pd.DataFrame(rows).sort_values(["subject_id", "days_image"]).reset_index(drop=True)

# A subject is "fully unmatched" iff NONE of its sessions are within the window
subject_min_gap = report.groupby("subject_id")["gap_to_nearest_visit_days"].min()
unmatched_subjects = subject_min_gap[subject_min_gap > DATE_WINDOW].index

unmatched_report = report[report["subject_id"].isin(unmatched_subjects)].copy()
unmatched_report["gap_years"] = (unmatched_report["gap_to_nearest_visit_days"] / 365.25).round(2)

print(f"\nFully unmatched subjects: {len(unmatched_subjects)}")
print(f"Their on-disk sessions:   {len(unmatched_report)}")

# ---------------------------------------------------------------------------
# Step 6: Write outputs
# ---------------------------------------------------------------------------
csv_path = OUT_DIR / "unmatched_subjects_sessions.csv"
unmatched_report.to_csv(csv_path, index=False)
print(f"\nWrote: {csv_path}")

# Per-subject summary (one row per subject, not per session)
subj_summary = (
    unmatched_report.groupby(["subject_id", "label"])
    .agg(
        n_sessions=("image_id", "count"),
        first_mci_days=("first_mci_days", "first"),
        first_conversion_days=("first_conversion_days", "first"),
        visit_window_days_span=("visit_window_days_span", "first"),
        min_gap_days=("gap_to_nearest_visit_days", "min"),
        max_gap_days=("gap_to_nearest_visit_days", "max"),
    )
    .reset_index()
)
subj_summary["min_gap_years"] = (subj_summary["min_gap_days"] / 365.25).round(2)
subj_summary["max_gap_years"] = (subj_summary["max_gap_days"] / 365.25).round(2)
subj_summary = subj_summary.sort_values("min_gap_days").reset_index(drop=True)

summary_csv_path = OUT_DIR / "unmatched_subjects_summary.csv"
subj_summary.to_csv(summary_csv_path, index=False)
print(f"Wrote: {summary_csv_path}")

# ---------------------------------------------------------------------------
# Step 7: Markdown report
# ---------------------------------------------------------------------------
md_lines = []
md_lines.append("# OASIS3 rsfMRI: subjects with no in-window clinical match\n\n")
md_lines.append(
    f"Generated by `build_unmatched_subjects_report.py`, reusing the exact "
    f"classification and ±{DATE_WINDOW}-day matching window from "
    f"`build_oasis3_rsfmri_subject_lists.py::attach_fmri()`.\n\n"
)
md_lines.append(
    f"**{len(unmatched_subjects)} subjects** ({len(unmatched_report)} on-disk rsfMRI "
    f"sessions) have *no* session landing within {DATE_WINDOW} days of any labeled "
    f"visit in their relevant window (converter: MCI→AD span; non-converter: "
    f"MCI-only visits). None of their sessions can be assigned a diagnosis label "
    f"under the current matching rule.\n\n"
)
md_lines.append(
    f"- Median gap (nearest session to nearest visit): "
    f"{subj_summary['min_gap_days'].median():.0f} days "
    f"({(subj_summary['min_gap_days'].median() / 365.25):.2f} years)\n"
    f"- Mean gap: {subj_summary['min_gap_days'].mean():.0f} days\n"
    f"- Max gap: {subj_summary['min_gap_days'].max():.0f} days "
    f"({(subj_summary['min_gap_days'].max() / 365.25):.2f} years)\n"
    f"- Min gap (closest miss): {subj_summary['min_gap_days'].min():.0f} days\n\n"
)

md_lines.append("## Why they can't be matched\n\n")
md_lines.append(
    "For each subject, every on-disk rsfMRI session's acquisition day "
    "(`days_image`, the `d####` suffix in the OASIS3 session label) is compared "
    "against every labeled clinical visit day (`days_to_visit`) in that subject's "
    f"relevant window, and the smallest gap is kept. If that smallest gap still "
    f"exceeds {DATE_WINDOW} days, the subject has no scan close enough in time to "
    "any visit that carries an MCI/AD diagnosis inside the window the classifier "
    "cares about -- i.e. the imaging and the clinical follow-up happened at "
    "different, non-overlapping points in the subject's disease trajectory. This "
    "is not a parsing or matching-window-implementation issue: gaps run from "
    f"{subj_summary['min_gap_days'].min()} days up to "
    f"{subj_summary['min_gap_days'].max()} days ("
    f"{(subj_summary['min_gap_days'].max() / 365.25):.1f} years), so widening the "
    f"{DATE_WINDOW}-day window would import label noise rather than recover "
    "legitimate matches (see README.md §4 for the window rationale).\n\n"
)

md_lines.append("## Per-subject detail\n\n")
md_lines.append(
    "| subject_id | label | n_sessions | visit window (days) | min gap | max gap |\n"
    "|---|---|---|---|---|---|\n"
)
for _, r in subj_summary.iterrows():
    md_lines.append(
        f"| {r['subject_id']} | {r['label']} | {r['n_sessions']} | "
        f"{r['visit_window_days_span']} | {r['min_gap_days']:.0f}d "
        f"({r['min_gap_years']:.2f}y) | {r['max_gap_days']:.0f}d "
        f"({r['max_gap_years']:.2f}y) |\n"
    )

md_lines.append(
    "\nFull per-session breakdown (every on-disk rsfMRI session, its "
    "`days_image`, and its gap to the nearest labeled visit): "
    "`unmatched_subjects_sessions.csv`.\n"
    "Per-subject summary (one row per subject): `unmatched_subjects_summary.csv`.\n"
)

md_path = OUT_DIR / "unmatched_subjects_report.md"
md_path.write_text("".join(md_lines))
print(f"Wrote: {md_path}")

if len(unmatched_subjects) != 60:
    print(
        f"\nNOTE: found {len(unmatched_subjects)} unmatched subjects, not the "
        "60 referenced in the prior conversation. Source CSVs may have been "
        "regenerated since -- see the printed counts above for the current "
        "classification/session totals.",
        file=sys.stderr,
    )
