"""Build DATA/{ADNI,OASIS3}/__metadata__/{cohort}_demographics.csv.

GELSTM (`CLASSIFIER/model/GELSTM/dataset.py`) feeds `sex` and `age` as model
inputs, sourced from DELCODE's `cohorts_with_scans_on_disk.csv` today. Neither
external cohort's manifest carries either field: ADNI's Converters/
NonConverters/Longitudinal label CSVs and `All_Subjects_DXSUM_*.csv` have no
demographics columns at all. As of 2026-08-21
`DATA/ADNI/__metadata__/__artifacts__/All_Subjects_PTDEMOG_21Aug2026.csv`
(ADNI's dedicated demographics table) covers this gap and matches all 237 manifest subjects. 9 subjects carry more
than one PTDEMOG row (a re-consent/"4_init" row alongside the original
screening row) with a PTDOB that drifts by a few months between rows —
resolved by keeping the earliest-`VISDATE` row per subject (an explicit,
auditable tie-break, not a silent pick; mirrors
`DATA/ADNI/src/unzip/build_visit_baselines.py`'s tiered-fallback pattern),
never by averaging or preferring gender/DOB independently.
OASIS-3's `__metadata__/others/OASIS3_demographics.csv` already exists and
covers all 128 manifest subjects, one row per subject.

`DATA/ADNI/__metadata__/__artifacts__/ADNIMERGE2.tar.gz` (an R package with
200+ `.rda` tables, including a `PTDEMOG.rda` duplicate of the CSV consumed
here) is deliberately left unextracted: it needs a new `pyreadr` dependency
and the CSV alone is already sufficient for `sex`/`age`. A future consumer
wanting a specific `.rda` table (APOE genotype, MMSE, ...) can add a targeted
extractor then.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]

_ADNI_METADATA_DIR = _REPO_ROOT / "DATA" / "ADNI" / "__metadata__"
_ADNI_ARTIFACTS_DIR = _ADNI_METADATA_DIR / "__artifacts__"
ADNI_DEFAULT_OUTPUT_CSV = _ADNI_METADATA_DIR / "adni_demographics.csv"

_OASIS3_METADATA_DIR = _REPO_ROOT / "DATA" / "OASIS3" / "__metadata__"
OASIS3_DEFAULT_DEMOGRAPHICS_CSV = _OASIS3_METADATA_DIR / "others" / "OASIS3_demographics.csv"
OASIS3_DEFAULT_OUTPUT_CSV = _OASIS3_METADATA_DIR / "oasis3_demographics.csv"

DEMOGRAPHICS_COLUMNS = ["subject_id", "sex", "age_at_baseline", "education", "source"]

_PTGENDER_MAP = {1: "m", 2: "f"}
_GENDER_MAP = {1: "m", 2: "f"}


def _to_flat_subject_id(original_subject_id: str) -> str:
    """'002_S_0729' -> 'ADNI002S0729' — matches build_adni_manifest._to_flat_subject_id."""
    return f"ADNI{original_subject_id.replace('_', '')}"


def _latest_csv(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No file matching {directory / pattern} found.")
    return matches[-1]


def build_adni_demographics(
    *,
    ptdemog_csv: Path | None = None,
    baselines_csv: Path = _ADNI_METADATA_DIR / "adni_visit_baselines.csv",
    manifest_csv: Path = _ADNI_METADATA_DIR / "cohort_manifest.csv",
) -> pd.DataFrame:
    ptdemog_csv = ptdemog_csv or _latest_csv(_ADNI_ARTIFACTS_DIR, "All_Subjects_PTDEMOG_*.csv")

    ptdemog = pd.read_csv(ptdemog_csv, low_memory=False)
    ptdemog["_original_id"] = ptdemog["PTID"].astype(str)
    ptdemog["_visdate"] = pd.to_datetime(ptdemog["VISDATE"], errors="coerce")

    baselines = pd.read_csv(baselines_csv).set_index("subject_id")["baseline_date"]
    baselines = pd.to_datetime(baselines)

    manifest = pd.read_csv(manifest_csv)
    needed_flat_ids = set(manifest["subject_id"].astype(str).unique())
    original_by_flat = {
        _to_flat_subject_id(orig): orig for orig in ptdemog["_original_id"].unique()
    }

    rows: list[dict] = []
    missing: list[str] = []
    for flat_id in sorted(needed_flat_ids):
        original_id = original_by_flat.get(flat_id)
        subject_rows = ptdemog[ptdemog["_original_id"] == original_id] if original_id else ptdemog.iloc[0:0]

        with_gender = subject_rows.dropna(subset=["PTGENDER"])
        with_gender = with_gender[with_gender["PTGENDER"].astype(int).isin(_PTGENDER_MAP)]
        with_dob = with_gender.dropna(subset=["PTDOB"])
        if with_dob.empty:
            missing.append(flat_id)
            continue

        # Multiple PTDEMOG rows per subject (e.g. an initial-screening row and
        # a later re-consent "4_init" row) can carry a PTDOB that drifts by a
        # few months — keep the earliest-VISDATE row rather than silently
        # picking the last one or averaging.
        canonical = with_dob.sort_values("_visdate", na_position="last").iloc[0]

        baseline_date = baselines.get(original_id)
        dob = pd.to_datetime(str(canonical["PTDOB"]), format="%m/%Y", errors="coerce")
        age_at_baseline = (
            (baseline_date - dob).days / 365.25
            if baseline_date is not None and pd.notna(dob)
            else None
        )
        educ = canonical.get("PTEDUCAT")

        rows.append(
            {
                "subject_id": flat_id,
                "sex": _PTGENDER_MAP[int(canonical["PTGENDER"])],
                "age_at_baseline": age_at_baseline,
                "education": int(educ) if pd.notna(educ) else None,
                "source": "PTDEMOG",
            }
        )

    if missing:
        raise ValueError(
            f"adni demographics: {len(missing)} manifest subject(s) have no usable PTGENDER/PTDOB "
            f"in {ptdemog_csv}: {missing[:20]}. Every manifest subject must resolve to a "
            "demographics row — see errors.md, no silent age=-1 default."
        )

    return pd.DataFrame(rows, columns=DEMOGRAPHICS_COLUMNS)


def build_oasis3_demographics(
    *,
    demographics_csv: Path = OASIS3_DEFAULT_DEMOGRAPHICS_CSV,
    manifest_csv: Path = _OASIS3_METADATA_DIR / "cohort_manifest.csv",
) -> pd.DataFrame:
    demographics = pd.read_csv(demographics_csv)
    demographics["OASISID"] = demographics["OASISID"].astype(str)
    by_subject = demographics.set_index("OASISID")

    manifest = pd.read_csv(manifest_csv)
    needed_ids = set(manifest["subject_id"].astype(str).unique())

    rows: list[dict] = []
    missing: list[str] = []
    for subject_id in sorted(needed_ids):
        if subject_id not in by_subject.index:
            missing.append(subject_id)
            continue
        row = by_subject.loc[subject_id]
        gender = row.get("GENDER")
        sex = _GENDER_MAP.get(int(gender)) if pd.notna(gender) else None
        if sex is None:
            missing.append(subject_id)
            continue
        rows.append(
            {
                "subject_id": subject_id,
                "sex": sex,
                # AgeatEntry is age at OASIS-3 study entry, not necessarily this
                # subject's rs-fMRI baseline visit — best-effort, same caveat as
                # DELCODE's brthdat-derived age (create_downstream_data_splits.py).
                "age_at_baseline": float(row["AgeatEntry"]) if pd.notna(row.get("AgeatEntry")) else None,
                "education": float(row["EDUC"]) if pd.notna(row.get("EDUC")) else None,
                "source": "OASIS3_demographics",
            }
        )

    if missing:
        raise ValueError(
            f"oasis3 demographics: {len(missing)} manifest subject(s) missing from "
            f"{demographics_csv} or lacking a usable GENDER value: {missing[:20]}. "
            "Every manifest subject must resolve to a demographics row."
        )

    return pd.DataFrame(rows, columns=DEMOGRAPHICS_COLUMNS)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=("adni", "oasis3", "all"), required=True)
    args = parser.parse_args(argv)

    cohorts = ("adni", "oasis3") if args.cohort == "all" else (args.cohort,)
    for cohort in cohorts:
        if cohort == "adni":
            df = build_adni_demographics()
            output = ADNI_DEFAULT_OUTPUT_CSV
        else:
            df = build_oasis3_demographics()
            output = OASIS3_DEFAULT_OUTPUT_CSV
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False)
        print(f"{cohort}: {len(df)} subjects -> {output}")


if __name__ == "__main__":
    main()
