"""Build DATA/OASIS3/__metadata__/cohort_manifest.csv.

OASIS-3 shares ADNI's day-coded session convention (``ses-d<days>``) but has
no clean scheduled-visit-month code in its label CSVs the way ADNI's viscodes
do (its own ``viscode`` values are opaque UDS-form-visit IDs, e.g.
``OAS30007_UDSd1_d2617``), so ``protocol_month`` is left ``None`` for every
OASIS-3 session rather than guessed — per §2 of the comparison plan, a wrong
guess here is exactly the failure mode this manifest exists to prevent.

Per §1.3 of the plan (as written 2026-08-20), 46 of the 128 OASIS-3 subject
directories had fMRIPrep output but zero postprocessed BOLD — an empty-dir
bug, not a genuine exclusion. As of 2026-08-21 that gap is closed: the
postprocessed flat product now matches the fMRIPrep flat product exactly
(128 dirs, 0 empty, 239 sessions for both — re-run §7's two count blocks to
confirm), so the §5 triage this module warns about appears to have already
landed. The ``--acknowledge-empty-subjects`` escape hatch and the build-time
assertion it exists to bypass are kept regardless, since a future re-run over
different data could reintroduce empty subject directories and this is
exactly the class of bug A.0 exists to catch immediately rather than three
weeks later.

Expected counts (re-verified 2026-08-21, superseding the plan's 2026-08-20
snapshot): 128 contributing subjects, 239 sessions.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from CLASSIFIER.common.visits import visit_identity
from DATA.manifest._day_coded import iter_flat_sessions, subject_dirs_on_disk
from DATA.manifest.schema import MANIFEST_COLUMNS, assert_no_cross_label_duplicates, validate_manifest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_METADATA_DIR = _REPO_ROOT / "DATA" / "OASIS3" / "__metadata__"
DEFAULT_FMRI_ROOT = _REPO_ROOT / "DATA" / "OASIS3" / "__fmri_wholebrain_sch200_flat__" / "fmri"
DEFAULT_OUTPUT_CSV = _METADATA_DIR / "cohort_manifest.csv"

EXPECTED_SUBJECTS = 128
EXPECTED_SESSIONS = 239

# OASIS3_MR_json.csv is inconsistent about 'ses-' vs 'sess-' in its filename column.
_MR_JSON_SESSION_RE = re.compile(r"sub-(OAS\d+)_sess?-d(\d+)")


def _subjects_with_duplicate_days(df: pd.DataFrame) -> set[str]:
    """Subject IDs with >1 session sharing the same days_from_baseline."""
    counts = df.groupby(["subject_id", "days_from_baseline"]).size()
    return {subject_id for subject_id, _ in counts[counts > 1].index}


def _latest_csv(pattern: str) -> Path:
    matches = sorted(_METADATA_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No file matching {_METADATA_DIR / pattern} found.")
    return matches[-1]


def _load_label_ids(csv_path: Path) -> set[str]:
    return set(pd.read_csv(csv_path)["subject_id"].astype(str).unique())


def _scanner_lookup() -> dict[tuple[str, int], tuple[str | None, str | None]]:
    path = _METADATA_DIR / "OASIS3_MR_json.csv"
    if not path.exists():
        return {}
    lookup: dict[tuple[str, int], tuple[str | None, str | None]] = {}
    df = pd.read_csv(path, usecols=["subject_id", "filename", "Manufacturer", "ManufacturersModelName"])
    for _, row in df.iterrows():
        m = _MR_JSON_SESSION_RE.search(str(row["filename"]))
        if m is None:
            continue
        key = (m.group(1), int(m.group(2)))
        if key not in lookup:
            lookup[key] = (row.get("Manufacturer"), row.get("ManufacturersModelName"))
    return lookup


def build_oasis3_manifest(
    *,
    fmri_root: Path = DEFAULT_FMRI_ROOT,
    converters_csv: Path | None = None,
    non_converters_csv: Path | None = None,
) -> pd.DataFrame:
    converters_csv = converters_csv or _latest_csv("Extended_rsfMRI_MCI_Converters_*.csv")
    non_converters_csv = non_converters_csv or _latest_csv("Extended_rsfMRI_MCI_NonConverters_*.csv")

    converter_ids = _load_label_ids(converters_csv)
    stable_ids = _load_label_ids(non_converters_csv)
    assert_no_cross_label_duplicates(converter_ids, stable_ids, cohort="oasis3")

    scanner_lookup = _scanner_lookup()

    sessions_by_subject: dict[str, list] = {}
    for session in iter_flat_sessions(fmri_root):
        sessions_by_subject.setdefault(session.subject_id, []).append(session)

    rows: list[dict] = []
    for subject_id, sessions in sessions_by_subject.items():
        sessions = sorted(sessions, key=lambda s: s.day)
        days = [s.day for s in sessions]
        visit_index, delta_t_months = visit_identity("oasis3", days)

        label = (
            "converter"
            if subject_id in converter_ids
            else "stable"
            if subject_id in stable_ids
            else None
        )

        for idx, session in enumerate(sessions):
            scanner_vendor, scanner_model = scanner_lookup.get(
                (subject_id, session.day), (None, None)
            )
            rows.append(
                {
                    "cohort": "oasis3",
                    "subject_id": subject_id,
                    "session_id": f"ses-d{session.day:04d}",
                    "days_from_baseline": session.day,
                    "visit_index": visit_index[idx],
                    "protocol_month": None,
                    "delta_t_months": delta_t_months[idx],
                    "bold_path": str(session.bold_path),
                    "fc_path": None,
                    "label": label,
                    "scanner_vendor": scanner_vendor,
                    "scanner_model": scanner_model,
                    "site": None,
                }
            )

    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--acknowledge-empty-subjects",
        action="store_true",
        help=(
            "Explicitly acknowledge the §1.3 empty-dir subjects (fMRIPrep output, "
            "zero postprocessed BOLD) and build the manifest anyway. Without this, "
            "the build fails loudly and lists them."
        ),
    )
    parser.add_argument(
        "--acknowledge-duplicate-day-sessions",
        action="store_true",
        help=(
            "Explicitly acknowledge subjects with two BOLD scans on the same "
            "elapsed day (e.g. paired task-restingstate / task-restingstateMB4 "
            "acquisitions) and build the manifest anyway. Without this, the "
            "build fails loudly and lists them — see assert_delta_t_monotonic."
        ),
    )
    args = parser.parse_args(argv)

    df = build_oasis3_manifest()
    on_disk = subject_dirs_on_disk(DEFAULT_FMRI_ROOT)
    contributing = set(df["subject_id"].unique())
    known_empty = frozenset(on_disk - contributing) if args.acknowledge_empty_subjects else frozenset()
    known_duplicate_day = (
        frozenset(_subjects_with_duplicate_days(df))
        if args.acknowledge_duplicate_day_sessions
        else frozenset()
    )

    summary = validate_manifest(
        df,
        cohort="oasis3",
        subject_dirs_on_disk=on_disk,
        expected_subjects=EXPECTED_SUBJECTS,
        expected_sessions=EXPECTED_SESSIONS,
        known_empty_subjects=known_empty,
        known_duplicate_day_subjects=known_duplicate_day,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"oasis3: {summary['subjects']} subjects, {summary['sessions']} sessions -> {args.output}")
    if summary["acknowledged_empty_subjects"]:
        print(
            f"  acknowledged {len(summary['acknowledged_empty_subjects'])} empty "
            f"subject dir(s) (pending §5 triage): {summary['acknowledged_empty_subjects']}"
        )
    if summary["acknowledged_duplicate_day_subjects"]:
        print(
            f"  acknowledged {len(summary['acknowledged_duplicate_day_subjects'])} subject(s) "
            f"with same-day duplicate scans: {summary['acknowledged_duplicate_day_subjects']}"
        )


if __name__ == "__main__":
    main()
