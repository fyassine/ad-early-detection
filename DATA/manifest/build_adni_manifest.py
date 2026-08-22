"""Build DATA/ADNI/__metadata__/cohort_manifest.csv.

ADNI's flat product encodes elapsed days from baseline in the session token
(``ses-d<days>``, see ``DATA.manifest._day_coded``) rather than DELCODE's
nominal protocol month. ``protocol_month`` is recovered per session by
matching it to the closest dated visit row in the Converters/NonConverters
label CSVs (within ``PROTOCOL_MONTH_TOLERANCE_DAYS``) and parsing that row's
viscode with ``CLASSIFIER.common.visits.parse_adni_protocol_month`` — which
returns ``None`` for ADNI's unscheduled ``'v'``-coded visits rather than
guessing a month, per §2 of the comparison plan.

Expected counts (DOCS/meetings/ninth-meeting/comparison-plan-v2.md §7,
verified 2026-08-22): 268 subjects, 674 sessions in
``__fmri_wholebrain_sch200_flat__``. Was 237/567 as of 2026-08-20 — the
2026-08-21 evening ``postprocess_local.sh --flatten-only --overwrite`` run
both applied the reorientation-affine fix (§3, A.3) and caught up the 31
ADNI subjects that were denoised but stuck unflattened (same bug family as
OASIS-3's §1.3 empty-dir gap), landing on 268/272 eligible (4 excluded by
motion QC) with zero duplicate-same-day sessions. Re-run §7's count block
and update these after any further postprocessing pass over the
late-arriving ADNI subjects — do not assume the numbers grew on their own.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from CLASSIFIER.common.visits import parse_adni_protocol_month, visit_identity
from DATA.manifest._day_coded import iter_flat_sessions, subject_dirs_on_disk
from DATA.manifest.load import fc_path_for
from DATA.manifest.schema import MANIFEST_COLUMNS, assert_fc_paths_present, validate_manifest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_METADATA_DIR = _REPO_ROOT / "DATA" / "ADNI" / "__metadata__"
DEFAULT_FMRI_ROOT = _REPO_ROOT / "DATA" / "ADNI" / "__fmri_wholebrain_sch200_flat__" / "fmri"
DEFAULT_FC_ROOT = _REPO_ROOT / "DATA" / "ADNI" / "__fc_wholebrain_sch200_flat__" / "matrices"
DEFAULT_OUTPUT_CSV = _METADATA_DIR / "cohort_manifest.csv"

EXPECTED_SUBJECTS = 268
EXPECTED_SESSIONS = 674

# How close a session's elapsed-day count must be to a label-CSV row's
# elapsed-day-from-baseline (derived from its examdate) to inherit that row's
# viscode / image_id. Wider than the fMRI<->clinical-visit date_diff_days
# window (365d) used upstream, but still tight enough not to borrow a
# protocol month from a visit years away.
PROTOCOL_MONTH_TOLERANCE_DAYS = 45


def _to_flat_subject_id(original_subject_id: str) -> str:
    """'002_S_0729' -> 'ADNI002S0729' (matches DATA/ADNI/src/unzip's to_bids_subject, sans 'sub-')."""
    return f"ADNI{original_subject_id.replace('_', '')}"


def _site_code(original_subject_id: str) -> str | None:
    parts = original_subject_id.split("_S_")
    return parts[0] if len(parts) == 2 else None


def _latest_csv(pattern: str) -> Path:
    matches = sorted(_METADATA_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No file matching {_METADATA_DIR / pattern} found.")
    return matches[-1]


def _load_label_ids(csv_path: Path) -> set[str]:
    return set(pd.read_csv(csv_path)["subject_id"].astype(str).unique())


def _load_visit_rows(csv_path: Path) -> pd.DataFrame:
    """One row per (subject_id, examdate) with days_from_baseline + viscode + image_id."""
    df = pd.read_csv(csv_path)
    baselines = pd.read_csv(_METADATA_DIR / "adni_visit_baselines.csv").set_index("subject_id")[
        "baseline_date"
    ]
    df = df.dropna(subset=["examdate"]).copy()
    df["examdate"] = pd.to_datetime(df["examdate"], errors="coerce")
    df["_baseline_date"] = df["subject_id"].astype(str).map(baselines).apply(pd.to_datetime)
    df = df.dropna(subset=["examdate", "_baseline_date"])
    df["days_from_baseline"] = (df["examdate"] - df["_baseline_date"]).dt.days
    return df[["subject_id", "days_from_baseline", "viscode", "image_id"]]


def _scanner_lookup() -> dict[int, tuple[str | None, str | None]]:
    path = _METADATA_DIR / "All_Subjects_Functional_MRI_Images_12May2026.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path).dropna(subset=["image_id"])
    return {
        int(row["image_id"]): (row.get("fmri_mfr"), row.get("fmri_mfr_model"))
        for _, row in df.iterrows()
    }


def _nearest_visit_row(rows: pd.DataFrame, day: int) -> pd.Series | None:
    if rows.empty:
        return None
    diffs = (rows["days_from_baseline"] - day).abs()
    idx = diffs.idxmin()
    if diffs.loc[idx] > PROTOCOL_MONTH_TOLERANCE_DAYS:
        return None
    return rows.loc[idx]


def build_adni_manifest(
    *,
    fmri_root: Path = DEFAULT_FMRI_ROOT,
    fc_root: Path = DEFAULT_FC_ROOT,
    converters_csv: Path | None = None,
    non_converters_csv: Path | None = None,
) -> pd.DataFrame:
    converters_csv = converters_csv or _latest_csv("Extended_rsfMRI_MCI_Converters_*.csv")
    non_converters_csv = non_converters_csv or _latest_csv("Extended_rsfMRI_MCI_NonConverters_*.csv")

    converter_ids = _load_label_ids(converters_csv)
    stable_ids = _load_label_ids(non_converters_csv)
    from DATA.manifest.schema import assert_no_cross_label_duplicates

    assert_no_cross_label_duplicates(converter_ids, stable_ids, cohort="adni")

    visit_rows = pd.concat(
        [_load_visit_rows(converters_csv), _load_visit_rows(non_converters_csv)],
        ignore_index=True,
    )
    visit_rows_by_subject = {
        sid: g for sid, g in visit_rows.groupby("subject_id", group_keys=False)
    }
    scanner_lookup = _scanner_lookup()

    sessions_by_flat_id: dict[str, list] = {}
    for session in iter_flat_sessions(fmri_root):
        sessions_by_flat_id.setdefault(session.subject_id, []).append(session)

    original_id_by_flat_id: dict[str, str] = {
        _to_flat_subject_id(original_id): original_id for original_id in converter_ids | stable_ids
    }

    rows: list[dict] = []

    for flat_id, sessions in sessions_by_flat_id.items():
        sessions = sorted(sessions, key=lambda s: s.day)
        days = [s.day for s in sessions]
        visit_index, delta_t_months = visit_identity("adni", days)

        original_id = original_id_by_flat_id.get(flat_id)
        label = (
            "converter"
            if original_id in converter_ids
            else "stable"
            if original_id in stable_ids
            else None
        )
        rows_for_subject = visit_rows_by_subject.get(original_id, pd.DataFrame())

        for idx, session in enumerate(sessions):
            nearest = _nearest_visit_row(rows_for_subject, session.day)
            protocol_month = (
                parse_adni_protocol_month(str(nearest["viscode"])) if nearest is not None else None
            )
            scanner_vendor = scanner_model = None
            if nearest is not None and pd.notna(nearest["image_id"]):
                scanner_vendor, scanner_model = scanner_lookup.get(
                    int(nearest["image_id"]), (None, None)
                )
            fc_candidate = fc_path_for(session.bold_path, fc_root)
            rows.append(
                {
                    "cohort": "adni",
                    "subject_id": flat_id,
                    "session_id": f"ses-d{session.day:04d}",
                    "days_from_baseline": session.day,
                    "visit_index": visit_index[idx],
                    "protocol_month": protocol_month,
                    "delta_t_months": delta_t_months[idx],
                    "bold_path": str(session.bold_path),
                    "fc_path": str(fc_candidate) if fc_candidate.exists() else None,
                    "label": label,
                    "scanner_vendor": scanner_vendor,
                    "scanner_model": scanner_model,
                    "site": _site_code(original_id) if original_id else None,
                }
            )

    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--fc-root", type=Path, default=DEFAULT_FC_ROOT)
    parser.add_argument(
        "--require-fc",
        action="store_true",
        help="Fail loudly if any session's fc_path is missing (run after FC extraction).",
    )
    args = parser.parse_args(argv)

    df = build_adni_manifest(fc_root=args.fc_root)
    summary = validate_manifest(
        df,
        cohort="adni",
        subject_dirs_on_disk=subject_dirs_on_disk(DEFAULT_FMRI_ROOT),
        expected_subjects=EXPECTED_SUBJECTS,
        expected_sessions=EXPECTED_SESSIONS,
    )
    if args.require_fc:
        assert_fc_paths_present(df, cohort="adni")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"adni: {summary['subjects']} subjects, {summary['sessions']} sessions -> {args.output}")


if __name__ == "__main__":
    main()
