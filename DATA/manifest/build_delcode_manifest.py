"""Build DATA/DELCODE/__metadata__/cohort_manifest.csv.

DELCODE is the reference cohort: its filenames already carry a nominal
protocol month (``_M<n>_``) rather than elapsed days, so ``delta_t_months``
here is defined as that month value directly (see
``CLASSIFIER.common.visits.visit_identity``) — this is what makes the A.2
reproduction gate ("DELCODE-trained GEGRU must still score AUC 0.8321 exactly
after the visit-parsing refactor") satisfiable.
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from CLASSIFIER.common.visits import parse_month, visit_identity
from DATA.manifest.schema import validate_manifest

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FMRI_ROOT = _REPO_ROOT / "DATA" / "DELCODE" / "__fmri_wholebrain_sch200_flat__" / "fmri"
DEFAULT_FC_ROOT = _REPO_ROOT / "DATA" / "DELCODE" / "__fc_wholebrain_sch200_flat__" / "matrices"
DEFAULT_COHORTS_CSV = _REPO_ROOT / "DATA" / "DELCODE" / "__metadata__" / "cohorts_with_scans_on_disk.csv"
DEFAULT_OUTPUT_CSV = _REPO_ROOT / "DATA" / "DELCODE" / "__metadata__" / "cohort_manifest.csv"

_SES_M_RE = re.compile(r"(ses-\d+)_M(\d+)_")
_FC_SUFFIX = "_whole_brain_correlation_matrix_z_transformed.npz"


def _scan_date_to_date(value: str) -> datetime | None:
    """cohorts_with_scans_on_disk.csv scan_date is 'DD-MM-YYYY'; pseudonymized, best-effort."""
    try:
        return datetime.strptime(str(value), "%d-%m-%Y")
    except ValueError:
        return None


def _fc_path_for(bold_path: Path, fc_root: Path) -> str | None:
    stem = bold_path.name.removesuffix(".nii.gz")
    candidate = fc_root / f"{stem}{_FC_SUFFIX}"
    return str(candidate) if candidate.exists() else None


def build_delcode_manifest(
    *,
    fmri_root: Path = DEFAULT_FMRI_ROOT,
    fc_root: Path = DEFAULT_FC_ROOT,
    cohorts_csv: Path = DEFAULT_COHORTS_CSV,
) -> pd.DataFrame:
    cohorts = pd.read_csv(cohorts_csv)
    cohorts["Pseudonym"] = cohorts["Pseudonym"].astype(str)
    label_lookup = {
        (row["Pseudonym"], int(str(row["visit"]).removeprefix("M"))): row["diagnosis"]
        for _, row in cohorts.iterrows()
    }
    scan_date_lookup = {
        (row["Pseudonym"], int(str(row["visit"]).removeprefix("M"))): _scan_date_to_date(
            row["scan_date"]
        )
        for _, row in cohorts.iterrows()
    }

    sessions_by_subject: dict[str, list[tuple[int, Path]]] = {}
    for subject_dir in sorted(fmri_root.glob("sub-*")):
        if not subject_dir.is_dir():
            continue
        subject_id = subject_dir.name.removeprefix("sub-")
        for bold_path in sorted(subject_dir.glob("*.nii.gz")):
            month = parse_month(bold_path.name)
            if month is None:
                continue
            sessions_by_subject.setdefault(subject_id, []).append((month, bold_path))

    rows: list[dict] = []
    for subject_id, sessions in sessions_by_subject.items():
        sessions = sorted(sessions, key=lambda s: s[0])
        months = [m for m, _ in sessions]
        visit_index, delta_t_months = visit_identity("delcode", months)

        baseline_date = scan_date_lookup.get((subject_id, months[0]))
        for idx, (month, bold_path) in enumerate(sessions):
            ses_match = _SES_M_RE.search(bold_path.name)
            ses_token = ses_match.group(1) if ses_match else f"ses-{idx + 1:02d}"
            scan_date = scan_date_lookup.get((subject_id, month))
            days_from_baseline = (
                (scan_date - baseline_date).days
                if scan_date is not None and baseline_date is not None
                else None
            )
            rows.append(
                {
                    "cohort": "delcode",
                    "subject_id": subject_id,
                    "session_id": f"{ses_token}_M{month}",
                    "days_from_baseline": days_from_baseline,
                    "visit_index": visit_index[idx],
                    "protocol_month": month,
                    "delta_t_months": delta_t_months[idx],
                    "bold_path": str(bold_path),
                    "fc_path": _fc_path_for(bold_path, fc_root),
                    "label": label_lookup.get((subject_id, month)),
                    "scanner_vendor": None,
                    "scanner_model": None,
                    "site": None,
                }
            )

    from DATA.manifest.schema import MANIFEST_COLUMNS

    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV)
    args = parser.parse_args(argv)

    df = build_delcode_manifest()
    subject_dirs = {p.name.removeprefix("sub-") for p in DEFAULT_FMRI_ROOT.glob("sub-*") if p.is_dir()}
    summary = validate_manifest(df, cohort="delcode", subject_dirs_on_disk=subject_dirs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"delcode: {summary['subjects']} subjects, {summary['sessions']} sessions -> {args.output}")


if __name__ == "__main__":
    main()
