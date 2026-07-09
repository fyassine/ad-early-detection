#!/usr/bin/env python3
"""
scan_zip_manifest.py
=====================
Builds a manifest of every DICOM zip in DATA/ADNI/__dicom_zips_flat__,
classified as anat (T1w/MPRAGE) or func (resting-state BOLD), with a
computed BIDS destination path — sub-ADNI<SiteSubj>/ses-d<NNNN>/{anat,func}/... —
mirroring the layout already used by DATA/OASIS3/__bold_and_smri__.

Pure Python, no dcm2niix, no extraction — only peeks each zip's internal
first member name (ADNI/<subject>/<series_description>/<date_time>/<image_id>/
<file>.dcm) to read the series description and acquisition date directly, so
this does not depend on the coverage of any external metadata CSV. Safe and
fast to re-run.

The per-subject baseline date (from build_visit_baselines.py's output) is
required to compute ses-d<NNNN> — run that script first.

Usage
-----
    python scan_zip_manifest.py
"""

from __future__ import annotations

import csv
import re
import sys
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2]  # .../ADNI/
DEFAULT_ZIP_DIR = DATA_DIR / "__dicom_zips_flat__"
DEFAULT_BASELINES_CSV = DATA_DIR / "__metadata__" / "adni_visit_baselines.csv"
DEFAULT_OUTPUT_CSV = DATA_DIR / "__metadata__" / "adni_bids_manifest.csv"
DEFAULT_BIDS_ROOT = DATA_DIR / "__bold_and_smri__"

ZIP_NAME_RE = re.compile(r"^(\d{3}_S_\d{4})_(\d+)\.zip$")
INTERNAL_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})")

# Reused from DATA/ADNI/src/download/download_adni_smri.py
T1W_DESCRIPTION_RE = re.compile(r"MPRAGE|MP-RAGE|MP RAGE|SPGR|IR-SPGR|FSPGR|3D\s*T1", re.IGNORECASE)
# Reused from DATA/PREPROCESSING/src/fritz/run_fritz_pipeline.sh's BOLD detection regex
BOLD_DESCRIPTION_RE = re.compile(r"rsfmri|fcmri|fmri|resting|bold|rest", re.IGNORECASE)

OUTPUT_COLUMNS = [
    "zip_path",
    "subject_id",
    "sub_bids",
    "image_id",
    "series_description",
    "scan_type",
    "acquisition_date",
    "ses_label",
    "run_label",
    "dest_relpath",
]


@dataclass
class ZipEntry:
    zip_path: Path
    subject_id: str
    image_id: str
    series_description: str
    acquisition_date: str  # YYYY-MM-DD
    scan_type: str  # "anat" | "func"


def classify_series(series_description: str) -> str | None:
    if T1W_DESCRIPTION_RE.search(series_description):
        return "anat"
    if BOLD_DESCRIPTION_RE.search(series_description):
        return "func"
    return None


def peek_zip(zip_path: Path) -> tuple[str, str] | None:
    """
    Returns (series_description, acquisition_date) read from the zip's first
    member's internal path, or None if the zip is empty/unreadable/doesn't
    match the expected ADNI/<subject>/<series>/<date_time>/<image_id>/<file>
    layout.
    """
    try:
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
    except (zipfile.BadZipFile, OSError):
        return None
    if not names:
        return None

    parts = names[0].split("/")
    if len(parts) < 5:
        return None
    series_description, date_time = parts[2], parts[3]
    m = INTERNAL_DATE_RE.match(date_time)
    if not m:
        return None
    return series_description, m.group(1)


def load_baselines(baselines_csv: Path) -> dict[str, str]:
    if not baselines_csv.exists():
        raise FileNotFoundError(f"{baselines_csv} not found — run build_visit_baselines.py first.")
    baselines: dict[str, str] = {}
    with baselines_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            baselines[row["subject_id"]] = row["baseline_date"]
    return baselines


def to_bids_subject(subject_id: str) -> str:
    """002_S_1261 -> sub-ADNI002S1261 (matches run_fritz_pipeline.sh's mapping)."""
    return f"sub-ADNI{subject_id.replace('_', '')}"


def scan_zips(zip_dir: Path) -> tuple[list[ZipEntry], list[Path]]:
    """Returns (classified entries, unclassified/unreadable zip paths)."""
    entries: list[ZipEntry] = []
    unclassified: list[Path] = []
    for zip_path in sorted(zip_dir.glob("*.zip")):
        m = ZIP_NAME_RE.match(zip_path.name)
        if not m:
            unclassified.append(zip_path)
            continue
        subject_id, image_id = m.group(1), m.group(2)

        peeked = peek_zip(zip_path)
        if peeked is None:
            unclassified.append(zip_path)
            continue
        series_description, acquisition_date = peeked

        scan_type = classify_series(series_description)
        if scan_type is None:
            unclassified.append(zip_path)
            continue

        entries.append(
            ZipEntry(
                zip_path=zip_path,
                subject_id=subject_id,
                image_id=image_id,
                series_description=series_description,
                acquisition_date=acquisition_date,
                scan_type=scan_type,
            )
        )
    return entries, unclassified


def effective_baselines(entries: list[ZipEntry], dxsum_baselines: dict[str, str]) -> dict[str, str]:
    """
    Per-subject baseline date used for ses-d<NNNN>: the earlier of the DXSUM
    baseline visit and that subject's own earliest scan in __dicom_zips_flat__.
    DXSUM's "bl"/"4_bl" row is a clinical visit and can postdate an imaging
    scan (especially for the ~40% of subjects where build_visit_baselines.py
    had to fall back to their earliest-ever DXSUM row rather than a true
    baseline code) — taking the min guarantees every ses-d<NNNN> is >= 0
    instead of inventing a negative-day naming scheme OASIS3 doesn't have.
    """
    earliest_scan: dict[str, str] = {}
    for entry in entries:
        current = earliest_scan.get(entry.subject_id)
        if current is None or entry.acquisition_date < current:
            earliest_scan[entry.subject_id] = entry.acquisition_date

    effective: dict[str, str] = {}
    for subject_id, scan_date in earliest_scan.items():
        dxsum_date = dxsum_baselines.get(subject_id)
        effective[subject_id] = min(dxsum_date, scan_date) if dxsum_date else scan_date
    return effective


def build_manifest_rows(
    entries: list[ZipEntry], baselines: dict[str, str], bids_root: Path
) -> tuple[list[dict[str, str]], list[ZipEntry]]:
    """Returns (manifest rows, entries skipped for lacking a baseline date)."""
    rows: list[dict[str, str]] = []
    skipped: list[ZipEntry] = []

    # Group counts to decide run- suffixing: only added when >1 zip shares
    # the same (subject, ses_label, scan_type).
    group_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    computed: list[tuple[ZipEntry, str, str]] = []  # (entry, sub_bids, ses_label)

    for entry in entries:
        baseline_date = baselines.get(entry.subject_id)
        if baseline_date is None:
            skipped.append(entry)
            continue
        days = (date.fromisoformat(entry.acquisition_date) - date.fromisoformat(baseline_date)).days
        if days < 0:
            raise ValueError(
                f"{entry.zip_path.name}: negative days_since_baseline ({days}) for "
                f"subject {entry.subject_id} — effective_baselines() should have "
                "prevented this."
            )
        ses_label = f"ses-d{days:04d}"
        sub_bids = to_bids_subject(entry.subject_id)
        group_counts[(sub_bids, ses_label, entry.scan_type)] += 1
        computed.append((entry, sub_bids, ses_label))

    run_indices: dict[tuple[str, str, str], int] = defaultdict(int)
    for entry, sub_bids, ses_label in computed:
        key = (sub_bids, ses_label, entry.scan_type)
        run_label = ""
        # func always gets a run- suffix (even a lone acquisition), matching
        # DATA/OASIS3/__bold_and_smri__'s raw convention, which
        # organize_bids_dataset() in run_fritz_pipeline.sh depends on via its
        # "*_run-*_bold.nii.gz" glob. anat only gets one when the session has
        # more than one T1w acquisition (also matches OASIS3's raw layout).
        if entry.scan_type == "func" or group_counts[key] > 1:
            run_indices[key] += 1
            run_label = f"run-{run_indices[key]:02d}"

        if entry.scan_type == "anat":
            # sub-X_ses-Y[_run-NN]_T1w — run- (if any) precedes the suffix.
            name_parts = [sub_bids, ses_label]
            if run_label:
                name_parts.append(run_label)
            name_parts.append("T1w")
        else:
            # sub-X_ses-Y_task-rest[_run-NN]_bold — run- (if any) sits
            # between the task entity and "bold", matching OASIS3's raw
            # convention exactly (e.g. sub-OAS30001_ses-d3132_task-rest_run-01_bold).
            name_parts = [sub_bids, ses_label, "task-rest"]
            if run_label:
                name_parts.append(run_label)
            name_parts.append("bold")
        stem = "_".join(name_parts)
        dest_relpath = f"{sub_bids}/{ses_label}/{entry.scan_type}/{stem}.nii.gz"

        rows.append(
            {
                "zip_path": str(entry.zip_path),
                "subject_id": entry.subject_id,
                "sub_bids": sub_bids,
                "image_id": entry.image_id,
                "series_description": entry.series_description,
                "scan_type": entry.scan_type,
                "acquisition_date": entry.acquisition_date,
                "ses_label": ses_label,
                "run_label": run_label,
                "dest_relpath": dest_relpath,
            }
        )

    rows.sort(key=lambda r: (r["sub_bids"], r["ses_label"], r["scan_type"], r["dest_relpath"]))
    return rows, skipped


def write_manifest(rows: list[dict[str, str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zip-dir", default=str(DEFAULT_ZIP_DIR))
    p.add_argument("--baselines-csv", default=str(DEFAULT_BASELINES_CSV))
    p.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    p.add_argument("--bids-root", default=str(DEFAULT_BIDS_ROOT))
    args = p.parse_args()

    dxsum_baselines = load_baselines(Path(args.baselines_csv))
    entries, unclassified = scan_zips(Path(args.zip_dir))
    baselines = effective_baselines(entries, dxsum_baselines)
    rows, skipped = build_manifest_rows(entries, baselines, Path(args.bids_root))
    write_manifest(rows, Path(args.output_csv))

    n_anat = sum(1 for r in rows if r["scan_type"] == "anat")
    n_func = sum(1 for r in rows if r["scan_type"] == "func")
    print(f"Wrote {len(rows)} manifest rows -> {args.output_csv}")
    print(f"  anat: {n_anat}  func: {n_func}")
    print(f"  unclassified/unreadable zips: {len(unclassified)}")
    print(f"  skipped (no baseline date for subject): {len(skipped)}")
    if unclassified:
        print("  Sample unclassified:", [p.name for p in unclassified[:10]], file=sys.stderr)
    if skipped:
        print(
            "  Sample skipped subjects:",
            sorted({e.subject_id for e in skipped})[:10],
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
