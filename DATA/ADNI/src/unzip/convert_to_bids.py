#!/usr/bin/env python3
"""
convert_to_bids.py
====================
Converts every zip listed in DATA/ADNI/__metadata__/adni_bids_manifest.csv
(built by scan_zip_manifest.py) to NIfTI via dcm2niix and places the result
under DATA/ADNI/__bold_and_smri__/<dest_relpath> — the same
sub-*/ses-d<NNNN>/{anat,func}/ layout already used by
DATA/OASIS3/__bold_and_smri__, so both datasets can be fed to the same
Fritz BIDS-organization step.

Resumable: any row whose destination .nii.gz already exists is skipped.
dcm2niix failures are logged and skipped rather than aborting the run.

Usage
-----
    python convert_to_bids.py                       # convert everything
    python convert_to_bids.py --subjects 002_S_1261  # convert one subject
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2]  # .../ADNI/
DEFAULT_MANIFEST_CSV = DATA_DIR / "__metadata__" / "adni_bids_manifest.csv"
DEFAULT_BIDS_ROOT = DATA_DIR / "__bold_and_smri__"

DCM2NIIX_TIMEOUT_S = 180


def convert_row(zip_path: Path, dest_nii: Path) -> None:
    """
    Extracts zip_path's DICOMs to a temp dir next to it, runs dcm2niix, and
    moves the produced .nii.gz/.json to dest_nii / dest_nii.with_suffix('.json').
    Raises on any failure — the caller decides whether to skip and continue.
    """
    tmp_root = zip_path.parent / f"_convert_{zip_path.stem}"
    tmp_root.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_root)

        series_dirs = sorted({f.parent for f in tmp_root.rglob("*.dcm")})
        if not series_dirs:
            raise ValueError(f"No DICOM files found in {zip_path.name}")

        nii_tmp = tmp_root / "_nii"
        nii_tmp.mkdir(exist_ok=True)
        for sdir in series_dirs:
            subprocess.run(
                ["dcm2niix", "-z", "y", "-f", "%i_%s_%d", "-o", str(nii_tmp), str(sdir)],
                check=True,
                capture_output=True,
                timeout=DCM2NIIX_TIMEOUT_S,
            )

        niis = sorted(nii_tmp.glob("*.nii.gz"))
        if not niis:
            raise ValueError(f"dcm2niix produced no output for {zip_path.name}")
        # A zip contains exactly one series (per ADNI's zip-per-series
        # convention) — dcm2niix may still split it into multiple output
        # files (e.g. phase/magnitude); keep the largest one.
        chosen = max(niis, key=lambda p: p.stat().st_size)

        dest_nii.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(chosen), str(dest_nii))
        chosen_json = chosen.with_suffix("").with_suffix(".json")
        if chosen_json.exists():
            shutil.move(str(chosen_json), str(dest_nii.with_suffix("").with_suffix(".json")))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def load_manifest(manifest_csv: Path, subjects: set[str] | None) -> list[dict[str, str]]:
    if not manifest_csv.exists():
        raise FileNotFoundError(f"{manifest_csv} not found — run scan_zip_manifest.py first.")
    with manifest_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if subjects:
        rows = [r for r in rows if r["subject_id"] in subjects]
    return rows


def run_pass(rows: list[dict[str, str]], bids_root: Path) -> tuple[int, int, int]:
    """Returns (converted, skipped_existing, failed)."""
    converted = 0
    skipped_existing = 0
    failed = 0
    for row in rows:
        dest_nii = bids_root / row["dest_relpath"]
        if dest_nii.exists():
            skipped_existing += 1
            continue

        zip_path = Path(row["zip_path"])
        print(f"  -> {row['sub_bids']} {row['ses_label']} {row['scan_type']}: {zip_path.name}")
        try:
            convert_row(zip_path, dest_nii)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as e:
            print(f"     FAILED: {e}", file=sys.stderr)
            failed += 1
            continue
        converted += 1
    return converted, skipped_existing, failed


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest-csv", default=str(DEFAULT_MANIFEST_CSV))
    p.add_argument("--bids-root", default=str(DEFAULT_BIDS_ROOT))
    p.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Limit conversion to these ADNI subject_id values (e.g. 002_S_1261)",
    )
    args = p.parse_args()

    subjects = set(args.subjects) if args.subjects else None
    rows = load_manifest(Path(args.manifest_csv), subjects)
    print(f"Converting {len(rows)} manifest row(s)...")

    converted, skipped_existing, failed = run_pass(rows, Path(args.bids_root))
    print(f"Done: {converted} converted, {skipped_existing} already done, {failed} failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
