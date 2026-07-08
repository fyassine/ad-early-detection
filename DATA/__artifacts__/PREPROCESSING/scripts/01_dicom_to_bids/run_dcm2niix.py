#!/usr/bin/env python3
"""Stage 1.1: DICOM -> NIfTI conversion via dcm2niix.

Converts every scan-series folder under <subject_dir>/SCANS/*/resources/DICOM/files/ into a
NIfTI + BIDS JSON sidecar pair in a flat per-subject staging directory. The JSON sidecar
(`-b y`) carries dcm2niix's own accurate per-subject values (RepetitionTime, EchoTime,
SliceTiming, ImageType, SeriesDescription, ...) — build_bids.py reads these directly instead
of hand-rolling sidecar content, which is what made the original BIDS_og.py hardcode a wrong
RepetitionTime for every subject.

Usage:
    python run_dcm2niix.py <subject_scans_dir> <staging_out_dir>

Example (SAMPLE subject):
    python run_dcm2niix.py \\
        SAMPLE/03a0a6663-M0_T1_01/SCANS \\
        staging/03a0a6663-M0_T1_01
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def find_dicom_dirs(scans_dir: Path) -> list[Path]:
    return sorted(
        d for d in scans_dir.glob("*/resources/DICOM/files")
        if d.is_dir() and any(d.iterdir())
    )


def run_dcm2niix(dicom_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "dcm2niix",
        "-b", "y",   # emit BIDS JSON sidecar
        "-ba", "y",  # anonymize BIDS text fields
        "-z", "y",   # gzip output
        "-f", "%p_%s",  # filename: ProtocolName_SeriesNumber
        "-o", str(out_dir),
        str(dicom_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [WARN] dcm2niix failed for {dicom_dir}:\n{result.stderr}", file=sys.stderr)
    else:
        print(f"  converted {dicom_dir.parents[2].name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scans_dir", type=Path, help="<subject>/SCANS directory")
    parser.add_argument("staging_out_dir", type=Path, help="flat output dir for NIfTI+JSON pairs")
    args = parser.parse_args()

    if shutil.which("dcm2niix") is None:
        sys.exit("dcm2niix not found on PATH. Activate the env: conda activate ad-early-detection")

    dicom_dirs = find_dicom_dirs(args.scans_dir)
    if not dicom_dirs:
        sys.exit(f"No non-empty DICOM series found under {args.scans_dir}")

    print(f"Found {len(dicom_dirs)} series under {args.scans_dir}")
    for dicom_dir in dicom_dirs:
        run_dcm2niix(dicom_dir, args.staging_out_dir)


if __name__ == "__main__":
    main()
