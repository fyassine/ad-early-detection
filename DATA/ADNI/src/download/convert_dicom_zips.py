#!/usr/bin/env python3
"""
convert_dicom_zips.py
======================
Convert {subject_id}_{image_id}.zip files (raw DICOM ZIPs saved by
download_collection.py) to {subject_id}_{image_id}.nii.gz via dcm2niix.

Runs as a separate process from download_collection.py so dcm2niix
(CPU-bound, ~1-2 min/image) never blocks the browser-driven downloads. The
two scripts communicate only through the filesystem: this script polls
--zip-dir for *.zip files and writes finished NIfTIs to --output-dir,
skipping any zip whose .nii.gz already exists there.

Usage
-----
    python convert_dicom_zips.py                 # one pass over existing zips
    python convert_dicom_zips.py --watch         # keep polling for new zips
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

import download_collection as dc  # reuse logging setup + Colors

SRC_DIR    = Path(__file__).resolve().parent   # .../src/download/
ADNI_SRC_DIR = SRC_DIR.parent                 # .../src/
DATA_DIR   = ADNI_SRC_DIR.parent              # .../ADNI/
PROJECT_ROOT = DATA_DIR.parent.parent         # ad-early-detection/

DEFAULT_ZIP_DIR    = DATA_DIR / "__dicom_zips_flat__"
DEFAULT_OUTPUT_DIR = DATA_DIR / "__fmri_wholebrain_sch200_flat__"


def _default_log_file() -> Path:
    """Return logs/adni-download/<YYYYMMDD_HHMMSS>/adni_dicom_to_nifti.log."""
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "logs" / "adni-download" / ts / "adni_dicom_to_nifti.log"


DEFAULT_LOG_FILE   = _default_log_file()

ZIP_NAME_RE = re.compile(r"^(\d{3}_S_\d{4})_(\d+)\.zip$")
DCM2NIIX_TIMEOUT_S = 180


def convert_zip(zip_path: Path, output_dir: Path) -> list[Path]:
    """
    Extract DICOMs from zip_path (named {subject_id}_{image_id}.zip),
    convert with dcm2niix, and save the result(s) as
    output_dir/{subject_id}_{image_id}.nii.gz — additional series in the
    same zip (rare) get a _2, _3, ... suffix. Returns the written paths.
    """
    m = ZIP_NAME_RE.match(zip_path.name)
    if not m:
        raise ValueError(f"Unexpected zip filename {zip_path.name!r}; expected {{subject_id}}_{{image_id}}.zip")
    subject_id, image_id = m.group(1), m.group(2)

    tmp_root = zip_path.parent / f"_convert_{zip_path.stem}"
    tmp_root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
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
                check=True, capture_output=True, timeout=DCM2NIIX_TIMEOUT_S,
            )

        niis = sorted(nii_tmp.glob("*.nii.gz"))
        if not niis:
            raise ValueError(f"dcm2niix produced no output for {zip_path.name}")

        for i, nii in enumerate(niis):
            suffix = "" if i == 0 else f"_{i + 1}"
            dest = output_dir / f"{subject_id}_{image_id}{suffix}.nii.gz"
            shutil.move(str(nii), str(dest))
            written.append(dest)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    return written


def run_pass(zip_dir: Path, output_dir: Path) -> tuple[int, int]:
    """Convert every zip in zip_dir without a corresponding .nii.gz yet. Returns (converted, skipped)."""
    converted = 0
    skipped = 0
    for zip_path in sorted(zip_dir.glob("*.zip")):
        m = ZIP_NAME_RE.match(zip_path.name)
        if not m:
            dc.log(f"  ! Skipping unexpected filename {zip_path.name}", dc.Colors.YELLOW)
            continue
        subject_id, image_id = m.group(1), m.group(2)
        dest = output_dir / f"{subject_id}_{image_id}.nii.gz"
        if dest.exists():
            skipped += 1
            continue

        dc.log(f"  ↳ Converting {zip_path.name}...")
        try:
            written = convert_zip(zip_path, output_dir)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as e:
            dc.log(f"  ✗ Conversion failed for {zip_path.name}: {e}", dc.Colors.RED)
            continue
        for p in written:
            dc.log(f"  ✓ {p}", dc.Colors.GREEN)
        converted += 1
    return converted, skipped


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zip-dir",       default=str(DEFAULT_ZIP_DIR))
    p.add_argument("--output-dir",    default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--log-file",      default=str(DEFAULT_LOG_FILE))
    p.add_argument("--watch",         action="store_true", help="Keep polling for new zips")
    p.add_argument("--poll-interval", type=int, default=30)
    args = p.parse_args()

    dc._logger = dc.setup_logging(Path(args.log_file))
    zip_dir = Path(args.zip_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dc.log(f"\n{'═'*60}")
    dc.log(f"  DICOM ZIP -> NIfTI Converter")
    dc.log(f"{'═'*60}")
    dc.log(f"  Zip dir    : {zip_dir}")
    dc.log(f"  Output dir : {output_dir}")
    dc.log(f"  Watch      : {args.watch} (poll every {args.poll_interval}s)")
    dc.log(f"{'═'*60}\n")

    while True:
        converted, skipped = run_pass(zip_dir, output_dir)
        dc.log(f"  Pass done: {converted} converted, {skipped} already done", dc.Colors.CYAN)
        if not args.watch:
            break
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
