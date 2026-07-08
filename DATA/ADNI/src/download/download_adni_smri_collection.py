#!/usr/bin/env python3
"""
download_adni_smri_collection.py
=================================
Bulk-downloads T1w/MPRAGE sMRI images from an existing LONI IDA Data
Collection.

Workflow this completes
------------------------
1. `download_adni_smri.py` resolves the paired sMRI image ID for every
   downloaded fMRI scan and writes the comma-separated list to
   `__metadata__/smri_image_ids.txt`.
2. Paste that list into LONI's Advanced Search "Image ID" field, select all
   results, and add them to a new Data Collection (by hand, in the browser).
3. Run this script, pointing it at that collection's name — it drives the
   same "Not Downloaded" batch-download loop as `download_collection.py`,
   just with sMRI-flavoured defaults (collection name, output directory).

This is a thin entry point over `download_collection.py`'s generic loop —
no new browser-automation logic lives here.

Usage
-----
    python download_adni_smri_collection.py --collection smri-all-v1
    python download_adni_smri_collection.py --collection smri-all-v1 --batch-size 5
    python download_adni_smri_collection.py --collection smri-all-v1 --headless false

Output
------
    DATA/ADNI/__smri_dicom_zips_flat__/{subject_id}_{image_id}.zip
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
from pathlib import Path

from download_collection import DATA_DIR, DEFAULT_ENV_FILE, PROJECT_ROOT, run

DEFAULT_COLLECTION = "smri-all-v1"
DEFAULT_OUTPUT_DIR = DATA_DIR / "__smri_dicom_zips_flat__"
DEFAULT_BATCH_SIZE = 5


def _default_log_file() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "logs" / "adni_download" / f"download_adni_smri_collection_{ts}.log"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--collection", default=DEFAULT_COLLECTION)
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    p.add_argument("--log-file", default=str(_default_log_file()))
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--headless", default="true", help="Run browser headless (default: true)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
