#!/usr/bin/env python3
"""Stage 1.x: write dataset_description.json (+ participants.tsv, README) at the BIDS root.

The original BIDS_og.py never wrote this file. MRIQC and fMRIPrep both refuse to run without
it, and the BIDS validator flags its absence as an error — so this fixes a hard correctness gap,
not just a nicety.

Usage:
    python make_dataset_description.py <bids_root> --name "Glioma Resting-State" [--force]
"""
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bids_root", type=Path)
    parser.add_argument("--name", default="Glioma Resting-State fMRI")
    parser.add_argument("--force", action="store_true", help="overwrite if already present")
    args = parser.parse_args()

    args.bids_root.mkdir(parents=True, exist_ok=True)

    desc_path = args.bids_root / "dataset_description.json"
    if desc_path.exists() and not args.force:
        print(f"{desc_path} already exists, leaving as-is (use --force to overwrite)")
    else:
        desc_path.write_text(json.dumps({
            "Name": args.name,
            "BIDSVersion": "1.8.0",
            "DatasetType": "raw",
            "Authors": ["di54lup"],
        }, indent=4))
        print(f"Wrote {desc_path}")

    participants_path = args.bids_root / "participants.tsv"
    if not participants_path.exists():
        # Populated per-subject by build_bids.py appending a row; header only here.
        participants_path.write_text("participant_id\n")
        print(f"Wrote {participants_path}")

    readme_path = args.bids_root / "README"
    if not readme_path.exists():
        readme_path.write_text(
            f"{args.name}\n\nBIDS dataset assembled by scripts/01_dicom_to_bids/. "
            "See ../../docs/PIPELINE_OVERVIEW.md for the full pipeline.\n"
        )
        print(f"Wrote {readme_path}")


if __name__ == "__main__":
    main()
