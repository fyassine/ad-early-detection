#!/usr/bin/env python3
"""Stage 1.4: remove empty anat/func/dwi/fmap subfolders from a BIDS tree.

Preserves the original pipeline's step 1.3 (find/exec deleting subjects with empty `anat/`
dirs) as a standalone, general utility: glioma resection-cavity subjects sometimes lack a
given sequence and leave an empty modality folder behind, which trips up the BIDS validator
and fMRIPrep. Generalized beyond just `anat/` to any BIDS modality folder.

Usage:
    python cleanup_empty_anat.py <bids_root> [--dry-run]
"""
import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bids_root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    modality_dirs = [
        d for d in args.bids_root.glob("sub-*/ses-*/*")
        if d.is_dir() and d.name in ("anat", "func", "fmap", "dwi")
    ]

    removed = 0
    for d in modality_dirs:
        if not any(d.iterdir()):
            print(f"{'[DRY-RUN] would delete' if args.dry_run else 'Deleting'} empty: {d}")
            if not args.dry_run:
                d.rmdir()
            removed += 1

    print(f"{removed} empty modality directories {'found' if args.dry_run else 'removed'}.")


if __name__ == "__main__":
    main()
