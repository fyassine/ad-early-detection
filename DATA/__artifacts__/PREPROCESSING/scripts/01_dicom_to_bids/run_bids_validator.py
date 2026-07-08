#!/usr/bin/env python3
"""Stage 1.5: validate the assembled BIDS tree.

Restores the explicit BIDS-validation step from the original docs (step 2) that was dropped
from the user's online-research pipeline sketch. Two checks are run:

  1. pybids.BIDSLayout(validate=True) — fast, no container needed, already installed
     (env/environment.yml). Good first-pass check, but less complete than the official validator.
  2. The official bids-validator JS tool via apptainer, if a pulled .sif image is available at
     containers/bids-validator.sif (pull with: apptainer pull containers/bids-validator.sif
     docker://bids/validator:latest) — more thorough, optional.

Exits non-zero if either check reports errors, so this can gate MRIQC/fMRIPrep submission in
tests/test_pipeline_sample.sh.

Usage:
    python run_bids_validator.py <bids_root>
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run_pybids_check(bids_root: Path) -> bool:
    from bids import BIDSLayout
    try:
        BIDSLayout(str(bids_root), validate=True)
        print("[pybids] BIDSLayout loaded with validate=True — no structural errors found.")
        return True
    except Exception as e:
        print(f"[pybids] validation FAILED: {e}", file=sys.stderr)
        return False


def run_container_check(bids_root: Path, sif_path: Path) -> bool | None:
    if not sif_path.exists():
        print(f"[bids-validator container] {sif_path} not found, skipping. "
              f"Pull it with containers/pull_containers.sh or "
              f"`apptainer pull {sif_path} docker://bids/validator:latest` for a more thorough check.")
        return None
    if shutil.which("apptainer") is None:
        print("[bids-validator container] apptainer not on PATH (module load apptainer/1.3.4), skipping.")
        return None
    result = subprocess.run(
        ["apptainer", "run", "--bind", f"{bids_root}:/data:ro", str(sif_path), "/data"],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bids_root", type=Path)
    parser.add_argument("--container-sif", type=Path,
                         default=Path(__file__).resolve().parents[2] / "containers" / "bids-validator.sif")
    args = parser.parse_args()

    pybids_ok = run_pybids_check(args.bids_root)
    container_ok = run_container_check(args.bids_root, args.container_sif)

    if not pybids_ok or container_ok is False:
        sys.exit(1)
    print("BIDS validation passed.")


if __name__ == "__main__":
    main()
