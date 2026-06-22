"""
run_all_processing.py — Master script to generate all network-subset FC matrices.

Runs the Schaefer-only subset experiments (no fMRI reprocessing needed) and
optionally processes all follow-up fMRI visits for full longitudinal coverage.

Usage (from repo root):
    # Schaefer-only subsets from existing __fc_wholebrain_sch200_flat__ matrices:
    python -m DATA.src.processing.run_all_processing

    # Also reprocess all follow-up visits from __fmri_wholebrain_sch200_flat__/fmri into __fc_wholebrain_sch200_flat__:
    python -m DATA.src.processing.run_all_processing --reprocess-followups

    # Full run including Tian-atlas experiments:
    python -m DATA.src.processing.run_all_processing \\
        --reprocess-followups \\
        --tian-atlas /mnt/e/fyassine/ad-early-detection/DATA/src/processing/atlas/Tian_Subcortex_S2_3T.nii.gz \\
        --tian-labels /mnt/e/fyassine/ad-early-detection/DATA/src/processing/atlas/Tian_Subcortex_S2_3T_label.txt

Experiment table:
    __fc_wholebrain_sch200_flat__          Whole brain               (Schaefer 200, all visits after reprocessing)
    __fc_hippo_tian2_flat__                Hippocampus only          (Tian Scale II, 4 ROIs)
    __fc_limbic_sch200_flat__              Limbic only               (Schaefer subset, 12 ROIs)
    __fc_dan_sch200_flat__                 Dorsal Attention only     (Schaefer subset, 26 ROIs)
    __fc_dmn-hippo_sch200-tian2_flat__     DMN + Hippocampus         (combined masker, 50 ROIs)
    __fc_dmn-limbic_sch200_flat__          DMN + Limbic              (Schaefer subset, 58 ROIs)
    __fc_dmn-hippo-limbic_sch200-tian2_flat__  DMN + Hippocampus + Limbic (combined masker, 62 ROIs)
    __fc_dmn-hippo-limbic-dan_sch200-tian2_flat__  All combined      (combined masker, 88 ROIs)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# Schaefer-only experiments: no fMRI data needed, just slice __fc_wholebrain_sch200_flat__ matrices.
SCHAEFER_JOBS = [
    {
        "version": "__fc_limbic_sch200_flat__",
        "networks": ["Limbic"],
        "suffix": "limbic",
        "description": "Limbic only (12 ROIs)",
    },
    {
        "version": "__fc_dan_sch200_flat__",
        "networks": ["DorsAttn"],
        "suffix": "dorsal_attention",
        "description": "Dorsal Attention Network only (26 ROIs)",
    },
    {
        "version": "__fc_dmn-limbic_sch200_flat__",
        "networks": ["Default", "Limbic"],
        "suffix": "dmn_limbic",
        "description": "DMN + Limbic (58 ROIs)",
    },
]

# Tian-atlas experiments: require fMRI data + Tian NIfTI.
TIAN_JOBS = [
    {
        "version": "__fc_hippo_tian2_flat__",
        "description": "Hippocampus only (Tian Scale II, 4 ROIs)",
        "script": "process_using_tian_atlas",
        "extra_args": [],
    },
    {
        "version": "__fc_dmn-hippo_sch200-tian2_flat__",
        "description": "DMN + Hippocampus (50 ROIs)",
        "script": "process_combined_schaefer_tian",
        "extra_args": ["--networks", "Default", "--output-suffix", "dmn_hippo"],
    },
    {
        "version": "__fc_dmn-hippo-limbic_sch200-tian2_flat__",
        "description": "DMN + Limbic + Hippocampus (62 ROIs)",
        "script": "process_combined_schaefer_tian",
        "extra_args": ["--networks", "Default", "Limbic", "--output-suffix", "dmn_limbic_hippo"],
    },
    {
        "version": "__fc_dmn-hippo-limbic-dan_sch200-tian2_flat__",
        "description": "All combined — DMN + Limbic + DAN + Hippocampus (88 ROIs)",
        "script": "process_combined_schaefer_tian",
        "extra_args": ["--networks", "Default", "Limbic", "DorsAttn", "--output-suffix", "all_combined"],
    },
]


def run_schaefer_job(job: dict) -> bool:
    version = job["version"]
    networks = job["networks"]
    suffix = job["suffix"]
    description = job["description"]

    print(f"\n{'='*60}")
    print(f"  {version} — {description}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, "-m",
        "DATA.src.processing.subset_schaefer_networks",
        "--networks", *networks,
        "--output-version", version,
        "--output-suffix", suffix,
    ]
    print(f"  Command: {' '.join(cmd[2:])}")

    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"  ERROR: {version} failed with return code {result.returncode}")
        return False
    return True


def print_tian_commands(tian_atlas: Path | None, tian_labels: Path | None, n_jobs: int = 16) -> None:
    print("\n" + "=" * 60)
    print("  TIAN-ATLAS EXPERIMENTS")
    print("=" * 60)

    if tian_atlas is None:
        print(
            "\n  Tian atlas not provided — skipping __fc_hippo_tian2_flat__, __fc_dmn-hippo_sch200-tian2_flat__, __fc_dmn-hippo-limbic_sch200-tian2_flat__, __fc_dmn-hippo-limbic-dan_sch200-tian2_flat__.\n"
            "  Download Tian Scale II atlas and re-run with:\n"
            "    --tian-atlas /mnt/e/fyassine/ad-early-detection/DATA/src/processing/atlas/Tian_Subcortex_S2_3T.nii.gz\n"
            "    --tian-labels /mnt/e/fyassine/ad-early-detection/DATA/src/processing/atlas/Tian_Subcortex_S2_3T_label.txt\n"
            "\n  Atlas download: https://github.com/yetianmed/subcortex\n"
            "  Or via templateflow: tpl-MNI152NLin6Asym_atlas-Tian_res-1_dseg.nii.gz"
        )
        print("\n  Ready-to-run commands once atlas is available:")
        for job in TIAN_JOBS:
            script = job["script"]
            version = job["version"]
            extra = " ".join(job["extra_args"])
            print(
                f"\n  # {version} — {job['description']}\n"
                f"  python -m DATA.src.processing.{script} \\\n"
                f"      --atlas-path /mnt/e/fyassine/ad-early-detection/DATA/src/processing/atlas/Tian_Subcortex_S2_3T.nii.gz \\\n"
                f"      --labels-path /mnt/e/fyassine/ad-early-detection/DATA/src/processing/atlas/Tian_Subcortex_S2_3T_label.txt"
                + (f" \\\n      {extra}" if extra else "")
                + (f" \\\n      --output-version {version}" if "output-version" not in extra else "")
            )
        return

    for job in TIAN_JOBS:
        version = job["version"]
        script = job["script"]
        description = job["description"]
        extra_args = job["extra_args"]

        print(f"\n{'='*60}")
        print(f"  {version} — {description}")
        print(f"{'='*60}")

        if script == "process_using_tian_atlas":
            # This script uses --atlas-path / --labels-path / --output-root
            output_root = REPO_ROOT / "DATA" / "DELCODE" / version
            cmd = [
                sys.executable, "-m",
                f"DATA.src.processing.{script}",
                "--atlas-path", str(tian_atlas),
                "--output-root", str(output_root),
                "--n-jobs", str(n_jobs),
                *extra_args,
            ]
            if tian_labels is not None:
                cmd += ["--labels-path", str(tian_labels)]
        else:
            # process_combined_schaefer_tian uses --tian-atlas / --tian-labels / --output-version
            cmd = [
                sys.executable, "-m",
                f"DATA.src.processing.{script}",
                "--tian-atlas", str(tian_atlas),
                "--output-version", version,
                "--n-jobs", str(n_jobs),
                *extra_args,
            ]
            if tian_labels is not None:
                cmd += ["--tian-labels", str(tian_labels)]

        print(f"  Command: {' '.join(cmd[2:])}")
        result = subprocess.run(cmd, cwd=str(REPO_ROOT))
        if result.returncode != 0:
            print(f"  ERROR: {version} failed with return code {result.returncode}")


def run_followup_reprocessing(fmri_root: Path) -> bool:
    """Process all follow-up visits from fmri_root into __fc_wholebrain_sch200_flat__/matrices/."""
    print(f"\n{'='*60}")
    print(f"  __fc_wholebrain_sch200_flat__ — Reprocess all visits from {fmri_root.name}")
    print(f"{'='*60}")
    output_dir = REPO_ROOT / "DATA" / "DELCODE" / "__fc_wholebrain_sch200_flat__" / "matrices"
    cmd = [
        sys.executable, "-m",
        "DATA.src.processing.process_using_schaeffer_atlas",
        "--fmri-root", str(fmri_root),
        "--output-dir", str(output_dir),
    ]
    print(f"  Command: {' '.join(cmd[2:])}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"  ERROR: __fc_wholebrain_sch200_flat__ reprocessing failed (code {result.returncode})")
        return False
    return True


def main(
    tian_atlas: Path | None,
    tian_labels: Path | None,
    skip_schaefer: bool,
    reprocess_followups: bool,
    n_jobs: int = 16,
) -> None:
    print("=" * 60)
    print("  NETWORK SUBSET PROCESSING — MASTER RUNNER")
    print("=" * 60)
    print(f"  Repo root: {REPO_ROOT}")
    print(f"  Reprocess follow-ups: {reprocess_followups}")
    print(f"  Schaefer jobs: {len(SCHAEFER_JOBS)}")
    print(f"  Tian jobs: {len(TIAN_JOBS)} ({'atlas provided' if tian_atlas else 'atlas missing — will print commands only'})")
    print(f"  Tian parallel workers: {n_jobs}")

    # Step 0: Reprocess all follow-up visits from __fmri_wholebrain_sch200_flat__/fmri into __fc_wholebrain_sch200_flat__/matrices
    if reprocess_followups:
        fmri_root = REPO_ROOT / "DATA" / "DELCODE" / "__fmri_wholebrain_sch200_flat__" / "fmri"
        if not fmri_root.exists():
            print(f"\nWARNING: __fmri_wholebrain_sch200_flat__/fmri not found at {fmri_root}, skipping reprocessing.")
        else:
            print("\n\n── STEP 0: REPROCESS ALL VISITS ────────────────────────────")
            ok = run_followup_reprocessing(fmri_root)
            if ok:
                print("  __fc_wholebrain_sch200_flat__ matrices updated with all follow-up visits.")
            # Re-run Schaefer subsets after __fc_wholebrain_sch200_flat__ is updated so follow-ups are sliced too
            skip_schaefer = False

    # Step 1: Schaefer-only subsets
    if not skip_schaefer:
        print("\n\n── SCHAEFER SUBSET EXPERIMENTS ──────────────────────────────")
        failed = []
        for job in SCHAEFER_JOBS:
            ok = run_schaefer_job(job)
            if not ok:
                failed.append(job["version"])
        if failed:
            print(f"\nWARNING: {len(failed)} Schaefer job(s) failed: {failed}")
        else:
            print(f"\nAll {len(SCHAEFER_JOBS)} Schaefer subset experiments completed.")
    else:
        print("\n  Schaefer-only jobs skipped (--skip-schaefer).")

    # Step 2: Tian-atlas experiments
    print("\n── TIAN ATLAS EXPERIMENTS ───────────────────────────────────")
    print_tian_commands(tian_atlas, tian_labels, n_jobs=n_jobs)

    print("\n\n── NEXT STEPS ───────────────────────────────────────────────")
    print(
        "  For each data version, run the experiment notebooks in order:\n"
        "  1. CLASSIFIER/notebooks/NETWORK_GAAE_RUNNER.ipynb  (GAAE pretraining)\n"
        "  2. CLASSIFIER/notebooks/NETWORK_GEC_RUNNER.ipynb   (GEC classification)\n"
        "  3. Precompute longitudinal embeddings:\n"
        "     python -m PROGNOSER.src.build_subject_embeddings --all --strategy all_aggs\n"
        "\n  See DOCS/prognosis_pipeline.md for full instructions."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tian-atlas", type=Path, default=None,
                        help="Path to Tian Subcortex Scale II NIfTI file")
    parser.add_argument("--tian-labels", type=Path, default=None,
                        help="Path to Tian label text file (one label per line)")
    parser.add_argument("--skip-schaefer", action="store_true",
                        help="Skip Schaefer-only subset jobs")
    parser.add_argument("--reprocess-followups", action="store_true",
                        help="Reprocess all follow-up visits from __fmri_wholebrain_sch200_flat__/fmri into __fc_wholebrain_sch200_flat__/matrices")
    parser.add_argument("--n-jobs", type=int, default=16,
                        help="Parallel workers for Tian atlas jobs (default: 16). Lower if you hit OOM.")
    args = parser.parse_args()
    main(
        tian_atlas=args.tian_atlas,
        tian_labels=args.tian_labels,
        skip_schaefer=args.skip_schaefer,
        reprocess_followups=args.reprocess_followups,
        n_jobs=args.n_jobs,
    )
