#!/usr/bin/env python3
"""Synthetic fMRIPrep-output fixture for testing postprocess_confounds.py / qc_motion_table.py
without waiting for a real fMRIPrep run to finish.

Generates, in <out_dir>:
  - <prefix>_desc-preproc_bold.nii.gz   (small random 4D volume)
  - <prefix>_desc-preproc_bold.json     (RepetitionTime sidecar)
  - <prefix>_desc-brain_mask.nii.gz     (all-ones mask matching the volume)
  - <prefix>_desc-confounds_timeseries.tsv
  - <prefix>_AROMAnoiseICs.csv
  - <prefix>_desc-MELODIC_mixing.tsv

Column/value conventions match real fMRIPrep 20.2.7 --use-aroma output exactly enough to
exercise postprocess_confounds.py's parsing logic (NaN in framewise_displacement's first row,
1-based noise component indices, non_steady_state_outlier_* one-hot columns).

Usage:
    python make_synthetic_confounds.py <out_dir> --prefix sub-TEST_ses-1_task-rest \\
        --n-volumes 60 --t-r 2.58 --n-components 12 --n-dummy 2
"""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--prefix", default="sub-TEST_ses-1_task-rest")
    parser.add_argument("--n-volumes", type=int, default=60)
    parser.add_argument("--t-r", type=float, default=2.58)
    parser.add_argument("--n-components", type=int, default=12)
    parser.add_argument("--n-noise-ics", type=int, default=4)
    parser.add_argument("--n-dummy", type=int, default=2)
    parser.add_argument("--shape", type=int, nargs=3, default=(10, 10, 8))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.out_dir / args.prefix

    # --- BOLD + mask ---
    shape4d = (*args.shape, args.n_volumes)
    data = rng.normal(loc=1000, scale=50, size=shape4d).astype(np.float32)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    nib.Nifti1Image(data, affine).to_filename(str(prefix) + "_desc-preproc_bold.nii.gz")
    (Path(str(prefix) + "_desc-preproc_bold.json")).write_text(
        json.dumps({"RepetitionTime": args.t_r}, indent=2)
    )
    mask = np.ones(args.shape, dtype=np.uint8)
    nib.Nifti1Image(mask, affine).to_filename(str(prefix) + "_desc-brain_mask.nii.gz")

    # --- confounds timeseries ---
    n = args.n_volumes
    fd = np.abs(rng.normal(loc=0.1, scale=0.05, size=n))
    fd[0] = np.nan  # fMRIPrep convention: first-row NaN for frame-difference-derived columns
    non_steady = np.zeros((n, args.n_dummy))
    for i in range(args.n_dummy):
        non_steady[i, i] = 1.0
    confounds = pd.DataFrame({
        "white_matter": rng.normal(size=n),
        "csf": rng.normal(size=n),
        "global_signal": rng.normal(size=n),
        "framewise_displacement": fd,
        **{f"non_steady_state_outlier_{i:02d}": non_steady[:, i] for i in range(args.n_dummy)},
    })
    confounds.to_csv(str(prefix) + "_desc-confounds_timeseries.tsv", sep="\t", index=False)

    # --- AROMA noise ICs + MELODIC mixing matrix ---
    noise_ics_1based = rng.choice(
        np.arange(1, args.n_components + 1), size=args.n_noise_ics, replace=False
    )
    (Path(str(prefix) + "_AROMAnoiseICs.csv")).write_text(
        ",".join(str(i) for i in sorted(noise_ics_1based))
    )
    mixing = rng.normal(size=(n, args.n_components))
    pd.DataFrame(mixing).to_csv(str(prefix) + "_desc-MELODIC_mixing.tsv", sep="\t", header=False, index=False)

    print(f"Wrote synthetic fixture with prefix {prefix}")


if __name__ == "__main__":
    main()
