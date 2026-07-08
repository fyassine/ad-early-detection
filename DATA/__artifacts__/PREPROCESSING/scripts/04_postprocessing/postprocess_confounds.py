#!/usr/bin/env python3
"""Stage 4: confound regression + bandpass filtering.

NEW script — generated, not copied (the original rad_postprocessing2.slurm lives on the
unreachable CORE cluster). Reproduces the institutional `desc-ICAAROMA2Phys1GS` methodology
inferred from the original final_reorient.py's expected input filename: explicit ICA-AROMA
noise-component regression + physiological (WM/CSF) regressors + 1x global signal regression,
combined with bandpass filtering in a single nilearn.signal.clean() call (regressing and
filtering sequentially can reintroduce variance at the filter edges that confound regression
just removed).

Inputs (fMRIPrep 20.2.7 --use-aroma naming, see ../03_fmriprep/README.md for why this version):
  - <prefix>_desc-preproc_bold.nii.gz       standard preprocessed BOLD (NOT the AROMA-precleaned
                                              desc-smoothAROMAnonaggr_bold — that file already has
                                              AROMA noise removed, which would double-count if we
                                              also regressed the noise ICs out of it again)
  - <prefix>_desc-brain_mask.nii.gz
  - <prefix>_desc-confounds_timeseries.tsv  white_matter, csf, global_signal,
                                              non_steady_state_outlier_* columns
  - <task_prefix>_AROMAnoiseICs.csv         1-based noise component indices
  - <task_prefix>_desc-MELODIC_mixing.tsv   timepoints x components mixing matrix, no header

fMRIPrep never deletes dummy-scan volumes for --dummy-scans N (gap #6 from docs/OPEN_QUESTIONS.md)
— it only flags them via non_steady_state_outlier_* columns, which this script always includes
as nuisance regressors rather than assuming volumes were already removed.

Output naming preserves the institutional convention exactly, since ../05_reorientation/
final_reorient_nibabel.py expects it:
  sub-<ID>_ses-<N>_task-rest_run-<N>_space-<space>_desc-ICAAROMA2Phys1GS_bold.nii.gz

Usage:
    python postprocess_confounds.py \\
        --preproc-bold sub-01_ses-1_task-rest_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz \\
        --brain-mask   sub-01_ses-1_task-rest_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz \\
        --confounds-tsv sub-01_ses-1_task-rest_desc-confounds_timeseries.tsv \\
        --aroma-noise-ics sub-01_ses-1_task-rest_AROMAnoiseICs.csv \\
        --melodic-mixing sub-01_ses-1_task-rest_desc-MELODIC_mixing.tsv \\
        --output sub-01_ses-1_task-rest_run-1_space-MNI152NLin2009cAsym_res-2_desc-ICAAROMA2Phys1GS_bold.nii.gz \\
        --low-pass 0.1 --high-pass 0.01
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker
from nilearn.signal import clean


def load_aroma_noise_regressors(noise_ics_path: Path, mixing_path: Path) -> np.ndarray:
    noise_idx_1based = [
        int(x) for x in noise_ics_path.read_text().strip().split(",") if x.strip()
    ]
    mixing = pd.read_csv(mixing_path, sep="\t", header=None).values
    noise_idx_0based = [i - 1 for i in noise_idx_1based]
    return mixing[:, noise_idx_0based]


def build_confound_matrix(confounds_tsv: Path, aroma_noise: np.ndarray) -> pd.DataFrame:
    confounds_df = pd.read_csv(confounds_tsv, sep="\t")
    phys_gsr_cols = [c for c in ("white_matter", "csf", "global_signal") if c in confounds_df.columns]
    dummy_cols = [c for c in confounds_df.columns if c.startswith("non_steady_state_outlier_")]

    matrix = pd.DataFrame(
        aroma_noise,
        columns=[f"aroma_noise_ic_{i}" for i in range(aroma_noise.shape[1])],
    )
    for col in phys_gsr_cols:
        matrix[col] = confounds_df[col].to_numpy()
    for col in dummy_cols:
        matrix[col] = confounds_df[col].to_numpy()

    matrix = matrix.fillna(0.0)  # confound TSVs lead derivative/outlier columns with NaN
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preproc-bold", type=Path, required=True)
    parser.add_argument("--brain-mask", type=Path, required=True)
    parser.add_argument("--confounds-tsv", type=Path, required=True)
    parser.add_argument("--aroma-noise-ics", type=Path, required=True)
    parser.add_argument("--melodic-mixing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--t-r", type=float, default=None,
                         help="repetition time in seconds; read from the BOLD JSON sidecar if omitted")
    parser.add_argument("--low-pass", type=float, default=0.1, help="Hz")
    parser.add_argument("--high-pass", type=float, default=0.01, help="Hz")
    args = parser.parse_args()

    t_r = args.t_r
    if t_r is None:
        import json
        sidecar = args.preproc_bold.with_suffix("").with_suffix(".json")
        if not sidecar.exists():
            raise SystemExit(f"--t-r not given and no sidecar at {sidecar}")
        t_r = json.loads(sidecar.read_text())["RepetitionTime"]

    aroma_noise = load_aroma_noise_regressors(args.aroma_noise_ics, args.melodic_mixing)
    confound_matrix = build_confound_matrix(args.confounds_tsv, aroma_noise)

    masker = NiftiMasker(mask_img=str(args.brain_mask), standardize=False)
    signals = masker.fit_transform(str(args.preproc_bold))

    cleaned = clean(
        signals,
        confounds=confound_matrix.values,
        detrend=True,
        standardize=False,
        low_pass=args.low_pass,
        high_pass=args.high_pass,
        t_r=t_r,
    )

    cleaned_img = masker.inverse_transform(cleaned)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cleaned_img.to_filename(str(args.output))
    print(f"Wrote {args.output} (TR={t_r}s, band={args.high_pass}-{args.low_pass}Hz, "
          f"{confound_matrix.shape[1]} confound regressors: "
          f"{aroma_noise.shape[1]} AROMA noise ICs + {confound_matrix.shape[1] - aroma_noise.shape[1]} other)")


if __name__ == "__main__":
    main()
