# Pipeline overview

```
DICOM (SCANS/*/resources/DICOM/files)
  │  run_dcm2niix.py            [dcm2niix -b y -ba y -z y]
  ▼
staging/<subject>/  (flat NIfTI + JSON sidecar pairs)
  │  merge_runs.py              [only if >1 resting-state run; SAMPLE subject has 1, skipped]
  │  build_bids.py              [series_classification.py routes by SeriesDescription/ImageType]
  │  make_dataset_description.py
  │  cleanup_empty_anat.py
  │  run_bids_validator.py      [pybids + optional official bids-validator container] ── HARD GATE
  ▼
BIDS/sub-<ID>/ses-<N>/{anat,func,fmap,dwi}
  │  scripts/02_mriqc/submit_mriqc.slurm           [sbatch, cm4_inter, ~10-20min]
  ▼
mriqc_out/   ── manual QC checkpoint #1 (inspect HTML reports, raw-data artifacts)
  │  scripts/03_fmriprep/submit_fmriprep.slurm     [sbatch, cm4_inter, ~few hours,
  │                                                 --use-aroma, --dummy-scans N, --fs-no-reconall]
  ▼
fmriprep_out/   ── manual QC checkpoint #2 (inspect fMRIPrep's own HTML reports)
  │  scripts/04_postprocessing/postprocess_confounds.py + qc_motion_table.py
  │     [AROMA noise-IC + WM/CSF + GSR + bandpass via nilearn.signal.clean, single call]
  ▼
*_desc-ICAAROMA2Phys1GS_bold.nii.gz
  │  scripts/05_reorientation/final_reorient_nibabel.py
  ▼
*_desc-ICAAROMA2Phys1GS_bold_reoriented.nii.gz
  │  scripts/06_parcellation/  [out of scope, handled separately]
  ▼
(parcellated connectivity data)
```

## Why `desc-ICAAROMA2Phys1GS`

This naming is inherited from the original institutional pipeline's `final_reorient.py`, which
expected exactly this filename. Decoded:
- `ICAAROMA` — ICA-AROMA noise components explicitly regressed out (not fMRIPrep's own
  pre-cleaned `desc-smoothAROMAnonaggr_bold` — see `scripts/04_postprocessing/postprocess_confounds.py`
  docstring for why those are different and not interchangeable inputs).
- `Phys` — physiological regressors: WM + CSF mean signal.
- `1GS` — one pass of global signal regression.

All combined with bandpass filtering (default 0.01–0.1 Hz) in one `nilearn.signal.clean()` call.

## Two manual QC checkpoints (per the online research, now concretely wired in)

1. **Before fMRIPrep**: inspect MRIQC's per-subject HTML reports for raw-data artifacts
   (motion, ghosting, signal dropout) before committing to a multi-hour fMRIPrep run.
2. **After fMRIPrep**: inspect fMRIPrep's own HTML reports for registration/normalization
   failures. `scripts/04_postprocessing/qc_motion_table.py` additionally turns the mean-FD
   exclusion criterion (>0.5mm) and short-scan flag (<5min usable) into a concrete
   `qc_summary.tsv` table rather than leaving it as a manual guideline.

## Gaps fixed relative to the user's online-research pipeline sketch

| Gap | Fix |
|---|---|
| No BIDS-validator step | `scripts/01_dicom_to_bids/run_bids_validator.py`, hard-gates later stages |
| No `dataset_description.json` | `scripts/01_dicom_to_bids/make_dataset_description.py` |
| Multi-run merge assumed away | `scripts/01_dicom_to_bids/merge_runs.py`, conditional, skips if 1 run |
| Hardcoded TR in original BIDS_og.py | `build_bids.py` reads dcm2niix's own per-subject sidecar |
| Mean-FD guideline not operationalized | `scripts/04_postprocessing/qc_motion_table.py` |
| `--dummy-scans` assumed to remove volumes | `postprocess_confounds.py` always includes `non_steady_state_outlier_*` as regressors |
| Generic 36-parameter confound model assumed | Matched to the real institutional AROMA+Phys+GSR strategy instead (see above) |

## Logs

All stages write into a `logs/` tree organized by **stage → subject → run-timestamp**:

```
logs/01_dicom_to_bids/sub-<label>/<TS>/            stage1.log + per-substep .log + summary.txt
logs/02_mriqc/sub-<label>/<TS>_job<jobid>/         mriqc.{out,err} + summary.txt
logs/03_fmriprep/sub-<label>/<TS>_job<jobid>/      fmriprep.{out,err} + summary.txt
logs/04_postprocessing/sub-<label>/<TS>_job<jobid>/ postprocess.{out,err} + summary.txt
logs/<stage>/_slurm/%x_%j.{out,err}                raw Slurm bookkeeping (unfiltered safety net)
logs/_legacy/                                       pre-restructure root-dir logs
```

- The Slurm stages (2–4) keep this self-contained — submission commands are unchanged
  (`sbatch scripts/.../submit_*.slurm ...`); the job computes its own run dir and redirects
  there. Shared helpers live in `scripts/lib/logging.sh`.
- `.err` files are filtered free of tqdm progress-bar spam (`tr '\r' '\n' | grep -v '<bar>'`);
  the unfiltered stream is still preserved under `logs/<stage>/_slurm/` so nothing is lost.
- fMRIPrep/MRIQC use a persistent TemplateFlow cache (`.cache/templateflow/`) so MNI templates
  download once instead of re-downloading (and re-logging) every run.
