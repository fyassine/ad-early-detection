# fMRIPrep stage

## Prerequisites
1. `containers/pull_containers.sh` has been run (pulls `fmriprep-20.2.7.sif`).
2. `env/freesurfer_license/license.txt` exists — see `env/freesurfer_license/README.md`.
   `submit_fmriprep.slurm` exits early with a clear error if either is missing.

## Why fMRIPrep 20.2.7 (LTS), not a current release
fMRIPrep deprecated `--use-aroma` starting at 23.1 and removed it shortly after — AROMA was
spun out into a separate `fmripost-aroma` BIDS-app. The institutional confound strategy
(`../04_postprocessing/postprocess_confounds.py`) needs fMRIPrep to produce
`*_AROMAnoiseICs.csv` and the MELODIC mixing matrix directly (matching the original
`final_reorient.py`'s expected `desc-ICAAROMA2Phys1GS` naming), so this pins the 20.2.7 LTS
line, the last widely-used release with `--use-aroma` built in. If you need a newer fMRIPrep
release's other improvements, the alternative is: run a current fMRIPrep without `--use-aroma`,
then run `fmripost-aroma` as a separate step on its output — not implemented here, flagged in
`../../docs/OPEN_QUESTIONS.md` as a future migration if 20.2.7 becomes impractical to keep using.

## Why `--fs-no-reconall` (surface reconstruction skipped)
The original institutional pipeline's outputs are all volumetric MNI-space
(`desc-ICAAROMA2Phys1GS` in `MNI152NLin2009cAsym:res-2`) — there are no `fsaverage`/surface
outputs anywhere in it, and your source documentation does not specify recon-all either. So
surfaces aren't needed for this volumetric/atlas-based workflow, and skipping recon-all keeps a
single subject to ~a few hours, comfortably under the `cm4_inter` 8h cap. If a later analysis
needs surface-based parcellation, drop `--fs-no-reconall` from `submit_fmriprep.slurm` and move
the job back to `teramem_inter` (10-day limit), since full recon-all can exceed 8h per subject.

## `--dummy-scans`
fMRIPrep does **not** delete volumes for `--dummy-scans N` — it only annotates the first N
volumes as `non_steady_state_outlier_00`..`non_steady_state_outlier_0N` columns in
`*_desc-confounds_timeseries.tsv`. `postprocess_confounds.py` includes these columns as nuisance
regressors unconditionally. The correct N for this acquisition protocol still needs to be
confirmed (see `../../docs/OPEN_QUESTIONS.md`) — pass `0` if the scanner already discarded
dummy volumes before recording (common on this protocol; verify against the acquisition
parameters/protocol PDF before assuming).

## Runtime/resource notes
- `--mem-mb` and `--nthreads`/`--omp-nthreads` are wired to the Slurm allocation
  (`$SLURM_CPUS_PER_TASK`, and a fixed `60000` MB budget on the `cm4_inter` partition — adjust
  `--mem` in the `#SBATCH` header and this value together).
- For an array over many subjects, change `--participant-label` to use `$SLURM_ARRAY_TASK_ID`
  and add `#SBATCH --array=1-N` once you have a participant list.

## Logs
- Clean, per-run logs: `logs/03_fmriprep/sub-<label>/<timestamp>_job<jobid>/fmriprep.{out,err}`
  plus a `summary.txt` header. The `.err` is filtered free of tqdm progress-bar spam.
- Raw Slurm bookkeeping (unfiltered safety net): `logs/03_fmriprep/_slurm/%x_%j.{out,err}`.
- MNI templates are cached once in `.cache/templateflow/` (bound into the container as
  `/templateflow`), so they are not re-downloaded — and not re-logged — on later runs.
