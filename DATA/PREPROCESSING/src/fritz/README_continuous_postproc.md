# Continuous Fritz-side postprocessing

Overlap postprocessing with fMRIPrep instead of waiting for the whole CORE array
to finish. fMRIPrep keeps running on CORE (QOS-throttled to a few concurrent
tasks); meanwhile the subjects that have already **COMPLETED** get pulled to
Fritz and denoised here, using Fritz's otherwise-idle cores.

This is possible because postprocessing is **per-subject independent** — the
container (`postprocessing.sif`) denoises one `--subject_id` reading only that
subject's fMRIPrep output. A subject's postprocessing depends only on its own
fMRIPrep run, never on the rest of the array, so there is no data-dependency
reason to wait. Fritz has apptainer + plenty of cores, so it can run the same
image CORE would.

## What it produces

Matching the DELCODE contract (`DATA/DELCODE/__fmri_wholebrain_sch200_flat__`),
each QC-passing, denoised, reoriented rest-BOLD lands one-file-per-session under:

```
DATA/<COHORT>/__fmri_wholebrain_sch200_flat__/fmri/sub-<ID>/
    sub-<ID>_ses-<S>_task-rest_space-MNI152NLin2009cAsym_res-2_desc-ICAAROMA2Phys1GS_bold_reoriented.nii.gz
```

`<COHORT>` is `OASIS3` or `ADNI`. That directory is the input to the downstream
Schaefer-200 FC extraction (`process_using_schaeffer_atlas.py` →
`__fc_wholebrain_sch200_flat__`), unchanged.

## Pipeline (per subject)

1. **Gate** — only subjects whose fMRIPrep SLURM task is `COMPLETED` (queried
   live via `sacct` on CORE) are eligible. This avoids pulling a subject whose
   fMRIPrep output is still being written (running/timeout/failed tasks are
   ignored).
2. **Pull** — `rsync` COPY of that subject's fMRIPrep dir CORE → Fritz
   (`DATA/<COHORT>/derivatives/fmriprep/sub-<ID>/`). The CORE source is never
   moved or removed.
3. **QC** — `qc_motion_gate.py`: mean framewise displacement per session from
   the fMRIPrep confounds TSV; sessions with **mean FD > 0.5 mm** are excluded
   (convention reused from the reference `qc_motion_table.py`). A per-session
   ledger is written to
   `DATA/<COHORT>/__fmri_wholebrain_sch200_flat__/__artifacts__/qc_motion.csv`.
4. **Denoise** — `apptainer run postprocessing.sif` (strategy
   `ICAAROMA2Phys1GS`, `--dummy 10 --FWHM 6 --LPF 0.1 --HPF 0.01`), mirroring
   `src/core/postprocessing_array_oasis_adni.slurm`. Output lands in
   `DATA/<COHORT>/derivatives/postprocessed/`.
5. **Reorient** — `final_reorient.py`: `*_bold.nii.gz` →
   `*_bold_reoriented.nii.gz` (nibabel `as_closest_canonical` + radiological
   x-flip; no FSL).
6. **Flatten** — QC-passing reoriented BOLD copied into the flat product above.

> The container denoises *all* of a subject's sessions in one run; only the
> QC-passing sessions are reoriented and flattened. A subject with zero passing
> sessions is skipped before the container runs.

## First-time setup (once)

Stage the postprocessing image onto Fritz (it is otherwise only read on CORE):

```bash
bash DATA/PREPROCESSING/src/fritz/postprocess_local.sh --stage-sif --dry-run   # preview
bash DATA/PREPROCESSING/src/fritz/postprocess_local.sh --stage-sif             # ~353 MB copy
```

Lands at `DATA/PREPROCESSING/images/postprocessing.sif` (override with
`POSTPROC_SIMG=`).

## Running

```bash
# See exactly what would be pulled/processed, no transfers or container runs:
bash postprocess_local.sh --dataset adni --dry-run

# Process the first 2 eligible subjects end-to-end (smoke test):
bash postprocess_local.sh --dataset adni --limit 2

# Full continuous pass (default 3 subjects concurrent; container wants 16c/32G each):
bash postprocess_local.sh --dataset adni

# Keep picking up newly-finished subjects every 10 min:
watch -n 600 bash DATA/PREPROCESSING/src/fritz/postprocess_local.sh --dataset adni
```

Resumable one-shot: it processes everything currently eligible then exits.
Already-flattened subjects are skipped (idempotent), so reruns are cheap. Pass
`--overwrite` to redo a subject.

### Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--dataset` | `both` | `oasis3`, `adni`, or `both` |
| `--jobid N` | auto | fMRIPrep array job id for the sacct gate; auto-resolved from the running/most-recent `fmriprep_oasis_adni` job if omitted |
| `--max-parallel N` | `3` | subjects processed concurrently (each container uses ~16 cores) |
| `--fd-threshold MM` | `0.5` | mean-FD exclusion cutoff |
| `--limit N` | `0` (all) | process at most N eligible subjects this pass |
| `--stage-sif` | — | copy `postprocessing.sif` CORE → Fritz, then continue |
| `--overwrite` | — | reprocess subjects already in the flat product |
| `--dry-run` | — | print the plan; no rsync, no container, no writes |
| `--use-password` | — | use `sshpass` + `CORE_PASSWORD` instead of key-based SSH |

## Credentials & paths

`CORE_USER` / `CORE_HOST` / `CORE_PASSWORD` come from the repo-root `.env`
(key-based SSH is the default). CORE-side roots default to the live
`/data2/core-rad-fni/flakhal/preprocessing` tree; override with `CORE_PREP_ROOT`.

## Relation to the CORE path

This does **not** replace `src/core/submit_array.sh --stage postprocessing` —
it's an alternative that moves the postprocessing compute onto Fritz and lets it
run continuously. Use one or the other per cohort, not both (both would produce
the same flat product). The CORE path keeps everything on one machine; this path
frees CORE's throttled CPUs and starts postprocessing hours earlier.
