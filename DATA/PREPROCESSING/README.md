# Preprocessing Pipeline — AD Early Detection

> **Reference pipeline:** [Pipeline overall v2.md](Pipeline%20overall%20v2.md)

---

## Where each step runs

| Step | Description | Machine | Notes |
|------|-------------|---------|-------|
| **1.1** | DICOM → NIfTI (`dcm2niix`) | **Fritz** | Install `dcm2niix`; only needed for ADNI (OASIS3 bold is already NIfTI/BIDS-structured) |
| **1.2** | Merge fMRI runs (`fslmerge`) | **Fritz** | FSL 6.0.7 is installed; applies to OASIS3 multi-run sessions |
| **1.3** | Remove empty `anat/` dirs (`find`) | **Fritz** | Pure bash |
| **2** | BIDS organisation | **Fritz** | stdlib only; restructures data into BIDS hierarchy |
| **2.1** | Rename BIDS folders | **Fritz** | Pure bash |
| **2.2** | Copy `dataset_description.json` | **Fritz** | Trivial bash |
| **→** | **rsync BIDS output → CORE** | Fritz → CORE | `rsync -avuzh` to `/home/flakhal/` on CORE |
| **3** | fMRIPrep (`fmriprep-23.0.2.simg`) | **CORE** | Singularity array job; `smoteval`'s image at `/data2/core-rad/smoteval/images/` |
| **4** | Postprocessing (`postprocessing.sif`) | **CORE** | Singularity array job; `flbrandl`'s image at `/data2/core-rad/flbrandl/` |
| **5** | Transfer to LRZ | CORE → LRZ | `rsync -avuzh` |
| **6** | Reorient | LRZ | `final_reorient.py`; requires FSL + FreeSurfer |

---

## Datasets

| Dataset | Raw format on Fritz | BIDS output location |
|---------|---------------------|----------------------|
| **OASIS3** | Already NIfTI, semi-BIDS in `DATA/OASIS3/__bold_and_smri__/sub-*/ses-*/func/` | Merge multi-run → full BIDS in `DATA/OASIS3/BIDS/` |
| **ADNI** | DICOM `.zip` files in `DATA/ADNI/__dicom_zips_flat__/` | dcm2niix → BIDS in `DATA/ADNI/BIDS/` |

---

## Fritz steps (1–2): quick-start

The orchestrator script for Fritz is:

```
DATA/PREPROCESSING/src/fritz/run_fritz_pipeline.sh
```

It runs Steps 1–2 for **both** datasets (or a single one), then rsyncs the
BIDS output to CORE.

Usage:

```bash
# Run pipeline for both datasets and rsync to CORE
bash DATA/PREPROCESSING/src/fritz/run_fritz_pipeline.sh --dataset both

# Run for OASIS3 only
bash DATA/PREPROCESSING/src/fritz/run_fritz_pipeline.sh --dataset oasis3

# Run for ADNI only
bash DATA/PREPROCESSING/src/fritz/run_fritz_pipeline.sh --dataset adni

# Dry-run (skip rsync)
bash DATA/PREPROCESSING/src/fritz/run_fritz_pipeline.sh --dataset both --dry-run
```

### Step 1.2 run merging — how BOLD runs are grouped

`organize_bids_dataset()` no longer merges every `*_run-*_bold` file in a
session into a single `task-rest` output. That was unsafe: some sessions
contain **several distinct task acquisitions with different matrix sizes**, and
`fslmerge` aborts on a size mismatch — under `set -e` that killed the entire
run. The current logic:

1. **Group by task label.** Each distinct `_task-<label>_` in a session is
   merged independently and written as its own BIDS output
   (`sub-*_ses-*_task-<label>_bold.nii.gz`). A session can therefore now
   produce more than one func file; downstream fMRIPrep processes them all.
2. **Volume filter** (unchanged): runs with `< 50` TRs are dropped as
   localisers / single-band references (SBRef).
3. **Spatial-dim guard:** within a task group, only runs matching the first
   valid run's `dim1×dim2×dim3` are merged; a mismatched run is logged and
   skipped rather than aborting. A wholesale `fslmerge` failure is also caught,
   logged, and skipped.

#### Sessions needing later curation

The grouping keeps the pipeline running end-to-end, but the following sessions
have **multiple task acquisitions and/or dropped runs** — you'll likely want to
decide which acquisition to keep for analysis. This is deferred, not resolved.

**OASIS3 — 140 sessions across 138 subjects mix two task labels** (each is now
emitted as two separate func files):

| Task combination | Sessions |
|---|---|
| `task-restingstate` + `task-restingstateMB4` | 137 |
| `task-rest` + `task-testrest` | 3 |

Regenerate the full per-session list any time with:

```bash
for ses in DATA/OASIS3/__bold_and_smri__/sub-*/ses-*/func; do
  t=$(ls "$ses"/*_run-*_bold.nii.gz 2>/dev/null \
        | sed -E 's/.*_task-([A-Za-z0-9]+)_run.*/\1/' | sort -u | tr '\n' ' ')
  [[ $(wc -w <<<"$t") -gt 1 ]] && echo "$(dirname "${ses#DATA/OASIS3/__bold_and_smri__/}"): $t"
done
```

**ADNI — 4 sessions have a run dropped by the volume filter** (SBRef or
truncated scan mislabeled as a bold run; the real timeseries is kept):

| Session | Runs (vols / dims) | Kept |
|---|---|---|
| `sub-ADNI070S6229/ses-d0000` | run-01 = 2v, run-02 = 197v (64×64×48) | run-02 |
| `sub-ADNI073S6669/ses-d0000` | run-01 = 976v, run-02 = 1v (88×88×64) | run-01 |
| `sub-ADNI073S6673/ses-d0000` | run-01 = 976v, run-02 = 1v (88×88×64) | run-01 |
| `sub-ADNI073S6929/ses-d0000` | run-01 = 1v, run-02 = 976v (88×88×64) | run-02 |

These SBRef scans should ideally be excluded or renamed `_sbref` upstream in
`DATA/ADNI/src/unzip/convert_to_bids.py` (its series-description → BIDS-name
mapping doesn't distinguish SBRef from the real acquisition).

---

## CORE steps (3–4): SLURM array jobs

Reference scripts (adapt paths before submitting):

| Script | Purpose |
|--------|---------|
| [`src/core/fmriprep_array_v1.slurm`](src/core/fmriprep_array_v1.slurm) | fMRIPrep array job |
| [`src/core/postprocessing_array_v1.slurm`](src/core/postprocessing_array_v1.slurm) | Postprocessing array job |

Submit example:

```bash
# On CORE — after BIDS data has arrived via rsync
sbatch --array=1-272%10 src/core/fmriprep_array_v1.slurm
```

---

## OASIS3/ADNI overlap workflow: push, submit early, pull incrementally

Raw→BIDS push is a one-time ~1–3h rsync; fMRIPrep + postprocessing is days of
aggregate SLURM-array compute. Don't wait for the push to finish before
submitting the array job — fMRIPrep processes subjects independently, and
each array task waits (bounded, 30 min) for its own subject to land if the
rsync hasn't reached it yet. This overlaps the transfer under the compute for
free, without a bespoke streaming pipeline.

Each step below is a separate, independently rerunnable command — nothing is
chained automatically, so you control when each one runs and can check its
log (`DATA/PREPROCESSING/logs/`, color-coded) before triggering the next.

```bash
# 1. Push: organizes BIDS locally, then rsyncs to CORE — run in background
nohup bash src/fritz/run_fritz_pipeline.sh --dataset both > push.log 2>&1 &

# 2. Submit the fMRIPrep array immediately — sized from the local subject
#    count, which is already final even though the remote rsync is still running
bash src/fritz/submit_fmriprep_array_core.sh --dataset oasis3
bash src/fritz/submit_fmriprep_array_core.sh --dataset adni

# Check progress at any time (color-coded by SLURM task state)
bash src/fritz/submit_fmriprep_array_core.sh --status --dataset both

# 3. Once fMRIPrep has processed enough subjects, submit postprocessing
bash src/fritz/submit_fmriprep_array_core.sh --dataset oasis3 --stage postprocessing

# 4. Pull derivatives back at any time — safe to rerun repeatedly as more
#    sessions finish (only transfers new/changed files)
bash src/fritz/pull_derivatives_from_core.sh --dataset both
```

New scripts:

| Script | Purpose |
|--------|---------|
| [`src/core/fmriprep_array_oasis_adni.slurm`](src/core/fmriprep_array_oasis_adni.slurm) | fMRIPrep array for OASIS3/ADNI's flat BIDS layout; waits (bounded) for a subject to arrive if the push hasn't reached it yet |
| [`src/core/postprocessing_array_oasis_adni.slurm`](src/core/postprocessing_array_oasis_adni.slurm) | Postprocessing array for OASIS3/ADNI |
| [`src/fritz/submit_fmriprep_array_core.sh`](src/fritz/submit_fmriprep_array_core.sh) | Sizes and submits either array job on CORE; `--status` shows a live squeue snapshot |
| [`src/fritz/pull_derivatives_from_core.sh`](src/fritz/pull_derivatives_from_core.sh) | Incremental, rerunnable rsync of derivatives back to `DATA/<COHORT>/derivatives/` |

Both `.slurm` files require `DATASET=oasis3|adni` to be exported at submit
time (`submit_fmriprep_array_core.sh` does this for you).

---

## Requirements

| Tool | Where | Notes |
|------|-------|-------|
| `dcm2niix` | Fritz | `sudo apt install dcm2niix` or conda |
| FSL 6.0.7 | Fritz | Already installed; ensure `$FSLDIR` is set |
| Singularity / Apptainer | CORE | Available system-wide |
| FreeSurfer license | CORE | `/data2/core-rad/swunderl/freesurfer/license.txt` |

---

## Logs

Runtime logs from the Fritz script are written to `DATA/PREPROCESSING/logs/`.
