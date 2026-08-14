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

There is no orchestrator — each step is one command you run deliberately, on
one named machine. See
[`DATA/PREPROCESSING/pipeline-Fritz-CORE.md`](../PREPROCESSING/pipeline-Fritz-CORE.md)
for the full run order and the rationale.

Organising to BIDS is purely local and never touches CORE:

```bash
# Both datasets (or --dataset oasis3 / --dataset adni)
bash DATA/PREPROCESSING/src/fritz/organize_bids.sh --dataset both

# Smoke test: first 2 subjects only, into DATA/<COHORT>/BIDS_smoketest
bash DATA/PREPROCESSING/src/fritz/organize_bids.sh --dataset oasis3 --limit 2
```

Inspect the output under `DATA/<COHORT>/BIDS/`, then push it as a separate
step (see the CORE section below).

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

The active OASIS3/ADNI scripts are `src/core/*_oasis_adni.slurm` (flat
`sub-*/ses-*` BIDS layout). The `*_v1.slurm` scripts are the legacy
DELCODE-non-converter versions (per-session `INPUT_BASE/<session>/sub-*`
layout) — do not use them for OASIS3/ADNI.

---

## OASIS3/ADNI workflow: push, submit on CORE, pull

Each step is a separate, independently rerunnable command — nothing is chained
automatically, so you control when each one runs and can check its log
(`DATA/PREPROCESSING/logs/`, color-coded) before triggering the next. Steps 1–3
run on Fritz, 4–5 on CORE, 6 back on Fritz.

The array is sized **on CORE**, from the subjects actually present there — not
guessed from a local count. So the push must finish before you submit: the
fMRIPrep job fails loudly on a missing subject rather than waiting for one to
arrive (the old 30-minute arrival poll was removed 2026-07-17, since a failed
push was indistinguishable from a slow one).

```bash
# ── On Fritz ──
# 1. Organise to BIDS locally (see the Fritz quick-start above)
bash src/fritz/organize_bids.sh --dataset both

# 2. Push BIDS to CORE — a one-time ~1–3h rsync. Let it finish.
bash src/fritz/push_bids_to_core.sh --dataset both

# 3. Ship the CORE-side scripts + create the SLURM log dirs.
#    Rerun whenever a .slurm file or submit_array.sh changes.
bash src/fritz/push_scripts_to_core.sh

# ── On CORE ──
ssh $CORE_USER@$CORE_HOST
cd /data2/core-rad-fni/flakhal/preprocessing/scripts/core

# 4. Submit fMRIPrep — sizes --array from CORE's own subject count
bash submit_array.sh --dataset oasis3 --stage fmriprep
bash submit_array.sh --dataset adni   --stage fmriprep

# Check progress at any time
squeue -u "$USER" -n fmriprep_oasis_adni

# 5. Once fMRIPrep has processed enough subjects, submit postprocessing
bash submit_array.sh --dataset oasis3 --stage postprocessing

# ── Back on Fritz ──
# 6. Pull derivatives back at any time — safe to rerun repeatedly as more
#    sessions finish (only transfers new/changed files)
bash src/fritz/pull_derivatives_from_core.sh --dataset both
```

Scripts:

| Script | Machine | Purpose |
|--------|---------|---------|
| [`src/fritz/organize_bids.sh`](src/fritz/organize_bids.sh) | Fritz | Steps 1.2/1.3/2/2.2 — merge runs, organise to BIDS. Purely local |
| [`src/fritz/push_bids_to_core.sh`](src/fritz/push_bids_to_core.sh) | Fritz | rsync `DATA/<COHORT>/BIDS` → CORE fMRIPrep `INPUT_BASE` |
| [`src/fritz/push_scripts_to_core.sh`](src/fritz/push_scripts_to_core.sh) | Fritz | rsync `src/core/*` → CORE; creates the SLURM log dirs |
| [`src/core/submit_array.sh`](src/core/submit_array.sh) | **CORE** | Sizes `--array` from CORE's filesystem and `sbatch`es either stage |
| [`src/core/fmriprep_array_oasis_adni.slurm`](src/core/fmriprep_array_oasis_adni.slurm) | CORE | fMRIPrep array for OASIS3/ADNI's flat BIDS layout |
| [`src/core/postprocessing_array_oasis_adni.slurm`](src/core/postprocessing_array_oasis_adni.slurm) | CORE | Postprocessing array for OASIS3/ADNI |
| [`src/fritz/pull_derivatives_from_core.sh`](src/fritz/pull_derivatives_from_core.sh) | Fritz | Incremental, rerunnable rsync of derivatives back to `DATA/<COHORT>/derivatives/` |

Both `.slurm` files require `DATASET=oasis3|adni` to be exported at submit
time (`submit_array.sh` does this for you).

Add `--dry-run` to any push script or to `submit_array.sh` to see what it would
do without transferring or submitting. `--limit N` on `organize_bids.sh` and
`push_bids_to_core.sh` runs a 2-subject smoke test against a separate
`_smoketest`-suffixed tree, never the real dataset.

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
