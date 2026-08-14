# Fritz → CORE Preprocessing Pipeline Plan
_Last updated: 2026-07-17_

Each step below is one command you run deliberately, on one named machine.
Nothing chains automatically into the next step.

## Why manual

This pipeline used to be two orchestrators that each hid the machine boundary
they crossed: `run_fritz_pipeline.sh` organized BIDS **and** rsynced it to CORE
in one shot, and `submit_fmriprep_array_core.sh` ran on Fritz but sized the
SLURM array from the *local* subject count before `sbatch`ing over SSH — a local
guess about remote state.

That coupling forced a workaround: the fMRIPrep job polled for up to 30 minutes
waiting for its subject to arrive, because it was designed to be submitted
*while* the push was still in flight. A failed push was indistinguishable from a
slow one for half an hour.

Now each step is inspectable before the next begins, and CORE-side decisions are
made **on CORE** from the data that is actually there. **Do not reintroduce the
coupling** — in particular, the arrival-wait guard was deliberately removed
(2026-07-17). Step 2 always precedes step 5, so a missing subject is a real
error and the job fails loudly.

---

## Run order

| # | Machine | Command |
|---|---------|---------|
| 0 | Fritz | `cd DATA/ADNI/src/unzip && python build_visit_baselines.py && python scan_zip_manifest.py && python convert_to_bids.py` — ADNI only |
| 1 | Fritz | `bash src/fritz/organize_bids.sh --dataset oasis3` |
| 2 | Fritz | `bash src/fritz/push_bids_to_core.sh --dataset oasis3` |
| 3 | Fritz | `bash src/fritz/push_scripts_to_core.sh` |
| 4 | CORE | `ssh $CORE_USER@$CORE_HOST` |
| 5 | CORE | `bash /data2/core-rad-fni/flakhal/preprocessing/scripts/core/submit_array.sh --dataset oasis3 --stage fmriprep` |
| 6 | CORE | `squeue -u "$USER" -n fmriprep_oasis_adni` — wait for completion |
| 7 | CORE | `bash …/scripts/core/submit_array.sh --dataset oasis3 --stage postprocessing` |
| 8 | Fritz | `bash src/fritz/pull_derivatives_from_core.sh --dataset oasis3` |

Swap `--dataset adni` for the ADNI cohort, or `--dataset both` on the Fritz-side
scripts (steps 1, 2, 8) to do both at once. Step 3 is dataset-independent.
Step 5's `--dataset` takes the **literal** value exported to the job
(`oasis3`, `adni`, `oasis3_smoketest`), not a cohort key.

Rerun step 3 whenever a `.slurm` file or `submit_array.sh` changes locally — it
is the only thing that updates CORE's copy.

### Smoke test

Steps 1 and 2 take `--limit N` to organize and push only the first N subjects
into a separate `_smoketest`-suffixed tree, never the real dataset:

```bash
# Fritz
bash src/fritz/organize_bids.sh   --dataset oasis3 --limit 2   # -> DATA/OASIS3/BIDS_smoketest
bash src/fritz/push_bids_to_core.sh --dataset oasis3 --limit 2 # -> .../data/oasis3_smoketest
# CORE
bash .../scripts/core/submit_array.sh --dataset oasis3_smoketest --stage fmriprep
```

Add `--dry-run` to either push script to see the rsync it would run without
transferring. Add `--dry-run` to `submit_array.sh` to see the sized `sbatch`
line without submitting.

### Optional: push the raw pre-BIDS trees

`bash src/fritz/push_bold_and_smri_to_core.sh --dataset both` ships the raw
`__bold_and_smri__` trees to `…/preprocessing/data/raw/<COHORT>/`. This is not
part of the fMRIPrep path — it is a backup/staging convenience, and nothing
downstream reads from it.

---

## Script index

| Script | Machine | Role |
|--------|---------|------|
| [`src/fritz/organize_bids.sh`](src/fritz/organize_bids.sh) | Fritz | Steps 1.2/1.3/2/2.2 — merge runs, organise to BIDS. Purely local, never touches CORE |
| [`src/fritz/push_bids_to_core.sh`](src/fritz/push_bids_to_core.sh) | Fritz | rsync `DATA/<COHORT>/BIDS` → CORE fMRIPrep `INPUT_BASE` |
| [`src/fritz/push_scripts_to_core.sh`](src/fritz/push_scripts_to_core.sh) | Fritz | rsync `src/core/*` → CORE; creates the SLURM log dirs |
| [`src/core/submit_array.sh`](src/core/submit_array.sh) | **CORE** | Sizes `--array` from CORE's own filesystem and `sbatch`es either stage |
| [`src/fritz/pull_derivatives_from_core.sh`](src/fritz/pull_derivatives_from_core.sh) | Fritz | Incremental, rerunnable rsync of derivatives → `DATA/<COHORT>/derivatives/` |
| [`src/fritz/push_bold_and_smri_to_core.sh`](src/fritz/push_bold_and_smri_to_core.sh) | Fritz | Optional raw `__bold_and_smri__` push (not on the fMRIPrep path) |

Credentials (`CORE_USER`/`CORE_HOST`/`CORE_PASSWORD`) come from the repo-root
`.env` for every Fritz-side script. Key-based SSH auth is the default; pass
`--use-password` to force `sshpass`. `submit_array.sh` needs none of this — it
runs on CORE and touches only the local filesystem.

---

## Pipeline steps

| Step | Description | Machine |
|------|-------------|---------|
| 1.1 | DICOM → NIfTI (`dcm2niix`) | Fritz (ADNI only — `DATA/ADNI/src/unzip/*.py`) |
| 1.2 | Merge fMRI runs (`fslmerge`) | Fritz (both datasets, multi-run sessions) |
| 1.3 | Remove empty `anat/` dirs | Fritz (safety net) |
| 2 | BIDS organisation (incl. copying `anat/`) | Fritz |
| 2.2 | Copy `dataset_description.json` | Fritz |
| 3 | fMRIPrep (Singularity array) | CORE |
| 4 | Postprocessing (Singularity array) | CORE |

Both datasets' raw source (`__bold_and_smri__`) share the identical
`sub-*/ses-d<NNNN>/{anat,func}/...` shape, so `organize_bids.sh` runs both
through the same `organize_bids_dataset()` function.

---

## Dataset notes

### OASIS3

- Raw data: `DATA/OASIS3/__bold_and_smri__/sub-*/ses-d<NNNN>/{anat,func}/`
- Already in NIfTI format with BIDS-like naming
- Some sessions have multiple func runs (`run-01`, `run-02`, `run-03`)
  - `run-01` is often very short (scout/localiser); runs < 50 TRs are filtered out
  - Remaining runs are merged with `fslmerge -t`, but only within a single task
    label and only across runs whose spatial dims match — a mismatch would abort
    `fslmerge` and, under `set -e`, kill the whole run
- `anat/` T1w files are present for essentially every session (1198 subjects,
  2137 session-level `anat/` dirs) and are copied into the BIDS output —
  previously an empty `anat/` was `mkdir -p`'d and Step 1.3 quietly deleted it,
  discarding all T1w before the push to CORE. Fixed.

### ADNI

- Raw source (zips): `DATA/ADNI/__dicom_zips_flat__/` — flat folder of
  DICOM zips, growing as a background LONI download job (`ADNI_Converters_sMRI`
  collection) continues adding paired T1w/MPRAGE zips
- Zip naming: `<SiteID>_S_<SubjectID>_<ImageID>.zip`, e.g.
  `002_S_1261_831069.zip` → subject `002_S_1261`, image `831069`
- Each zip contains exactly **one series** (one scan per zip); the series
  description and acquisition date are read straight from the zip's internal
  DICOM path (`ADNI/<subject>/<series>/<date_time>/<image_id>/...`), not
  from any external metadata CSV
- **BIDS subject ID mapping**: `002_S_1261` → `sub-ADNI002S1261`
- **BIDS session mapping**: `ses-d<NNNN>` = zero-padded days since the
  subject's baseline visit, computed by `src/unzip/build_visit_baselines.py`
  from the DXSUM diagnosis CSV (`VISCODE == "bl"`, falling back to `"4_bl"`,
  falling back to the subject's earliest DXSUM row), then clamped so it never
  goes negative relative to that subject's own earliest scan — see
  `effective_baselines()` in `src/unzip/scan_zip_manifest.py`

#### Series classification

| Regex | Type | Source |
|-------|------|--------|
| `MPRAGE\|MP-RAGE\|MP RAGE\|SPGR\|IR-SPGR\|FSPGR\|3D\s*T1` | anat (T1w) | `src/download/download_adni_smri.py`'s `T1W_DESCRIPTION_RE`, reused in `src/unzip/scan_zip_manifest.py` |
| `rsfmri\|fcmri\|fmri\|resting\|bold\|rest` (case-insensitive) | func (BOLD) | `organize_bids.sh`'s original BOLD regex, reused in `src/unzip/scan_zip_manifest.py` |

Zips matching neither are logged as `unclassified` and excluded.

#### ADNI → BIDS conversion (step 0, before `organize_bids.sh`)

```bash
cd DATA/ADNI/src/unzip
python build_visit_baselines.py   # -> __metadata__/adni_visit_baselines.csv
python scan_zip_manifest.py       # -> __metadata__/adni_bids_manifest.csv
python convert_to_bids.py         # -> ../../__bold_and_smri__/sub-ADNI*/ses-d*/{anat,func}/...
```

All three are resumable — re-running only processes new/missing zips. This
populates `DATA/ADNI/__bold_and_smri__` in the exact same layout as
`DATA/OASIS3/__bold_and_smri__`, so `organize_bids.sh --dataset adni` consumes
it identically to OASIS3.

---

## CORE paths

Everything below lives under `/data2/core-rad-fni/flakhal/preprocessing/` —
`flakhal` owns this tree with no per-user quota, unlike `/home` (which hit
`Disk quota exceeded` mid-smoketest on 2026-07-10). `DATASET` is `oasis3`,
`adni`, or the `*_smoketest` variant.

| Purpose | Path |
|---------|------|
| BIDS input (`INPUT_BASE`) | `…/preprocessing/data/${DATASET}` |
| fMRIPrep work dir (`WORK_DIR`) | `…/preprocessing/data/workdir/${DATASET}` |
| fMRIPrep output (`OUTPUT_BASE`) | `…/preprocessing/outputs/${DATASET}/fmriprep` |
| Postprocessed output | `…/preprocessing/outputs/${DATASET}/postprocessed` |
| Scripts synced here (step 3) | `…/preprocessing/scripts/core/` |
| Job logs (`--output`/`--error`) | `…/preprocessing/logs/{fmriprep,postprocessing}/` |
| TemplateFlow cache (`TEMPLATEFLOW_HOME`) | `/data2/core-rad-fni/flakhal/templateflow` |
| Postprocessing image (`POSTPROC_SIMG`) | `…/preprocessing/images/postprocessing.sif` (staged once — see below) |

`submit_array.sh` derives all of these from a single `PREP_ROOT`. The `.slurm`
files must hardcode the same root in their `#SBATCH --output`/`--error`
directives — SBATCH directives are parsed before the script runs and cannot read
shell variables. **If you move `PREP_ROOT`, update both `.slurm` files to match.**

Shared read-only resources (colleagues' trees, all under `/data2/core-rad-fni/`
— **not** `/data2/core-rad/`, which is permission-denied for `flakhal`):

| Resource | Path |
|----------|------|
| fMRIPrep image (`FMRIPREP_SIMG`) | `/data2/core-rad-fni/smoteval/images/fmriprep-23.0.2.simg` |
| FreeSurfer license (`FS_LICENSE`) | `/data2/core-rad-fni/swunderl/freesurfer/license.txt` |

The postprocessing image used to be read directly from a colleague's tree
(`/data2/core-rad-fni/flbrandl/postprocessing.sif`), which broke when an even
staler copy pointed at the bare `/data2/core-rad/flbrandl/…`
(permission-denied for `flakhal`) — the container never launched yet the job
logged `Postprocessing completed` and exited `COMPLETED`. It is now staged
once into flakhal's own tree so postprocessing owns its dependency:

```bash
ssh $CORE_USER@$CORE_HOST \
  'mkdir -p /data2/core-rad-fni/flakhal/preprocessing/images && \
   cp /data2/core-rad-fni/flbrandl/postprocessing.sif \
      /data2/core-rad-fni/flakhal/preprocessing/images/postprocessing.sif'
```

Both `*_oasis_adni.slurm` scripts check the Singularity exit code and fail
loudly (`>&2` + non-zero exit) instead of printing `… completed` after a
container that never ran.

`--fs-no-reconall` is passed in `fmriprep_array_oasis_adni.slurm`; T1w is
available for both datasets, but recon-all stays off (time) except where a
subject-specific rerun needs it.

The `*_v1.slurm` scripts are the legacy DELCODE-non-converter versions
(per-session `INPUT_BASE/<session>/sub-*` layout) — do not use them for
OASIS3/ADNI, and `push_scripts_to_core.sh` does not ship them.

---

## Status

| Item | Status |
|------|--------|
| Manual step split | ✅ Done 2026-07-17 — `organize_bids.sh` / `push_bids_to_core.sh` / `push_scripts_to_core.sh` / `submit_array.sh` (on CORE) |
| Arrival-wait guard removed | ✅ Done 2026-07-17 — fMRIPrep array now fails fast on a missing subject |
| ADNI series heuristics | ✅ **Verified** — full scan of all zips in `__dicom_zips_flat__` |
| ADNI T1w availability | ✅ **Available** — T1w/MPRAGE zips mixed into `__dicom_zips_flat__`; a background LONI download job continues adding more |
| ADNI → BIDS conversion | ✅ **Automated** — `src/unzip/*.py` builds `DATA/ADNI/__bold_and_smri__` from the raw zips |
| OASIS3 T1w availability | ✅ **Available** — every session in `__bold_and_smri__` has an `anat/` folder |
| fMRIPrep smoketest | ✅ **Passed** 2026-07-16 — OASIS3 2-subject array (job 4177046) both tasks `COMPLETED 0:0`, derivatives + QC reports written, no disk-quota errors |
| CORE destination path | ✅ Resolved 2026-07-16 — `/data2/core-rad-fni/flakhal/preprocessing/` (no per-user quota, unlike `/home`) |

## Open items

- [ ] Re-run the smoketest against the manual step sequence to confirm the
      rewritten scripts reproduce the 2026-07-16 result
- [ ] Re-run `src/unzip/*.py` once the background `ADNI_Converters_sMRI` LONI
      download finishes, to pick up the T1w zips it's still adding
- [ ] Some ADNI subjects only ever have resting-state fMRI in
      `__dicom_zips_flat__` with no matched T1w (no MPRAGE zip present) —
      `organize_bids_dataset()` still includes their `func/`-only sessions;
      fMRIPrep will need `--fs-no-reconall` for those specific subjects
