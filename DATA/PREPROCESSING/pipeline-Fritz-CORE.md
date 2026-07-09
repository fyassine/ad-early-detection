# Fritz → CORE Preprocessing Pipeline Plan
_Last updated: 2026-07-08_

## Status

| Section | Status |
|---------|--------|
| Steps 1–2 Fritz script | ✅ Written (`src/fritz/run_fritz_pipeline.sh`) |
| README updated | ✅ Done |
| ADNI series heuristics | ✅ **Verified** — full scan of all zips in `__dicom_zips_flat__` |
| ADNI T1w availability | ✅ **Available** — T1w/MPRAGE zips are already mixed into `__dicom_zips_flat__`; a background LONI download job continues adding more sMRI zips |
| ADNI → BIDS conversion | ✅ **Automated** — `src/unzip/*.py` builds `DATA/ADNI/__bold_and_smri__` from the raw zips |
| OASIS3 T1w availability | ✅ **Available** — every session in `__bold_and_smri__` already has an `anat/` folder |
| Steps 3–4 CORE SLURM scripts | ✅ Reference scripts in `src/core/` |

---

## Execution split

| Step | Description | Machine |
|------|-------------|---------|
| 1.1 | DICOM → NIfTI (`dcm2niix`) | Local (ADNI only, before this script runs — `DATA/ADNI/src/unzip/*.py`) |
| 1.2 | Merge fMRI runs (`fslmerge`) | **Fritz** (both datasets, multi-run sessions) |
| 1.3 | Remove empty `anat/` dirs | **Fritz** (safety net) |
| 2 | BIDS organisation (incl. copying `anat/`) | **Fritz** |
| 2.2 | Copy `dataset_description.json` | **Fritz** |
| → | **rsync BIDS → CORE** | Fritz → CORE |
| 3 | fMRIPrep (Singularity) | **CORE** |
| 4 | Postprocessing (Singularity) | **CORE** |

Both datasets' raw source (`__bold_and_smri__`) share the identical
`sub-*/ses-d<NNNN>/{anat,func}/...` shape, so `run_fritz_pipeline.sh` runs
both through the same `organize_bids_dataset()` function.

---

## Dataset notes

### OASIS3

- Raw data: `DATA/OASIS3/__bold_and_smri__/sub-*/ses-d<NNNN>/{anat,func}/`
- Already in NIfTI format with BIDS-like naming
- Some sessions have multiple func runs (`run-01`, `run-02`, `run-03`)
  - `run-01` is often very short (scout/localiser); runs < 50 TRs are filtered out
  - Remaining runs are merged with `fslmerge -t`
- `anat/` T1w files are present for essentially every session (1198 subjects,
  2137 session-level `anat/` dirs) and are now copied into the BIDS output —
  previously `run_oasis3()` `mkdir -p`'d an empty `anat/` and Step 1.3 quietly
  deleted it, discarding all T1w before the rsync to CORE. Fixed.

---

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
- **BIDS subject ID mapping**: `002_S_1261` → `sub-ADNI002S1261` ✅
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
| `rsfmri\|fcmri\|fmri\|resting\|bold\|rest` (case-insensitive) | func (BOLD) | `run_fritz_pipeline.sh`'s original BOLD regex, reused in `src/unzip/scan_zip_manifest.py` |

Zips matching neither are logged as `unclassified` and excluded.

#### ADNI → BIDS conversion pipeline (run before `run_fritz_pipeline.sh`)

```bash
cd DATA/ADNI/src/unzip
python build_visit_baselines.py   # -> __metadata__/adni_visit_baselines.csv
python scan_zip_manifest.py       # -> __metadata__/adni_bids_manifest.csv
python convert_to_bids.py         # -> ../../__bold_and_smri__/sub-ADNI*/ses-d*/{anat,func}/...
```

All three are resumable — re-running only processes new/missing zips. This
populates `DATA/ADNI/__bold_and_smri__` in the exact same layout as
`DATA/OASIS3/__bold_and_smri__`, so `run_fritz_pipeline.sh --dataset adni`
consumes it identically to OASIS3 (no more in-script `unzip`/`dcm2niix`).

---

## Fritz script usage

```bash
# Both datasets + rsync to CORE
bash DATA/PREPROCESSING/src/fritz/run_fritz_pipeline.sh --dataset both

# Single dataset
bash DATA/PREPROCESSING/src/fritz/run_fritz_pipeline.sh --dataset oasis3
bash DATA/PREPROCESSING/src/fritz/run_fritz_pipeline.sh --dataset adni

# Dry-run (no rsync)
bash DATA/PREPROCESSING/src/fritz/run_fritz_pipeline.sh --dataset both --dry-run
```

SSH config required on Fritz (`~/.ssh/config`):
```
Host HOST
    HostName srvcorem2.med.uni-muenchen.de
    User flakhal
```
Ensure key-based SSH auth: `ssh-copy-id flakhal@HOST`

---

## CORE SLURM scripts

| Script | Step | Key flags to adapt |
|--------|------|--------------------|
| `src/core/fmriprep_array_v1.slurm` | Step 3 | `INPUT_BASE`, `OUTPUT_BASE`, `FS_LICENSE` — `--fs-no-reconall` no longer strictly required now that T1w is available for both datasets, but keep if FreeSurfer recon time is a concern |
| `src/core/postprocessing_array_v1.slurm` | Step 4 | `INPUT_BASE`, `OUTPUT_BASE` |

---

## Open items

- [ ] Confirm CORE destination path (`/home/flakhal/` vs `/data2/` project space)
- [ ] Re-run `src/unzip/*.py` once the background `ADNI_Converters_sMRI` LONI
      download finishes, to pick up the T1w zips it's still adding
- [ ] Some ADNI subjects only ever have resting-state fMRI in
      `__dicom_zips_flat__` with no matched T1w (no MPRAGE zip present) —
      `organize_bids_dataset()` still includes their `func/`-only sessions;
      fMRIPrep will need `--fs-no-reconall` for those specific subjects
