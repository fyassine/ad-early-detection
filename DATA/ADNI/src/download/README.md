# ADNI Download Pipeline

This directory contains the **active scripts** for downloading ADNI resting-state fMRI data from the [LONI IDA portal](https://ida.loni.usc.edu).

---

## Setup (run once)

```bash
pip install playwright python-dotenv pandas tqdm
playwright install chromium

# Copy the template and fill in your LONI credentials
cp ../env.example ../.env && nano ../.env
```

---

## Scripts

### `download_collection.py` — Primary downloader ⭐
Logs in, navigates to a named LONI Data Collection, and batch-downloads DICOM ZIPs
via the "Not Downloaded" view. All other scripts in this package import from here.

```bash
# Default collection (mci-all-v2), batch size 5:
python download_collection.py

# Different collection, headless off (shows browser window):
python download_collection.py --collection ADNI_Converters_fMRI --headless false

# Available flags:
python download_collection.py --help
```

**Output:** `DATA/ADNI/__dicom_zips_flat__/{subject_id}_{image_id}.zip`  
**Logs:** `logs/adni-download/<YYYYMMDD_HHMMSS>/adni_collection_download.log`

---

### `download_collection_jar.py` — IDA Jar downloader
Alternative to `download_collection.py`. Downloads one image at a time using the
official IDA Downloader jar (`ida_downloader/IdaDownloader_15May2026.jar`).

Use this when Playwright-based batch downloads stall (the jar handles chunked/resumable
transfers and is IP-matched to the session that mints the URL).

```bash
# Smoke-test (1 image):
python download_collection_jar.py --collection ADNI_Converters_fMRI --limit 1

# Full run:
python download_collection_jar.py --collection ADNI_Converters_fMRI

python download_collection_jar.py --help
```

**Requires:** Java 12+ at `/usr/lib/jvm/java-21-openjdk-amd64` (or pass `--java-home`).  
**Output:** `DATA/ADNI/__dicom_zips_flat__/{subject_id}_{image_id}.zip`

---

### `convert_dicom_zips.py` — DICOM → NIfTI converter
Converts `{subject_id}_{image_id}.zip` files (raw DICOM ZIPs) to `.nii.gz` via `dcm2niix`.
Run alongside or after the downloader — it communicates only through the filesystem.

```bash
# One-pass conversion of all pending ZIPs:
python convert_dicom_zips.py

# Watch mode — keeps polling for new ZIPs every 30s:
python convert_dicom_zips.py --watch

python convert_dicom_zips.py --help
```

**Requires:** `dcm2niix` on PATH (`sudo apt install dcm2niix`).  
**Input:** `DATA/ADNI/__dicom_zips_flat__/*.zip`  
**Output:** `DATA/ADNI/__fmri_wholebrain_sch200_flat__/{subject_id}_{image_id}.nii.gz`

---

### `download_adni_fmri.py` — Earlier self-contained downloader
An older, self-contained downloader that uses the Advanced Search flow
(per-image search → add to collection → download as NIfTI). Still valid and
kept here as a reference/alternative entry point.

```bash
python download_adni_fmri.py --dry-run    # preview only
python download_adni_fmri.py --pilot-one  # download 1 image as smoke test
python download_adni_fmri.py              # full run
```

---

### `download_adni_smri.py` — T1w/sMRI image ID resolver
Finds the T1-weighted / MPRAGE structural scan matching each fMRI scan already
downloaded into `__dicom_zips_flat__`, by searching LONI for the same subject
on the *exact same exam date* (no fallback to nearby dates). Does **not**
touch LONI collections or download anything itself — it only resolves image
IDs and writes them to a plain text file, same comma-separated format as
`__metadata__/image_ids.txt`, so they can be pasted into LONI's Advanced
Search "Image ID" field to build a collection by hand (or fed into any tool
that accepts a list of image IDs).

```bash
python download_adni_smri.py --dry-run                    # preview target list only
python download_adni_smri.py --pilot-one --headless false # resolve 1, visible browser
python download_adni_smri.py                               # full run (resume-safe)

python download_adni_smri.py --help
```

**Resolution audit trail:** `DATA/ADNI/__metadata__/smri_resolution.csv`
(`subject_id, fmri_image_id, fmri_date, smri_image_id, smri_study_id, match_type, description`).
`match_type` is `resolved` (exactly one T1w candidate that day), `ambiguous`
(multiple candidates, first one picked — worth spot-checking), or
`unresolved` (no T1w scan found on that exact date).

**Output:** `DATA/ADNI/__metadata__/smri_image_ids.txt` — comma-separated
resolved T1 image IDs (regenerated from `smri_resolution.csv` on every run,
so it always reflects every `resolved`/`ambiguous` row found so far).
**Logs:** `logs/adni_download/` (note: underscore, distinct from the
hyphenated `logs/adni-download/` used by the fMRI scripts above)

Once you've built a LONI collection from `smri_image_ids.txt`, download it
and convert it exactly like the fMRI pipeline:

```bash
python download_collection_jar.py --collection <your_smri_collection> \
    --output-dir ../../__smri_dicom_zips_flat__
python convert_dicom_zips.py --zip-dir ../../__smri_dicom_zips_flat__ \
    --output-dir ../../__smri_wholebrain_flat__
```

---

### `extract_partial_zip.py` — Recover corrupt split-ZIPs
Recovers DICOM files from LONI split-ZIPs where Part1 is missing the central directory.
Used on a case-by-case basis when a batch download produces a truncated archive.

```bash
python extract_partial_zip.py <part1.zip> <output_dir>
python extract_partial_zip.py <part1.zip> <output_dir> --convert <nifti_out_dir>
```

---

## Typical full-pipeline run

```bash
# Terminal 1 — download ZIPs
python download_collection_jar.py --collection ADNI_Converters_fMRI

# Terminal 2 — convert to NIfTI in parallel (watch mode)
python convert_dicom_zips.py --watch
```

---

## Credentials

Credentials are stored in `../src/.env` (never committed):

```
ADNI_USERNAME=your@email.com
ADNI_PASSWORD=yourpassword
```

Copy `../.env.example` and fill it in. The file is `chmod 600` on write.

---

## Output directories

| Directory | Contents |
|---|---|
| `DATA/ADNI/__dicom_zips_flat__/` | Raw fMRI DICOM ZIPs from `download_collection*.py` |
| `DATA/ADNI/__fmri_wholebrain_sch200_flat__/` | Converted fMRI NIfTI files |
| `DATA/ADNI/__smri_dicom_zips_flat__/` | Raw T1w/sMRI DICOM ZIPs, once you download the collection built from `smri_image_ids.txt` |
| `DATA/ADNI/__metadata__/` | Metadata CSVs (image IDs, subject info, `smri_resolution.csv`, `smri_image_ids.txt`) |
| `logs/adni-download/<timestamp>/` | Per-run log files (fMRI scripts) |
| `logs/adni_download/` | Per-run log files (`download_adni_smri.py`) |
