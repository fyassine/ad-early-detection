# ADNI Source Scripts

```
src/
├── download/          ← Active pipeline — scripts you run
│   ├── README.md      ← How to use the pipeline (start here)
│   ├── download_collection.py       # Core: batch download via browser
│   ├── download_collection_jar.py   # Alt: per-image download via IDA jar
│   ├── download_adni_fmri.py        # Earlier self-contained downloader
│   ├── convert_dicom_zips.py        # Post-download DICOM → NIfTI
│   └── extract_partial_zip.py       # Utility: recover from split ZIPs
│
├── inspect/           ← Archived diagnostic scripts (not for regular use)
│   └── README.md      ← What each script investigated
│
├── ida_downloader/    ← IDA Downloader jar (required by download_collection_jar.py)
│   ├── IdaDownloader_15May2026.jar
│   └── run_ida_downloader.py
│
├── .env               ← Your LONI credentials (gitignored, never commit)
└── .env.example       ← Template — copy to .env and fill in
```

## Quick start

```bash
cd DATA/ADNI/src/download
cp ../.env.example ../.env && nano ../.env   # add your LONI username/password

# Download DICOM ZIPs
python download_collection_jar.py --collection ADNI_Converters_fMRI

# Convert to NIfTI (run in a second terminal)
python convert_dicom_zips.py --watch
```

See `download/README.md` for the full pipeline guide.
