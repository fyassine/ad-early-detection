# Inspect / Diagnostic Scripts (Archive)

> **These scripts are not part of the active download pipeline.**
> They were written during development to investigate specific LONI IDA portal
> behaviour and debug the download flow. They are kept here for reference in
> case the portal changes and similar issues arise again.

---

## Script inventory

| Script | What it investigated |
|---|---|
| `inspect_collection_dom.py` | DOM structure of the ADNI collection table (rows, scroll containers, AJAX responses) |
| `inspect_collection_page.py` | Visible `cell11_N` rows and pagination in the "Not Downloaded" view |
| `inspect_checkbox_selection.py` | Hidden `<input name="id">` tags and `CHECK_BOX_MANAGER` state when a row is checked |
| `inspect_not_downloaded_pages.py` | Whether the "Not Downloaded" first page of rows is stable or cached across full reloads |
| `inspect_stuck_item.py` | Why `#simple-download-link` sometimes never populates (polls for 5 minutes) |
| `debug_select_probe.py` | Checkbox selection flow for `ADNI_Converters_fMRI` — duplicate IDs, 1-CLICK button state |
| `probe_nextpage.py` | Confirms pagination works via `SelectHandler.nextPage()` — page 0 rows ≠ page 1 rows |
| `probe_notdownloaded.py` | Checks whether jar downloads register server-side as "Downloaded" |
| `diag_jar_fail.py` | Diagnoses why the IDA jar sometimes exits 0 but produces no ZIP |
| `test_adni_login_download.py` | Full step-by-step integration test: login → Advanced Search → add to collection → download |

---

## Running a script

All scripts import from `../download/download_collection.py`. The `sys.path` is
set automatically — just run from anywhere:

```bash
cd DATA/ADNI/src/inspect
python inspect_collection_page.py
```

Credentials are read from `../../.env` (the same `.env` used by the production scripts).
