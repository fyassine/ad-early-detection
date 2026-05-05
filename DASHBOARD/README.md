# fMRI Dashboard — Quick Start Guide

## Launching the Dashboard

### Build the frontend (one time, or after editing JS/CSS)

The dashboard frontend is a Vite project under `frontend/`. Build it before
starting uvicorn — the FastAPI server serves the bundled output from
`app/static/dist/`.

```bash
cd /mnt/e/fyassine/ad-early-detection/DASHBOARD/frontend
npm install
npm run build
```

### On the Remote Server

```bash
cd /mnt/e/fyassine/ad-early-detection/DASHBOARD

DATA_ROOT=/mnt/e/fyassine/ad-early-detection/DATA \
  /usr/bin/python3.10 -m uvicorn app.main:app --host 0.0.0.0 --port 8050
```

The server will print:
```
INFO: Uvicorn running on http://0.0.0.0:8050 (Press CTRL+C to quit)
```

---

## Accessing from Your Local Machine (SSH Port Forwarding)

Since the server has no public web access, you access it through an **SSH tunnel**.

### Step 1 — Open the tunnel (run this on YOUR local machine, not the server)

```bash
ssh -L 8050:localhost:8050 wunderlich@138.245.113.6
```

Keep this terminal open while using the dashboard.

> **What this does:** Forwards port `8050` on your local machine to port `8050` on the remote host through SSH. Traffic is encrypted.

### Step 2 — Open the dashboard

Open your browser and go to:

```
http://localhost:8050
```

### Step 3 — Use the dashboard

1. **Select a Metadata CSV** from the dropdown (e.g. `DELCODE / __v3__ / metadata / cohorts.csv`)
2. **Check scan folders** — choose `.npz` or `.nii.gz` folders (not both)
3. Click **Analyze**
4. Click any **diagnosis bar** to cross-filter the entire dashboard
5. Click any **patient row** to open their longitudinal trajectory

---

## Running with Docker (optional)

```bash
cd /mnt/e/fyassine/ad-early-detection/DASHBOARD

docker-compose up --build
```

Then follow Steps 1–3 above. The `docker-compose.yml` mounts `../DATA` as a read-only volume.

---

## Stopping the Server

Press `Ctrl+C` in the terminal running uvicorn.

---

## Notes

| Item | Detail |
|------|--------|
| Data volume | Mounted read-only — nothing is written to your data |
| Scan types | `.npz` (parcellated matrices) or `.nii.gz` (preprocessed volumes) — not mixed |
| Longitudinal | Patients with >1 visit are highlighted green in the table |
| Trajectory | Requires `.npz` scan folder — computes Global FC, DMN FC, Modularity per visit |
| Port | Default `8050` — change in `docker-compose.yml` or uvicorn command if needed |

---

## Patient view — tabs

Click any patient row to open the modal. Five tabs share a single
*selected visit* — clicking M36 anywhere updates every tab.

| Tab | What it shows |
|---|---|
| Overview | Longitudinal Global FC / DMN FC / Modularity / cognitive / CSF charts. Each fMRI chart now carries a **normative band** (mean ± 1 σ of MCI non-converters at baseline) plus a deviation strip and an fMRI-only **conversion score** (0 = MCI-NC like, 1 = AD like). |
| Manifold | 2-D UMAP scatter of every baseline subject (CN / SCD / MCI / Converter / AD), patient visits projected into the same fixed space and connected chronologically with an arrow on the final segment. Click a visit dot → all tabs sync. |
| Connectivity | Visit-aware ROI × ROI heatmap of the raw correlation matrix, with a "group by Schaefer network" toggle that pulls the DMN block into the top-left corner. |
| QC Viewer | Embedded NiiVue volume viewer (axial / coronal / sagittal). Visible only when the selected scan folders contain `.nii.gz`. |
| Brain View | Glass-brain SVG (axial + sagittal) showing the strongest edges of the selected visit. Threshold slider + per-network filters. Requires the Schaefer-coords JSON below. |

## New API endpoints

```
GET /api/cohort/stats?csv_path=…&scan_folders=…
GET /api/patient/{id}/manifold?csv_path=…&scan_folders=…
GET /api/patient/{id}/matrix?scan_folders=…&visit=M0
GET /api/patient/{id}/scan?scan_folders=…&visit=M0     # streams .nii.gz
GET /api/patient/{id}/scans?scan_folders=…             # list of available volumes
GET /api/atlas/schaefer/coords?n_parcels=200
```

Cohort statistics are cached in process memory keyed by `(csv_path, sorted scan_folders)` — first request fits UMAP (~few seconds for hundreds of subjects), subsequent requests are instant. Restart the server to invalidate the cache.

## Generating the Schaefer atlas coordinates (one time)

The Brain View tab needs ROI MNI centroids. Run this once:

```bash
python -m app.generate_schaefer_coords \
  --parcellation /path/to/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz \
  --labels       /path/to/Schaefer2018_200Parcels_7Networks_order.txt \
  --n-parcels    200
```

Output lands at `app/static/data/schaefer_200_coords.json`. Both reference files ship with the official Schaefer atlas release (CBIG GitHub) and are already cached locally by nilearn at `~/nilearn_data/schaefer_2018/` if you've ever called `nilearn.datasets.fetch_atlas_schaefer_2018()`.
