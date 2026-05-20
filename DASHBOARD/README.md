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

Press `Ctrl+C` in the terminal running uvicorn — that's the clean path when uvicorn is running in the foreground. See the **Restart workflow** section below for the full procedure when uvicorn is detached / backgrounded / hung.

---

## Restart workflow (kill old → rebuild → run)

Use this whenever you change Python code, JS/CSS, or environment variables.
The dashboard caches things aggressively, both server-side (Python module imports, `_load_failed` flags) and client-side (localStorage discovery cache), so a clean restart is the most reliable way to pick up changes.

### Step 1 — Find any running uvicorn processes

```bash
# Show all uvicorn workers + their command lines + ports
ps -fC python3 -fC python3.10 2>/dev/null | grep -E "uvicorn|app\.main" | grep -v grep
# Or specifically find what's bound to port 8050
ss -tlnp 2>/dev/null | grep 8050 || lsof -i :8050 2>/dev/null
```

You should see one (or more) lines like:
```
wunderl+ 1808555 ... /usr/bin/python3.10 -m uvicorn app.main:app --host 0.0.0.0 --port 8050
```

### Step 2 — Kill the old server cleanly

```bash
# Polite SIGTERM (lets in-flight requests finish + flushes job-status JSON on disk)
kill 1808555

# If it's been more than ~10 s and the process is still alive, force it
kill -9 1808555

# Nuke everything matching "uvicorn app.main" in one shot (use with care)
pkill -f "uvicorn.*app.main"
```

Confirm nothing is left holding the port:
```bash
ss -tlnp 2>/dev/null | grep 8050   # should print nothing
```

Detached `bash -c` wrappers (e.g. PID 515427-style copilot launchers) sometimes stay even after their child uvicorn dies — kill them by their PIDs too if `ps -ef | grep uvicorn` still shows them.

### Step 3 — Clear stale Python caches (if you've been editing code)

The interpreter reads `.pyc` bytecode in `__pycache__/` first; stale files from a different Python version (e.g. 3.10 vs 3.12) can mask edits to `.py` sources. After major refactors:

```bash
find /mnt/e/fyassine/ad-early-detection/DASHBOARD/app -name "__pycache__" -type d -exec rm -rf {} +  2>/dev/null
find /mnt/e/fyassine/ad-early-detection/DASHBOARD/app -name "*.pyc" -delete 2>/dev/null
```

### Step 3b — Clear GELSTM predictions cache (if GELSTM code or checkpoints changed)

If you changed GELSTM inference code or fixed `cond_vec`-related errors, delete the cached predictions so Stage 3 recomputes with the updated model:

```bash
rm -f /mnt/e/fyassine/ad-early-detection/DASHBOARD/.cache/gelstm/predictions_*.pkl
```

### Step 4 — Rebuild the frontend (only if you changed JS/CSS/HTML)

```bash
cd /mnt/e/fyassine/ad-early-detection/DASHBOARD/frontend
npm run build           # emits to ../app/static/dist
```

Backend-only changes don't need this — but if the dashboard UI still looks unchanged after a server restart, the build is what you missed.

> Tip — during JS development, `npm run dev` (Vite HMR) is faster than rebuilding every save, but for "production" sessions where the user only hits the FastAPI server, the build artefact is what's served.

### Step 5 — Start the server with the correct environment

The project-root venv (`/mnt/e/fyassine/ad-early-detection/.venv`) has `torch`, `torch_geometric`, `nilearn`, and everything the dashboard imports. `DASHBOARD/.venv` does **not** — picking the wrong interpreter is the most common cause of `ModuleNotFoundError: No module named 'torch'` at startup.

```bash
cd /mnt/e/fyassine/ad-early-detection/DASHBOARD

DATA_ROOT=/mnt/e/fyassine/ad-early-detection/DATA \
DASHBOARD_CACHE_ROOT=$PWD/.cache \
/mnt/e/fyassine/ad-early-detection/.venv/bin/python \
  -m uvicorn app.main:app --host 0.0.0.0 --port 8050 --reload
```

`--reload` watches `app/` for `.py` changes and auto-restarts; drop it for production / long-running sessions where you don't want spurious reloads.

To run **in the background** (so the terminal stays usable):
```bash
nohup env DATA_ROOT=/mnt/e/fyassine/ad-early-detection/DATA \
          DASHBOARD_CACHE_ROOT=$PWD/.cache \
  /mnt/e/fyassine/ad-early-detection/.venv/bin/python \
    -m uvicorn app.main:app --host 0.0.0.0 --port 8050 \
  > /tmp/dashboard.log 2>&1 &
echo "started pid $! → tail -f /tmp/dashboard.log"
```

### Step 6 — Verify the server is alive and configured correctly

```bash
# Healthcheck — should print the absolute DATA path + a non-zero CSV count
curl -s http://localhost:8050/api/discover \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('data_root'), len(d['csvs']),'CSVs')"
```

A correct response looks like:
```
/mnt/e/fyassine/ad-early-detection/DATA 47 CSVs
```

If you get `/data 0 CSVs`, `DATA_ROOT` wasn't set — kill the process and restart with the env var.

### Step 7 — Clear the browser-side cache (only if the page still looks wrong)

Our frontend caches `/api/discover` in `localStorage` for 5 minutes (so reloads don't show "Discovering data…" every time). If a hard reload (Cmd+Shift+R / Ctrl+Shift+R) still shows stale data:

```js
// In the browser DevTools console:
localStorage.removeItem('fmri_discovery_cache');
location.reload();
```

Or click the **Retry** button that appears on the connection-error banner — it does the same thing.

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

---

## Environment variables

| Var | Purpose | Default |
|-----|---------|---------|
| `DATA_ROOT` | Root of CSV + scan files. **Required for non-Docker runs** — the default `/data` will not have your data. | `/data` |
| `DASHBOARD_CACHE_ROOT` | Writable cache for warmup outputs, GELSTM predictions, dFC, timeseries, job status. | `DASHBOARD/.cache` |
| `CACHE_ROOT` | Used by `cohort_stats.py` for the UMAP/EBM/brain-age pickle. Defaults to `${DATA_ROOT}/.cache`. | `${DATA_ROOT}/.cache` |
| `CLASSIFIER_ROOT` | Where the GELSTM service finds model code. | `<repo>/CLASSIFIER` |
| `GELSTM_CHECKPOINT_DIR` | Where the GELSTM service finds `best_model_fold*.pth`, `gaae_encoder.pth`, `model_card.json`. | `${CLASSIFIER_ROOT}/model/GELSTM/checkpoints` |

## Running outside Docker (development)

The project-root venv (`<repo>/.venv`) has `torch`, `nilearn`, `torch_geometric`. **`DASHBOARD/.venv` does NOT** — use the project venv:

```bash
cd /mnt/e/fyassine/ad-early-detection/DASHBOARD
DATA_ROOT=/mnt/e/fyassine/ad-early-detection/DATA \
DASHBOARD_CACHE_ROOT=$PWD/.cache \
/mnt/e/fyassine/ad-early-detection/.venv/bin/python \
  -m uvicorn app.main:app --host 0.0.0.0 --port 8050 --reload
```

For Docker, `docker-compose up` does the right thing — the bundled `docker-compose.yml` already sets `DATA_ROOT=/data` and mounts `../DATA:/data:ro`.

## GELSTM ensemble deployment

The GELSTM service in `app/services/gelstm.py` expects this layout:

```
CLASSIFIER/model/GELSTM/checkpoints/
├── best_model_fold1.pth
├── best_model_fold2.pth
├── best_model_fold3.pth
├── best_model_fold4.pth
├── best_model_fold5.pth
├── gaae_encoder.pth          # the pretrained GAAE encoder (FiLM-conditioned)
└── model_card.json           # arch hyperparams (must match the trained checkpoints)
```

`model_card.json` example (must match the GELSTM training notebook's hyperparameters exactly):
```json
{
  "arch": {
    "in_features": 200, "gaae_hidden": 200, "gaae_latent": 64,
    "gaae_heads": 2, "gaae_cond_dim": 2, "gaae_dropout": 0.3,
    "lstm_hidden": 128, "lstm_layers": 2, "lstm_dropout": 0.3,
    "use_time_delta": true, "classifier_hidden": 64
  },
  "norm": {}
}
```

> Without `model_card.json`, the service uses hard-coded defaults (`gaae_heads=4`, `gaae_dropout=0.2`) which will **silently mismatch** notebook-trained checkpoints (heads=2, dropout=0.3) — `load_state_dict()` will fail with 86+ shape mismatches and the dashboard will say *"GELSTM ensemble not deployed"*.

The training notebook `CLASSIFIER/notebooks/GELSTM_DELCODE_WHOLE_BRAIN.ipynb` ends with a deployment cell that copies all the required files into this directory — re-run that cell after every training run.

## Cohort warmup pipeline

When you click **Analyze Data**, the backend kicks off a precompute subprocess (`app/precompute.py`) with 5 stages:

| Stage | What | Approx. wall time | Output |
|-------|------|-------------------|--------|
| 1. CohortStats | UMAP + EBM + brain-age + time-shift model | 5–10 min first run | `${DASHBOARD_CACHE_ROOT}/cohort_stats/*.pkl` |
| 2. Graph metrics | Small-worldness / clustering / path length per cohort | Hours per cohort if cold | `${DASHBOARD_CACHE_ROOT}/graph_metrics/*.json` |
| 3. GELSTM predictions | Per-subject 5-fold inference | ~5 s × N subjects | `${DASHBOARD_CACHE_ROOT}/gelstm/predictions_<ver>.pkl` |
| 4. QC volumes | Temporal-std for converter subjects | ~10 s per scan | next to the `.nii.gz` files |
| 5. **Dynamic FC** | Parcellate BOLD with Schaefer-200 → sliding-window FC → k-means states + dwell times | 10–30 min first run | `${DASHBOARD_CACHE_ROOT}/dfc/*.json` + per-scan `${DASHBOARD_CACHE_ROOT}/timeseries/*.npz` |

Stage 5 needs the raw **BOLD `.nii.gz`** folder selected (e.g. `__v1__/fmri`). Pre-parcellated correlation-matrix folders (e.g. `__v3__/matrices`) won't work — they're flat N×N, dFC needs T×N time-series.

Progress for stages is exposed via `GET /api/cohort/jobs/{job_id}` with fields `{status, stage, progress (0..1), error, started_at, finished_at}`. The Dynamic FC panel polls this and renders an in-place progress bar while Stage 5 runs.

## Cache layout

```
${DASHBOARD_CACHE_ROOT}/
├── cohort_stats/            # Stage 1 — pickled CohortStats (UMAP/EBM/…)
├── graph_metrics/           # Stage 2 — per-cohort topology JSON
├── gelstm/                  # Stage 3 — per-subject conversion probabilities
├── dfc/                     # Stage 5 — per-state dwell-time JSON per (csv, folders, k, w, s)
├── timeseries/              # Stage 5 — cached per-scan (T, 200) NPZ arrays
└── jobs/                    # warmup job status + PID + log files
```

Invalidate a stage by deleting its directory (or just the relevant key file). The warmup will rebuild it on next trigger. **Never** keep negative-result caches around — dFC explicitly does NOT cache `available=false` payloads to avoid this pitfall (see `_stage_dfc` in `app/precompute.py`).

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| "Discovering data…" stuck forever | Server unreachable / wrong `DATA_ROOT` (returns 0 CSVs) | Hard-reload (auto-times-out after 10 s in current frontend). Then restart server with the correct `DATA_ROOT`. |
| Kaplan-Meier "Insufficient data" | Either (a) wrong `DATA_ROOT` and the survival route 404'd, or (b) `_attach_atn_stage` exception. Server logs will show. | Restart with correct env. If ATN-only, check `app/services/survival.py:_attach_atn_stage` — the `classify_atn()` call must pass `abeta42`, `p_tau`, `total_tau` as kwargs, not a raw row. |
| dFC says "No BOLD .nii.gz files could be parcellated" | Either the fmri folder wasn't selected, or a stale negative cache | Make sure `__v1__/fmri` is in the scan folder picker. Then `rm DASHBOARD/.cache/dfc/dfc_*.json` and trigger warmup. |
| GELSTM panel empty despite ensemble loading | `cond_vec` kwarg mismatch (`encode()` doesn't accept it — use `condition_latent()` after encoding). 838+ `fold inference failed` lines in logs. | Pull the latest `app/services/gelstm.py:predict_subject` — it calls `condition_latent()` after `encode()`. |
| GELSTM "ensemble not deployed" | Missing `model_card.json`, or arch mismatch between card and checkpoints | Re-run the notebook's deployment cell, which writes `model_card.json` alongside the .pth files. |
| dFC progress bar shows 0% then jumps to 100% | Progress only updates every 5 subjects | Wait — incremental updates kick in after the first 5 parcellations. |
| `import torch` fails when starting the server | Using `DASHBOARD/.venv` instead of the project root venv | Switch to `<repo>/.venv/bin/python`. |
