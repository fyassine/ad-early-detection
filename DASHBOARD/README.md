# fMRI Dashboard — Quick Start Guide

## Launching the Dashboard

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
