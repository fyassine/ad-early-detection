"""
main.py — FastAPI application for the fMRI Data Dashboard.

Dataset-agnostic: discovers CSVs and scan folders under a mounted DATA directory.
The user selects a metadata CSV and scan folders, and the dashboard computes metrics.
Includes patient longitudinal trajectory endpoint for fMRI biomarkers.
"""

import os
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from .scanner import discover_csvs, discover_scan_folders, scan_selected_folders
from .metadata_parser import load_metadata, compute_metadata_metrics, get_patient_clinical_trajectory
from .biomarkers import get_subject_trajectory

# Data root — mounted as /data in Docker, or use env var, or fallback
DATA_ROOT = os.environ.get("DATA_ROOT", "/data")

app = FastAPI(title="fMRI Data Dashboard", version="2.0.0")

# Serve static files (frontend)
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    """Serve the dashboard frontend."""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/discover")
async def api_discover():
    """
    Discover all available CSVs and scan folders under DATA_ROOT.
    Returns structured info with format details for the frontend.
    """
    csvs = discover_csvs(DATA_ROOT)
    folders = discover_scan_folders(DATA_ROOT)

    return JSONResponse({
        "data_root": DATA_ROOT,
        "csvs": csvs,
        "scan_folders": folders,
    })


@app.get("/api/scan")
async def api_scan(folders: str = Query(..., description="Comma-separated relative folder paths")):
    """
    Scan selected folders for fMRI/parcellated files.
    Returns scan counts, subject counts, file type detected, and format info.
    """
    folder_list = [f.strip() for f in folders.split(",") if f.strip()]
    result = scan_selected_folders(DATA_ROOT, folder_list)
    return JSONResponse(result)


@app.get("/api/metadata")
async def api_metadata(
    csv_path: str = Query(..., description="Relative path to metadata CSV"),
    scan_folders: str = Query(default="", description="Comma-separated relative folder paths"),
    cohort: str = Query(default=None, description="Optional diagnosis cohort to filter by"),
):
    """
    Parse a metadata CSV and compute aggregate metrics.
    Cross-references with scan folders to determine which subjects have actual files on disk.
    If 'cohort' is provided, filters the entire analysis to just that diagnosis.
    """
    abs_csv = os.path.join(DATA_ROOT, csv_path)
    if not os.path.exists(abs_csv):
        return JSONResponse({"error": f"CSV not found: {csv_path}"}, status_code=404)

    df = load_metadata(abs_csv, cohort=cohort)

    scan_subjects = None
    scan_subject_counts = None
    if scan_folders:
        folder_list = [f.strip() for f in scan_folders.split(",") if f.strip()]
        if folder_list:
            scan_result = scan_selected_folders(DATA_ROOT, folder_list)
            scan_subjects = list(scan_result["subject_scan_counts"].keys())
            scan_subject_counts = scan_result["subject_scan_counts"]

    metrics = compute_metadata_metrics(df, scan_subjects, scan_subject_counts)
    return JSONResponse(metrics)


@app.get("/api/patient/{subject_id}/trajectory")
async def api_patient_trajectory(
    subject_id: str,
    scan_folders: str = Query(..., description="Comma-separated relative folder paths"),
):
    """
    Get longitudinal fMRI biomarker trajectory for a specific patient.
    Finds all their .npz files across visits, computes Global FC, DMN FC,
    Modularity per session, returns time-ordered metrics.
    """
    folder_list = [f.strip() for f in scan_folders.split(",") if f.strip()]
    result = get_subject_trajectory(DATA_ROOT, folder_list, subject_id)
    return JSONResponse(result)


@app.get("/api/patient/{subject_id}/clinical")
async def api_patient_clinical(
    subject_id: str,
    csv_path: str = Query(..., description="Relative path to metadata CSV"),
):
    """
    Get longitudinal clinical biomarker trajectory for a specific patient.
    Extracts MMSE, CDR, PACC5, Abeta42, Tau, pTau across available visits.
    """
    abs_csv = os.path.join(DATA_ROOT, csv_path)
    if not os.path.exists(abs_csv):
        return JSONResponse({"error": f"CSV not found: {csv_path}"}, status_code=404)

    df = load_metadata(abs_csv)
    result = get_patient_clinical_trajectory(df, subject_id)
    return JSONResponse(result)


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    data_exists = os.path.isdir(DATA_ROOT)
    return JSONResponse({
        "status": "ok",
        "data_root": DATA_ROOT,
        "data_accessible": data_exists,
    })
