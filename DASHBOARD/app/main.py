"""
main.py — FastAPI application for the fMRI Data Dashboard.

Dataset-agnostic: discovers CSVs and scan folders under a mounted DATA directory.
The user selects a metadata CSV and scan folders, and the dashboard computes metrics.
Includes patient longitudinal trajectory endpoint for fMRI biomarkers.
"""

import json
import math
import os
from pathlib import Path
from threading import Thread

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from .scanner import discover_csvs, discover_scan_folders, scan_selected_folders
from .metadata_parser import load_metadata, compute_metadata_metrics, get_patient_clinical_trajectory
from .biomarkers import (
    find_subject_nifti_files,
    find_subject_npz_files,
    get_subject_trajectory,
    index_nifti_by_subject,
    index_npz_by_subject,
    load_correlation_matrix,
    SCHAEFER_200_DMN_INDICES,
)
from .cohort_stats import COHORTS, get_cohort_stats, project_visits

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

    Also pre-warms the .npz / .nii.gz subject indices so the first patient
    open doesn't have to walk the folders again.
    """
    folder_list = [f.strip() for f in folders.split(",") if f.strip()]
    result = scan_selected_folders(DATA_ROOT, folder_list)
    # Warm the per-subject indices in the background — cheap dict insert if
    # already cached, otherwise walks once and caches.
    try:
        index_npz_by_subject(DATA_ROOT, folder_list)
        index_nifti_by_subject(DATA_ROOT, folder_list)
    except Exception:
        pass
    return JSONResponse(result)


@app.get("/api/cohort/warmup")
async def api_cohort_warmup(
    csv_path: str = Query(..., description="Relative path to metadata CSV"),
    scan_folders: str = Query(..., description="Comma-separated relative folder paths"),
):
    """
    Kick off the cohort_stats UMAP fit in a background thread so that the
    cache is ready by the time the user clicks a patient. Returns
    immediately; subsequent /api/cohort/stats calls block until the fit
    finishes (or return cached data instantly).
    """
    abs_csv = os.path.join(DATA_ROOT, csv_path)
    if not os.path.exists(abs_csv):
        return JSONResponse({"error": f"CSV not found: {csv_path}"}, status_code=404)
    folder_list = [f.strip() for f in scan_folders.split(",") if f.strip()]

    def _fit():
        try:
            get_cohort_stats(DATA_ROOT, csv_path, folder_list)
        except Exception:
            pass

    Thread(target=_fit, daemon=True).start()
    return JSONResponse({"status": "warming"})


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


# --------------------------------------------------------------------------- #
# Cohort-level reference data (normative bands + manifold)                    #
# --------------------------------------------------------------------------- #

@app.get("/api/cohort/stats")
async def api_cohort_stats(
    csv_path: str = Query(..., description="Relative path to metadata CSV"),
    scan_folders: str = Query(..., description="Comma-separated relative folder paths"),
):
    """
    Per-cohort biomarker statistics for normative bands + 2-D UMAP scatter
    of all baseline subjects. Cached per (csv_path, scan_folders).
    """
    abs_csv = os.path.join(DATA_ROOT, csv_path)
    if not os.path.exists(abs_csv):
        return JSONResponse({"error": f"CSV not found: {csv_path}"}, status_code=404)

    folder_list = [f.strip() for f in scan_folders.split(",") if f.strip()]
    stats = get_cohort_stats(DATA_ROOT, csv_path, folder_list)

    return JSONResponse({
        "cohorts": COHORTS,
        "biomarker_stats": stats.biomarker_stats,
        "manifold": {
            "points": stats.points,
            "centroids": stats.centroids,
            "conversion_axis": stats.conversion_axis,
            "n_rois": stats.n_rois,
        },
    })


@app.get("/api/cohort/reference")
async def api_cohort_reference(
    csv_path: str = Query(..., description="Relative path to metadata CSV"),
    scan_folders: str = Query(..., description="Comma-separated relative folder paths"),
    cohort: str = Query("healthy", description="Cohort whose mean matrix you want"),
):
    """
    Return the per-cohort *mean* correlation matrix (e.g. CN baseline mean)
    so the frontend can compute deviation maps for the Brain View
    "vs CN" comparison mode.
    """
    abs_csv = os.path.join(DATA_ROOT, csv_path)
    if not os.path.exists(abs_csv):
        return JSONResponse({"error": f"CSV not found: {csv_path}"}, status_code=404)

    folder_list = [f.strip() for f in scan_folders.split(",") if f.strip()]
    stats = get_cohort_stats(DATA_ROOT, csv_path, folder_list)
    matrix = stats.cohort_means.get(cohort.strip().lower())
    if matrix is None:
        return JSONResponse(
            {"error": f"No baseline subjects with .npz found for cohort '{cohort}'."},
            status_code=404,
        )
    return JSONResponse({
        "cohort": cohort,
        "n_rois": int(matrix.shape[0]),
        "n_subjects": stats.biomarker_stats.get(cohort, {}).get("global_fc", {}).get("n", 0),
        "matrix": _safe_round_matrix(matrix),
    })


@app.get("/api/patient/{subject_id}/manifold")
async def api_patient_manifold(
    subject_id: str,
    csv_path: str = Query(..., description="Relative path to metadata CSV"),
    scan_folders: str = Query(..., description="Comma-separated relative folder paths"),
):
    """
    Project a patient's longitudinal correlation matrices into the cached
    baseline UMAP. Returns one (x, y, conversion_score) per visit.
    """
    abs_csv = os.path.join(DATA_ROOT, csv_path)
    if not os.path.exists(abs_csv):
        return JSONResponse({"error": f"CSV not found: {csv_path}"}, status_code=404)

    folder_list = [f.strip() for f in scan_folders.split(",") if f.strip()]
    stats = get_cohort_stats(DATA_ROOT, csv_path, folder_list)

    records = find_subject_npz_files(DATA_ROOT, folder_list, subject_id)
    matrices: list = []
    visits: list = []
    files: list = []
    for rec in records:
        try:
            m = load_correlation_matrix(rec["abs_path"])
        except Exception:
            m = None
        matrices.append(m)
        visits.append(rec["visit"])
        files.append(rec["rel_path"])

    projections = project_visits(stats, matrices)
    trajectory = [
        {"visit": visits[i], "file": files[i], **projections[i]}
        for i in range(len(visits))
    ]

    return JSONResponse({
        "subject_id": subject_id,
        "trajectory": trajectory,
        "centroids": stats.centroids,
        "conversion_axis": stats.conversion_axis,
        "n_rois": stats.n_rois,
    })


# --------------------------------------------------------------------------- #
# Per-visit raw correlation matrix (for the Connectivity heatmap)             #
# --------------------------------------------------------------------------- #

def _safe_round_matrix(m: np.ndarray, decimals: int = 4) -> list:
    """JSON-safe nested list with NaN/inf clipped to 0."""
    arr = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.round(arr.astype(np.float64), decimals)
    return arr.tolist()


@app.get("/api/patient/{subject_id}/matrix")
async def api_patient_matrix(
    subject_id: str,
    scan_folders: str = Query(..., description="Comma-separated relative folder paths"),
    visit: str = Query(default=None, description="Visit code, e.g. 'M0'. Omit for baseline."),
):
    """
    Return the raw correlation matrix for one of a patient's visits.
    Used by the Connectivity heatmap and the Brain View edge selector.
    """
    folder_list = [f.strip() for f in scan_folders.split(",") if f.strip()]
    records = find_subject_npz_files(DATA_ROOT, folder_list, subject_id)
    if not records:
        return JSONResponse({"error": "No .npz files found for subject"}, status_code=404)

    target = None
    if visit:
        v = visit.strip().upper()
        for rec in records:
            if str(rec.get("visit", "")).upper() == v:
                target = rec
                break
    if target is None:
        target = records[0]  # fall back to baseline / earliest

    try:
        matrix = load_correlation_matrix(target["abs_path"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load matrix: {e}")

    n = matrix.shape[0]
    is_dmn_only = n <= 50
    dmn_indices = (
        list(range(n)) if is_dmn_only
        else [i for i in SCHAEFER_200_DMN_INDICES if i < n]
    )

    return JSONResponse({
        "subject_id": subject_id,
        "visit": target["visit"],
        "file": target["rel_path"],
        "n_rois": int(n),
        "matrix": _safe_round_matrix(matrix),
        "dmn_indices": dmn_indices,
    })


# --------------------------------------------------------------------------- #
# Per-visit raw NIfTI (for the QC viewer)                                     #
# --------------------------------------------------------------------------- #

def _safe_under_root(abs_path: str) -> bool:
    """Guard against directory traversal — abs_path must live under DATA_ROOT."""
    try:
        root = os.path.realpath(DATA_ROOT)
        target = os.path.realpath(abs_path)
        return target == root or target.startswith(root + os.sep)
    except Exception:
        return False


@app.get("/api/patient/{subject_id}/scan")
async def api_patient_scan(
    subject_id: str,
    scan_folders: str = Query(..., description="Comma-separated relative folder paths"),
    visit: str = Query(default=None, description="Visit code, e.g. 'M0'."),
):
    """
    Stream a patient's .nii.gz volume for the given visit. Used by NiiVue.
    """
    folder_list = [f.strip() for f in scan_folders.split(",") if f.strip()]
    records = find_subject_nifti_files(DATA_ROOT, folder_list, subject_id)
    if not records:
        raise HTTPException(status_code=404, detail="No .nii.gz files found for subject")

    target = None
    if visit:
        v = visit.strip().upper()
        for rec in records:
            if str(rec.get("visit", "")).upper() == v:
                target = rec
                break
    if target is None:
        target = records[0]

    abs_path = target["abs_path"]
    if not _safe_under_root(abs_path) or not os.path.isfile(abs_path):
        raise HTTPException(status_code=404, detail="File not accessible")

    media = "application/gzip" if abs_path.endswith(".gz") else "application/octet-stream"
    return FileResponse(
        abs_path,
        media_type=media,
        filename=os.path.basename(abs_path),
    )


@app.get("/api/patient/{subject_id}/scans")
async def api_patient_scans(
    subject_id: str,
    scan_folders: str = Query(..., description="Comma-separated relative folder paths"),
):
    """List a patient's available .nii.gz volumes (visit + filename)."""
    folder_list = [f.strip() for f in scan_folders.split(",") if f.strip()]
    records = find_subject_nifti_files(DATA_ROOT, folder_list, subject_id)
    return JSONResponse({
        "subject_id": subject_id,
        "scans": [
            {"visit": r["visit"], "filename": r["filename"], "file": r["rel_path"]}
            for r in records
        ],
    })


# --------------------------------------------------------------------------- #
# Schaefer atlas coordinates (for the Brain View glass brain)                 #
# --------------------------------------------------------------------------- #

@app.get("/api/atlas/schaefer/coords")
async def api_schaefer_coords(n_parcels: int = Query(default=200)):
    """
    Return MNI centroids + network labels for the Schaefer parcellation.
    Reads a static JSON shipped at static/data/schaefer_{n_parcels}_coords.json.
    """
    coord_file = STATIC_DIR / "data" / f"schaefer_{n_parcels}_coords.json"
    if not coord_file.exists():
        return JSONResponse(
            {
                "error": f"Schaefer {n_parcels}-parcel coordinates not generated yet.",
                "hint": "Run app/generate_schaefer_coords.py once with the parcellation NIfTI.",
            },
            status_code=404,
        )
    try:
        with coord_file.open("r") as f:
            data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read coords: {e}")
    return JSONResponse(data)


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    data_exists = os.path.isdir(DATA_ROOT)
    return JSONResponse({
        "status": "ok",
        "data_root": DATA_ROOT,
        "data_accessible": data_exists,
    })
