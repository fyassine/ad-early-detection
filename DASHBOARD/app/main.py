"""
main.py — FastAPI application for the fMRI Data Dashboard.

Dataset-agnostic: discovers CSVs and scan folders under a mounted DATA directory.
The user selects a metadata CSV and scan folders, and the dashboard computes metrics.
Includes patient longitudinal trajectory endpoint for fMRI biomarkers.
"""

import json
import math
import os
import asyncio
from pathlib import Path
from threading import Thread

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from .scanner import discover_csvs, discover_scan_folders, scan_selected_folders
from .metadata_parser import load_metadata, compute_metadata_metrics, get_patient_clinical_trajectory
from .biomarkers import (
    find_subject_nifti_files,
    find_subject_npz_files,
    get_subject_trajectory_stream,
    index_nifti_by_subject,
    index_npz_by_subject,
    load_correlation_matrix,
    SCHAEFER_200_DMN_INDICES,
)
from .cohort_stats import COHORTS, get_cohort_stats, project_visits

# Data root — mounted as /data in Docker, or use env var, or fallback
DATA_ROOT = os.environ.get("DATA_ROOT", "/data")

app = FastAPI(title="fMRI Data Dashboard", version="2.0.0")
app.add_middleware(GZipMiddleware, minimum_size=1024)

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
    cache is ready by the time the user clicks a patient. After the fit
    settles, the same thread also pre-computes the QC 3D mean volumes for
    every converter visit it can find — so the QC tab is instant on first
    click. Returns immediately.
    """
    abs_csv = os.path.join(DATA_ROOT, csv_path)
    if not os.path.exists(abs_csv):
        return JSONResponse({"error": f"CSV not found: {csv_path}"}, status_code=404)
    folder_list = [f.strip() for f in scan_folders.split(",") if f.strip()]

    def _fit():
        try:
            get_cohort_stats(DATA_ROOT, csv_path, folder_list)
        except Exception as e:
            print(f"[warmup] cohort fit failed: {e}")
            return
        # ── QC mean pre-compute for every converter visit ────────────────
        # This is best-effort and runs serially after the cohort fit so it
        # doesn't compete with the user's first patient open.
        try:
            df = load_metadata(abs_csv)
            if "diagnosis" not in df.columns or "subject_id" not in df.columns:
                return
            converters = (
                df[df["diagnosis"].astype(str).str.lower() == "converter"]
                ["subject_id"].dropna().astype(str).unique().tolist()
            )
            n_done = 0
            for sid in converters:
                for rec in find_subject_nifti_files(DATA_ROOT, folder_list, sid):
                    try:
                        _ensure_qc_mean(rec["abs_path"])
                        n_done += 1
                    except Exception:
                        continue
            print(f"[warmup] pre-computed {n_done} QC mean volumes for {len(converters)} converters")
        except Exception as e:
            print(f"[warmup] QC pre-compute failed: {e}")

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
    request: Request,
    scan_folders: str = Query(..., description="Comma-separated relative folder paths"),
    prioritize_visit: str | None = Query(default=None, description="Optional visit code to process first"),
):
    """
    Get longitudinal fMRI biomarker trajectory for a specific patient.
    Finds all their .npz files across visits, computes Global FC, DMN FC,
    Modularity per session, returns time-ordered metrics.
    Streams progress line-by-line as NDJSON.
    """
    folder_list = [f.strip() for f in scan_folders.split(",") if f.strip()]

    async def generate():
        for chunk in get_subject_trajectory_stream(
            DATA_ROOT,
            folder_list,
            subject_id,
            prioritize_visit=prioritize_visit,
        ):
            if await request.is_disconnected():
                break
            yield json.dumps(chunk) + "\n"
            await asyncio.sleep(0)

    return StreamingResponse(generate(), media_type="application/x-ndjson")


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
    visits: list = []
    files: list = []
    matrices: list = []
    for rec in records:
        visits.append(rec["visit"])
        files.append(rec["rel_path"])
        matrices.append(None)  # populated lazily for the transform() fallback

    # Prefer the precomputed coords from the co-fit UMAP (Round 4 §C). This
    # places converter visits *inside* the manifold instead of at the
    # boundary as `mapper.transform()` does.
    coords_table = stats.patient_visit_coords.get(subject_id, {}) or {}
    trajectory = []
    missing_indices = []
    for i, visit in enumerate(visits):
        cached = coords_table.get(visit)
        if cached and cached.get("x") is not None:
            trajectory.append({
                "visit": visit,
                "file": files[i],
                "x": cached.get("x"),
                "y": cached.get("y"),
                "conversion_score": cached.get("conversion_score"),
            })
        else:
            trajectory.append({"visit": visit, "file": files[i], "x": None, "y": None, "conversion_score": None})
            missing_indices.append(i)

    # Fall back to mapper.transform() for any visit not in the precomputed
    # table (e.g. a non-converter or a brand-new visit that wasn't part of
    # the cohort fit).
    if missing_indices:
        for i in missing_indices:
            try:
                matrices[i] = load_correlation_matrix(records[i]["abs_path"])
            except Exception:
                matrices[i] = None
        projections = project_visits(stats, matrices)
        for i in missing_indices:
            trajectory[i].update({
                "x": projections[i].get("x"),
                "y": projections[i].get("y"),
                "conversion_score": projections[i].get("conversion_score"),
            })

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


# Disk cache for QC mean volumes — indexed by sha1 of the source file's
# absolute path so a moved/renamed source invalidates automatically.
_QC_MEAN_DIR = Path(os.environ.get(
    "DASHBOARD_CACHE_DIR",
    os.path.join(DATA_ROOT, "_dashboard_cache"),
)) / "qc_mean"


def _qc_mean_path(src_abs: str) -> Path:
    import hashlib
    h = hashlib.sha1(src_abs.encode("utf-8")).hexdigest()[:16]
    base = os.path.basename(src_abs)
    if base.endswith(".nii.gz"):
        base = base[:-7]
    elif base.endswith(".nii"):
        base = base[:-4]
    return _QC_MEAN_DIR / f"{base}_{h}_mean.nii.gz"


def _ensure_qc_mean(src_abs: str) -> str:
    """
    Return the path to a cached 3D mean image for ``src_abs``. If the source
    is already 3D, returns it unchanged. If it's 4D, computes
    ``data.mean(axis=3)``, writes to disk, and returns the cached path.
    Subsequent calls reuse the cache.
    """
    cached = _qc_mean_path(src_abs)
    if cached.exists():
        return str(cached)

    try:
        import nibabel as nib  # nibabel is optional at runtime
    except ImportError:
        return src_abs  # fall back to streaming the original

    img = nib.load(src_abs)
    if img.ndim < 4 or img.shape[-1] <= 1:
        return src_abs  # already 3D, nothing to reduce

    # mean over the last (time) axis. Use float32 to halve disk size.
    data = np.asarray(img.dataobj).astype(np.float32, copy=False)
    mean = data.mean(axis=-1).astype(np.float32, copy=False)
    out = nib.Nifti1Image(mean, img.affine, img.header)
    # Strip the temporal slope so the saved header is consistent
    out.header.set_data_dtype(np.float32)
    cached.parent.mkdir(parents=True, exist_ok=True)
    # Use gzip level 9 — saves ~30% over the default 6 at negligible CPU cost
    # since each file is written exactly once.
    import gzip as _gzip
    raw_bytes = out.to_bytes()
    with _gzip.open(str(cached), "wb", compresslevel=9) as f:
        f.write(raw_bytes)
    return str(cached)


@app.get("/api/patient/{subject_id}/scan")
async def api_patient_scan(
    subject_id: str,
    scan_folders: str = Query(..., description="Comma-separated relative folder paths"),
    visit: str = Query(default=None, description="Visit code, e.g. 'M0'."),
    reduce: str = Query(
        default=None,
        description="Optional: 'mean' to receive a cached 3D temporal mean instead of the full 4D volume — much faster for QC.",
    ),
):
    """
    Stream a patient's .nii.gz volume for the given visit. Used by NiiVue.

    When ``reduce=mean`` and the source is 4D, computes (and caches) a 3D
    temporal mean image. For typical resting-state fMRI this drops the
    download from ~67 MB to ~3 MB and visibly cuts the QC viewer load
    time from 5–15 s to <1 s.
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

    if reduce and reduce.strip().lower() == "mean":
        try:
            abs_path = _ensure_qc_mean(abs_path)
        except Exception as e:
            # If reduction fails, fall back to streaming the original.
            print(f"[qc-mean] reduce failed for {abs_path}: {e}")

    media = "application/gzip" if abs_path.endswith(".gz") else "application/octet-stream"
    # Keep FileResponse — Starlette serves Range requests, which NiiVue uses to
    # stream large .nii.gz volumes. Don't swap to StreamingResponse.
    etag = f'"{int(os.path.getmtime(abs_path))}-{os.path.getsize(abs_path)}"'
    return FileResponse(
        abs_path,
        media_type=media,
        filename=os.path.basename(abs_path),
        headers={
            "Cache-Control": "public, max-age=86400, immutable",
            "ETag": etag,
        },
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
    return JSONResponse(
        data,
        headers={"Cache-Control": "public, max-age=86400, immutable"},
    )


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    data_exists = os.path.isdir(DATA_ROOT)
    return JSONResponse({
        "status": "ok",
        "data_root": DATA_ROOT,
        "data_accessible": data_exists,
    })
