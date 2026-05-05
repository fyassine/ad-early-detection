import json
import os
import asyncio

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from ..config import DATA_ROOT
from ..metadata_parser import load_metadata, get_patient_clinical_trajectory
from ..biomarkers import (
    find_subject_nifti_files,
    find_subject_npz_files,
    get_subject_trajectory_stream,
    load_correlation_matrix,
    SCHAEFER_200_DMN_INDICES,
)
from ..cohort_stats import get_cohort_stats, project_visits
from ..services.utils import _safe_round_matrix, _safe_under_root
from ..services.qc import _ensure_qc_mean

router = APIRouter()


@router.get("/api/patient/{subject_id}/trajectory")
async def api_patient_trajectory(
    subject_id: str,
    request: Request,
    scan_folders: str = Query(..., description="Comma-separated relative folder paths"),
    prioritize_visit: str | None = Query(default=None, description="Optional visit code to process first"),
):
    """
    Stream longitudinal fMRI biomarker trajectory as NDJSON.
    Computes Global FC, DMN FC, Modularity per session.
    """
    folder_list = [f.strip() for f in scan_folders.split(",") if f.strip()]

    async def generate():
        for chunk in get_subject_trajectory_stream(
            DATA_ROOT, folder_list, subject_id, prioritize_visit=prioritize_visit
        ):
            if await request.is_disconnected():
                break
            yield json.dumps(chunk) + "\n"
            await asyncio.sleep(0)

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@router.get("/api/patient/{subject_id}/clinical")
async def api_patient_clinical(
    subject_id: str,
    csv_path: str = Query(..., description="Relative path to metadata CSV"),
):
    """Get longitudinal clinical biomarker trajectory (MMSE, CDR, PACC5, Abeta42, Tau, pTau)."""
    abs_csv = os.path.join(DATA_ROOT, csv_path)
    if not os.path.exists(abs_csv):
        return JSONResponse({"error": f"CSV not found: {csv_path}"}, status_code=404)
    df = load_metadata(abs_csv)
    result = get_patient_clinical_trajectory(df, subject_id)
    return JSONResponse(result)


@router.get("/api/patient/{subject_id}/manifold")
async def api_patient_manifold(
    subject_id: str,
    csv_path: str = Query(..., description="Relative path to metadata CSV"),
    scan_folders: str = Query(..., description="Comma-separated relative folder paths"),
):
    """Project a patient's longitudinal correlation matrices into the cached baseline UMAP."""
    abs_csv = os.path.join(DATA_ROOT, csv_path)
    if not os.path.exists(abs_csv):
        return JSONResponse({"error": f"CSV not found: {csv_path}"}, status_code=404)

    folder_list = [f.strip() for f in scan_folders.split(",") if f.strip()]
    stats = get_cohort_stats(DATA_ROOT, csv_path, folder_list)

    records = find_subject_npz_files(DATA_ROOT, folder_list, subject_id)
    visits, files, matrices = [], [], []
    for rec in records:
        visits.append(rec["visit"])
        files.append(rec["rel_path"])
        matrices.append(None)

    # Prefer precomputed coords from the co-fit UMAP (places converter visits
    # inside the manifold rather than at the boundary via transform()).
    coords_table = stats.patient_visit_coords.get(subject_id, {}) or {}
    trajectory = []
    missing_indices = []
    for i, visit in enumerate(visits):
        cached = coords_table.get(visit)
        if cached and cached.get("x") is not None:
            trajectory.append({
                "visit": visit, "file": files[i],
                "x": cached.get("x"), "y": cached.get("y"),
                "conversion_score": cached.get("conversion_score"),
            })
        else:
            trajectory.append({
                "visit": visit, "file": files[i],
                "x": None, "y": None, "conversion_score": None,
            })
            missing_indices.append(i)

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


@router.get("/api/patient/{subject_id}/matrix")
async def api_patient_matrix(
    subject_id: str,
    scan_folders: str = Query(..., description="Comma-separated relative folder paths"),
    visit: str = Query(default=None, description="Visit code, e.g. 'M0'. Omit for baseline."),
):
    """Return the raw correlation matrix for one of a patient's visits."""
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
        target = records[0]

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


@router.get("/api/patient/{subject_id}/scan")
async def api_patient_scan(
    subject_id: str,
    scan_folders: str = Query(..., description="Comma-separated relative folder paths"),
    visit: str = Query(default=None, description="Visit code, e.g. 'M0'."),
    reduce: str = Query(
        default=None,
        description="'mean' to receive a cached 3D temporal mean instead of the full 4D volume.",
    ),
):
    """
    Stream a patient's .nii.gz volume for the given visit. Used by NiiVue.
    When reduce=mean and the source is 4D, serves a cached 3D temporal mean (~3 MB vs ~67 MB).
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
            print(f"[qc-mean] reduce failed for {abs_path}: {e}")

    media = "application/gzip" if abs_path.endswith(".gz") else "application/octet-stream"
    etag = f'"{int(os.path.getmtime(abs_path))}-{os.path.getsize(abs_path)}"'
    # Keep FileResponse — Starlette serves Range requests, which NiiVue uses.
    return FileResponse(
        abs_path,
        media_type=media,
        filename=os.path.basename(abs_path),
        headers={
            "Cache-Control": "public, max-age=86400, immutable",
            "ETag": etag,
        },
    )


@router.get("/api/patient/{subject_id}/scans")
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
