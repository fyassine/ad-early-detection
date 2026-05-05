import os
from threading import Thread

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from ..config import DATA_ROOT
from ..metadata_parser import load_metadata
from ..biomarkers import find_subject_nifti_files
from ..cohort_stats import COHORTS, get_cohort_stats, project_visits
from ..services.utils import _safe_round_matrix
from ..services.qc import _ensure_qc_mean

router = APIRouter()


@router.get("/api/cohort/warmup")
async def api_cohort_warmup(
    csv_path: str = Query(..., description="Relative path to metadata CSV"),
    scan_folders: str = Query(..., description="Comma-separated relative folder paths"),
):
    """
    Kick off the cohort_stats UMAP fit in a background thread so that the
    cache is ready by the time the user clicks a patient. After the fit
    settles, pre-computes QC 3D mean volumes for converter visits.
    Returns immediately.
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


@router.get("/api/cohort/stats")
async def api_cohort_stats(
    csv_path: str = Query(..., description="Relative path to metadata CSV"),
    scan_folders: str = Query(..., description="Comma-separated relative folder paths"),
):
    """Per-cohort biomarker statistics + 2-D UMAP scatter. Cached per (csv_path, scan_folders)."""
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


@router.get("/api/cohort/reference")
async def api_cohort_reference(
    csv_path: str = Query(..., description="Relative path to metadata CSV"),
    scan_folders: str = Query(..., description="Comma-separated relative folder paths"),
    cohort: str = Query("healthy", description="Cohort whose mean matrix you want"),
):
    """
    Return the per-cohort mean correlation matrix (e.g. CN baseline mean)
    for the Brain View 'vs CN' deviation mode.
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
