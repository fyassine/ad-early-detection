import os
from threading import Thread

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from ..config import DATA_ROOT
from ..metadata_parser import load_metadata
from ..biomarkers import find_subject_nifti_files
from ..cohort_stats import COHORTS, get_cohort_stats, project_visits
from ..services.utils import _safe_round_matrix
from ..services.qc import _ensure_qc_reduce as _ensure_qc_mean

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


@router.get("/api/cohort/effect-sizes")
async def api_cohort_effect_sizes(
    csv_path: str = Query(..., description="Relative path to metadata CSV"),
    scan_folders: str = Query(..., description="Comma-separated relative folder paths"),
):
    """Pairwise Cohen's d (Hedges-corrected) + bootstrap 95% CI between cohorts."""
    abs_csv = os.path.join(DATA_ROOT, csv_path)
    if not os.path.exists(abs_csv):
        return JSONResponse({"error": f"CSV not found: {csv_path}"}, status_code=404)
    folder_list = [f.strip() for f in scan_folders.split(",") if f.strip()]
    stats = get_cohort_stats(DATA_ROOT, csv_path, folder_list)

    from ..services.effect_sizes import pairwise_effect_sizes
    out: dict = {}
    for metric in ("global_fc", "dmn_fc", "modularity", "system_segregation",
                   "density", "pos_fc_ratio"):
        cohort_vals = {c: vals.get(metric, []) for c, vals in stats.biomarker_values.items()}
        out[metric] = pairwise_effect_sizes(cohort_vals)
    return JSONResponse({"metrics": out, "cohorts": COHORTS})


@router.get("/api/cohort/survival")
async def api_cohort_survival(
    csv_path: str = Query(..., description="Relative path to metadata CSV"),
    stratify_by: str = Query(default="apoe4", description="'apoe4' or 'none'"),
):
    """Kaplan-Meier survival curve for time-to-conversion, optionally stratified."""
    abs_csv = os.path.join(DATA_ROOT, csv_path)
    if not os.path.exists(abs_csv):
        return JSONResponse({"error": f"CSV not found: {csv_path}"}, status_code=404)

    from ..metadata_parser import load_metadata
    from ..services.survival import kaplan_meier
    df = load_metadata(abs_csv)
    stratum = stratify_by if stratify_by in ("apoe4", "atn", "none") else None
    if stratum == "none":
        stratum = None
    return JSONResponse(kaplan_meier(df, stratify_by=stratum))


@router.get("/api/cohort/ebm")
async def api_cohort_ebm(
    csv_path: str = Query(..., description="Relative path to metadata CSV"),
    scan_folders: str = Query(..., description="Comma-separated relative folder paths"),
):
    """Event-Based Model: ordered biomarker abnormality sequence (CN vs AD)."""
    abs_csv = os.path.join(DATA_ROOT, csv_path)
    if not os.path.exists(abs_csv):
        return JSONResponse({"error": f"CSV not found: {csv_path}"}, status_code=404)
    folder_list = [f.strip() for f in scan_folders.split(",") if f.strip()]
    stats = get_cohort_stats(DATA_ROOT, csv_path, folder_list)
    return JSONResponse(stats.ebm or {"sequence": [], "biomarkers": {}})


@router.get("/api/cohort/brain-age")
async def api_cohort_brain_age(
    csv_path: str = Query(..., description="Relative path to metadata CSV"),
    scan_folders: str = Query(..., description="Comma-separated relative folder paths"),
):
    """Brain-age model summary + cohort BAG distribution."""
    abs_csv = os.path.join(DATA_ROOT, csv_path)
    if not os.path.exists(abs_csv):
        return JSONResponse({"error": f"CSV not found: {csv_path}"}, status_code=404)
    folder_list = [f.strip() for f in scan_folders.split(",") if f.strip()]
    stats = get_cohort_stats(DATA_ROOT, csv_path, folder_list)
    m = stats.brain_age_model
    if m is None:
        return JSONResponse({"available": False})
    return JSONResponse({
        "available": True,
        "n_train": m.n_train,
        "n_features": m.n_features,
        "age_mean": m.age_mean,
        "cv_mae": m.cv_mae,
        "cv_r2": m.cv_r2,
        "bias_slope": m.bias_slope,
        "bias_intercept": m.bias_intercept,
        "cohort_bag_cv": m.cohort_bag,
    })


@router.get("/api/cohort/network-stats")
async def api_cohort_network_stats(
    csv_path: str = Query(..., description="Relative path to metadata CSV"),
    scan_folders: str = Query(..., description="Comma-separated relative folder paths"),
):
    """Per-cohort × per-Schaefer-network FC stats + percentile bands."""
    abs_csv = os.path.join(DATA_ROOT, csv_path)
    if not os.path.exists(abs_csv):
        return JSONResponse({"error": f"CSV not found: {csv_path}"}, status_code=404)
    folder_list = [f.strip() for f in scan_folders.split(",") if f.strip()]
    stats = get_cohort_stats(DATA_ROOT, csv_path, folder_list)
    return JSONResponse({
        "cohorts": COHORTS,
        "network_fc_stats": stats.network_fc_stats,
        "biomarker_percentiles": stats.biomarker_percentiles,
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


@router.get("/api/cohort/missingness")
async def api_cohort_missingness(
    csv_path: str = Query(..., description="Relative path to metadata CSV"),
):
    """
    Missing-data heatmap: returns a per-subject × per-biomarker observed/missing matrix.
    Rows are subjects (grouped by diagnosis), columns are clinical biomarkers + visit count.
    Value 1 = observed at ≥1 visit, 0 = entirely missing.
    """
    abs_csv = os.path.join(DATA_ROOT, csv_path)
    if not os.path.exists(abs_csv):
        return JSONResponse({"error": f"CSV not found: {csv_path}"}, status_code=404)

    import pandas as pd
    import math

    df = load_metadata(abs_csv)
    if "subject_id" not in df.columns:
        return JSONResponse({"error": "No subject_id column"}, status_code=400)

    BIOMARKERS = [
        ("age",        "Age"),
        ("sex",        "Sex"),
        ("apoe",       "APOE"),
        ("mmse_total", "MMSE"),
        ("cdr_sum",    "CDR-SB"),
        ("cdr_global", "CDR global"),
        ("abeta42",    "Aβ42"),
        ("total_tau",  "t-Tau"),
        ("p_tau",      "p-Tau181"),
        ("pacc5",      "PACC-5"),
    ]
    available = [(col, label) for col, label in BIOMARKERS if col in df.columns]
    biomarker_labels = [label for _, label in available] + ["Visits (n)"]

    # Get baseline diagnosis for ordering
    diag_order = {"healthy": 0, "scd": 1, "mci": 2, "converter": 3, "ad": 4}

    rows = []
    for sid, grp in df.groupby("subject_id"):
        diag = ""
        if "diagnosis" in grp.columns:
            diags = grp["diagnosis"].dropna().astype(str).str.lower()
            if len(diags):
                diag = diags.iloc[0]
        n_visits = int(grp["visit"].nunique()) if "visit" in grp.columns else 1
        observed = []
        for col, _ in available:
            has_val = grp[col].dropna()
            has_val = has_val[has_val.astype(str).str.strip().isin(["", "nan"]) == False]
            observed.append(1 if len(has_val) > 0 else 0)
        observed.append(min(n_visits, 10))  # visit count capped at 10 for colour scaling
        rows.append({
            "sid": str(sid),
            "diagnosis": diag,
            "sort_key": diag_order.get(diag, 5),
            "observed": observed,
        })

    rows.sort(key=lambda r: (r["sort_key"], r["sid"]))

    # Build diagnosis colour map
    DIAG_COLORS = {
        "healthy": "#6daa45", "scd": "#4f98a3", "mci": "#e8af34",
        "converter": "#e08040", "ad": "#d163a7",
    }
    subject_ids   = [r["sid"]       for r in rows]
    diagnoses     = [r["diagnosis"] for r in rows]
    diag_colors   = [DIAG_COLORS.get(d, "#7a7976") for d in diagnoses]
    matrix        = [r["observed"]  for r in rows]

    return JSONResponse({
        "subjects":        subject_ids,
        "diagnoses":       diagnoses,
        "diag_colors":     diag_colors,
        "biomarkers":      biomarker_labels,
        "matrix":          matrix,
        "diag_color_map":  DIAG_COLORS,
    })
