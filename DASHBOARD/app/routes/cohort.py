import os

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from ..cohort_stats import COHORTS, get_cohort_stats, is_stats_cached
from ..config import DASHBOARD_CACHE_ROOT, DATA_ROOT
from ..metadata_parser import load_metadata
from ..services.job_manager import (
    cancel_job,
    canonical_job_id,
    get_status,
    list_jobs,
    register_workspace,
    start_job,
)
from ..services.utils import _safe_round_matrix, cache_headers

router = APIRouter()


def _stats_pending_response(csv_path: str, folder_list: list[str]) -> JSONResponse:
    job = get_status(canonical_job_id(csv_path, folder_list), DASHBOARD_CACHE_ROOT)
    status = str(job.get("status") or "").lower()
    stage = str(job.get("stage") or "running").replace("_", " ")
    progress = job.get("progress")
    if status in {"starting", "running"}:
        if isinstance(progress, (int, float)):
            pct = max(0, min(100, int(round(float(progress) * 100.0))))
            note = f"Cohort warmup is still running ({pct}% — {stage})."
        else:
            note = f"Cohort warmup is still running ({stage})."
    else:
        note = "Cohort warmup has not finished for this dataset yet."
    return JSONResponse({
        "available": False,
        "note": note,
        "job": job if status != "unknown" else None,
    })


@router.get("/api/cohort/warmup")
async def api_cohort_warmup(
    csv_path: str = Query(..., description="Relative path to metadata CSV"),
    scan_folders: str = Query(..., description="Comma-separated relative folder paths"),
):
    """
    Launch a detached precompute job for this dataset. The job runs all
    expensive stages (UMAP fit, EBM, brain-age, graph metrics, GELSTM
    predictions, QC volumes) and writes disk caches that survive server
    restarts. Returns the job_id immediately — poll /api/cohort/jobs/{id}
    for progress.

    The subprocess is detached (start_new_session=True) so it keeps running
    even if uvicorn is restarted. The workspace is registered in
    watched_workspaces.json so it auto-restarts on the next server boot.
    """
    abs_csv = os.path.join(DATA_ROOT, csv_path)
    if not os.path.exists(abs_csv):
        return JSONResponse({"error": f"CSV not found: {csv_path}"}, status_code=404)
    folder_list = [f.strip() for f in scan_folders.split(",") if f.strip()]

    # Register so startup auto-warmup picks this workspace up on next boot.
    register_workspace(csv_path, folder_list, DASHBOARD_CACHE_ROOT)

    job_id, already_running = start_job(
        csv_path=csv_path,
        scan_folders=folder_list,
        data_root=DATA_ROOT,
        cache_root=DASHBOARD_CACHE_ROOT,
    )
    status = get_status(job_id, DASHBOARD_CACHE_ROOT)
    return JSONResponse({
        "status": "warming",
        "job_id": job_id,
        "already_running": already_running,
        "job": status,
    })


@router.get("/api/cohort/jobs")
def api_cohort_jobs():
    """List all precompute job statuses."""
    return JSONResponse(list_jobs(DASHBOARD_CACHE_ROOT))


@router.get("/api/cohort/jobs/{job_id}")
def api_cohort_job_status(job_id: str):
    """Get status for a specific precompute job."""
    status = get_status(job_id, DASHBOARD_CACHE_ROOT)
    if status.get("status") == "unknown":
        return JSONResponse(status, status_code=404)
    return JSONResponse(status)


@router.delete("/api/cohort/jobs/{job_id}")
def api_cohort_job_cancel(job_id: str):
    """Send SIGTERM to a running precompute job."""
    sent = cancel_job(job_id, DASHBOARD_CACHE_ROOT)
    return JSONResponse({"job_id": job_id, "signal_sent": sent})


@router.get("/api/cohort/stats")
def api_cohort_stats(
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
    }, headers=cache_headers(stats.fingerprint))


@router.get("/api/cohort/effect-sizes")
def api_cohort_effect_sizes(
    csv_path: str = Query(..., description="Relative path to metadata CSV"),
    scan_folders: str = Query(..., description="Comma-separated relative folder paths"),
):
    """Pairwise Cohen's d (Hedges-corrected) + bootstrap 95% CI between cohorts."""
    abs_csv = os.path.join(DATA_ROOT, csv_path)
    if not os.path.exists(abs_csv):
        return JSONResponse({"error": f"CSV not found: {csv_path}"}, status_code=404)
    folder_list = [f.strip() for f in scan_folders.split(",") if f.strip()]
    if not is_stats_cached(DATA_ROOT, csv_path, folder_list):
        return _stats_pending_response(csv_path, folder_list)
    stats = get_cohort_stats(DATA_ROOT, csv_path, folder_list)

    from ..services.effect_sizes import pairwise_effect_sizes
    out: dict = {}
    for metric in ("global_fc", "dmn_fc", "modularity", "system_segregation",
                   "density", "pos_fc_ratio"):
        cohort_vals = {c: vals.get(metric, []) for c, vals in stats.biomarker_values.items()}
        out[metric] = pairwise_effect_sizes(cohort_vals)
    return JSONResponse({"metrics": out, "cohorts": COHORTS},
                        headers=cache_headers(stats.fingerprint))


@router.get("/api/cohort/survival")
def api_cohort_survival(
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
def api_cohort_ebm(
    csv_path: str = Query(..., description="Relative path to metadata CSV"),
    scan_folders: str = Query(..., description="Comma-separated relative folder paths"),
):
    """Event-Based Model: ordered biomarker abnormality sequence (CN vs AD)."""
    abs_csv = os.path.join(DATA_ROOT, csv_path)
    if not os.path.exists(abs_csv):
        return JSONResponse({"error": f"CSV not found: {csv_path}"}, status_code=404)
    folder_list = [f.strip() for f in scan_folders.split(",") if f.strip()]
    stats = get_cohort_stats(DATA_ROOT, csv_path, folder_list)
    return JSONResponse(stats.ebm or {"sequence": [], "biomarkers": {}},
                        headers=cache_headers(stats.fingerprint))


@router.get("/api/cohort/brain-age")
def api_cohort_brain_age(
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
        return JSONResponse({"available": False}, headers=cache_headers(stats.fingerprint))
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
    }, headers=cache_headers(stats.fingerprint))


@router.get("/api/cohort/network-stats")
def api_cohort_network_stats(
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
    }, headers=cache_headers(stats.fingerprint))


@router.get("/api/cohort/reference")
def api_cohort_reference(
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
def api_cohort_missingness(
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
            has_val = has_val[not has_val.astype(str).str.strip().isin(["", "nan"])]
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


# --------------------------------------------------------------------------- #
# Phase 3 — graph topology, dynamic FC, risk distribution                     #
# --------------------------------------------------------------------------- #

@router.get("/api/cohort/graph-topology")
def api_cohort_graph_topology(
    csv_path: str = Query(..., description="Relative path to metadata CSV"),
    scan_folders: str = Query(..., description="Comma-separated relative folder paths"),
    density: float = Query(0.20, description="Edge density threshold (0–1)"),
    max_subjects: int = Query(40, description="Cap subjects per cohort to bound CPU"),
):
    """
    Per-cohort distribution of graph-theoretic metrics on baseline FC:
    small-worldness (Humphries 2008), modularity Q, clustering, path
    length, global efficiency. Computed on density-thresholded binary
    graphs to control for global FC differences.

    Heavy: ~0.5s per subject. We cap subjects per cohort via ``max_subjects``
    (random sample, seed=42) to keep total compute under ~30s.
    """
    import numpy as np
    abs_csv = os.path.join(DATA_ROOT, csv_path)
    if not os.path.exists(abs_csv):
        return JSONResponse({"error": f"CSV not found: {csv_path}"}, status_code=404)
    folder_list = [f.strip() for f in scan_folders.split(",") if f.strip()]
    if not is_stats_cached(DATA_ROOT, csv_path, folder_list):
        return _stats_pending_response(csv_path, folder_list)
    stats = get_cohort_stats(DATA_ROOT, csv_path, folder_list)

    from ..biomarkers import find_subject_npz_files, load_correlation_matrix
    from ..metadata_parser import _get_baseline, load_metadata
    from ..services.graph_metrics import (
        _HAS_NX,
        load_graph_metrics_cache,
        save_graph_metrics_cache,
        subject_graph_metrics,
    )

    if not _HAS_NX:
        return JSONResponse({
            "available": False,
            "note": "networkx is not installed in the dashboard environment.",
        })

    # Check disk cache first (written by precompute.py / previous on-demand call).
    cached_gm = load_graph_metrics_cache(DASHBOARD_CACHE_ROOT, csv_path, folder_list, density)
    if cached_gm is not None:
        cached_gm["cached"] = True
        return JSONResponse(cached_gm, headers=cache_headers(stats.fingerprint))

    df = load_metadata(abs_csv)
    if "diagnosis" not in df.columns or "subject_id" not in df.columns:
        return JSONResponse({"available": False, "note": "metadata missing diagnosis/subject_id"})

    baseline = _get_baseline(df)
    cohort_subjects: dict[str, list[str]] = {}
    for _, row in baseline.iterrows():
        diag = str(row.get("diagnosis", "")).lower()
        sid = str(row.get("subject_id"))
        if not sid or sid == "nan":
            continue
        cohort_subjects.setdefault(diag, []).append(sid)

    rng = np.random.default_rng(42)
    metrics_by_cohort: dict[str, dict] = {}
    for cohort in COHORTS:
        ids = cohort_subjects.get(cohort, [])
        if not ids:
            metrics_by_cohort[cohort] = {"n": 0, "metrics": {}}
            continue
        if len(ids) > max_subjects:
            sampled = list(rng.choice(ids, size=max_subjects, replace=False))
        else:
            sampled = list(ids)

        buckets: dict[str, list[float]] = {
            "small_worldness": [], "clustering": [], "path_length": [],
            "global_efficiency": [],
        }
        used = 0
        for sid in sampled:
            recs = find_subject_npz_files(DATA_ROOT, folder_list, sid)
            if not recs:
                continue
            try:
                m = load_correlation_matrix(recs[0]["abs_path"])
            except Exception:
                continue
            res = subject_graph_metrics(np.asarray(m), density=density,
                                        compute_hubs=False)
            for k in buckets:
                v = res.get(k)
                if v is not None:
                    buckets[k].append(float(v))
            used += 1

        metric_summary: dict = {}
        for k, vals in buckets.items():
            if not vals:
                metric_summary[k] = None
                continue
            arr = np.asarray(vals, dtype=np.float64)
            metric_summary[k] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=0)) if arr.size > 1 else 0.0,
                "p5":  float(np.quantile(arr, 0.05)),
                "p25": float(np.quantile(arr, 0.25)),
                "p50": float(np.quantile(arr, 0.50)),
                "p75": float(np.quantile(arr, 0.75)),
                "p95": float(np.quantile(arr, 0.95)),
                "n": int(arr.size),
            }
        metrics_by_cohort[cohort] = {"n": used, "n_sampled": len(sampled), "metrics": metric_summary}

    result = {
        "available": True,
        "density": density,
        "max_subjects": max_subjects,
        "cohorts": COHORTS,
        "metrics_by_cohort": metrics_by_cohort,
    }
    # Persist for subsequent requests (including after server restart).
    save_graph_metrics_cache(DASHBOARD_CACHE_ROOT, csv_path, folder_list, density, result)
    return JSONResponse(result, headers=cache_headers(stats.fingerprint))


@router.get("/api/cohort/risk-distribution")
def api_cohort_risk_distribution(
    csv_path: str = Query(..., description="Relative path to metadata CSV"),
    scan_folders: str = Query(..., description="Comma-separated relative folder paths"),
    bins: int = Query(20, description="Histogram bins"),
):
    """
    GELSTM ensemble conversion-probability distribution per cohort.

    Returns ``available=False`` when no checkpoints are deployed. When
    available, batches inference over baseline subjects and bins
    predictions per diagnosis cohort.
    """
    import numpy as np
    abs_csv = os.path.join(DATA_ROOT, csv_path)
    if not os.path.exists(abs_csv):
        return JSONResponse({"error": f"CSV not found: {csv_path}"}, status_code=404)

    from ..services.gelstm import get_gelstm_service
    svc = get_gelstm_service()
    if not svc.is_available():
        return JSONResponse({
            "available": False,
            "note": "GELSTM ensemble not deployed.",
        })

    folder_list = [f.strip() for f in scan_folders.split(",") if f.strip()]
    from ..biomarkers import find_subject_npz_files, load_correlation_matrix
    from ..metadata_parser import load_metadata

    df = load_metadata(abs_csv)
    if "diagnosis" not in df.columns or "subject_id" not in df.columns:
        return JSONResponse({"available": False, "note": "metadata missing diagnosis/subject_id"})

    cohort_probs: dict[str, list[float]] = {c: [] for c in COHORTS}
    for sid, grp in df.groupby(df["subject_id"].astype(str)):
        diag = (grp["diagnosis"].astype(str).str.lower().dropna().head(1).iloc[0]
                if "diagnosis" in grp.columns and not grp["diagnosis"].dropna().empty
                else None)
        if diag not in cohort_probs:
            continue
        recs = find_subject_npz_files(DATA_ROOT, folder_list, sid)
        if not recs:
            continue
        try:
            m = load_correlation_matrix(recs[0]["abs_path"])
        except Exception:
            continue
        sex_val = None
        age_val = None
        first = grp.iloc[0]
        if "sex" in first.index and first.get("sex") is not None:
            s0 = str(first["sex"]).strip().upper()[:1]
            sex_val = 0.0 if s0 == "F" else (1.0 if s0 == "M" else None)
        if "age" in first.index:
            try:
                age_val = float(first["age"])
            except (TypeError, ValueError):
                age_val = None
        pred = svc.predict_subject(sid, [np.asarray(m)], [0.0], sex_val, age_val)
        if pred.get("prob") is not None:
            cohort_probs[diag].append(float(pred["prob"]))

    edges = np.linspace(0, 1, bins + 1)
    histograms: dict[str, dict] = {}
    for cohort, probs in cohort_probs.items():
        if not probs:
            histograms[cohort] = {"counts": [0] * bins, "n": 0}
            continue
        arr = np.asarray(probs, dtype=np.float64)
        counts, _ = np.histogram(arr, bins=edges)
        histograms[cohort] = {
            "counts": counts.astype(int).tolist(),
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
        }

    return JSONResponse({
        "available": True,
        "model_version": svc._ensemble.model_version if svc._ensemble else "",
        "bin_edges": edges.tolist(),
        "histograms": histograms,
        "cohorts": COHORTS,
    })


@router.get("/api/cohort/network-disruption")
def api_cohort_network_disruption(
    csv_path: str = Query(..., description="Relative path to metadata CSV"),
    scan_folders: str = Query(..., description="Comma-separated relative folder paths"),
):
    """
    Per-Schaefer-7-network effect sizes between cohort pairs — same data
    as ``/api/population/network-atlas`` but exposed under the cohort
    namespace for the Cohort tier's per-network panel.
    """
    abs_csv = os.path.join(DATA_ROOT, csv_path)
    if not os.path.exists(abs_csv):
        return JSONResponse({"error": f"CSV not found: {csv_path}"}, status_code=404)
    folder_list = [f.strip() for f in scan_folders.split(",") if f.strip()]
    if not is_stats_cached(DATA_ROOT, csv_path, folder_list):
        return _stats_pending_response(csv_path, folder_list)
    stats = get_cohort_stats(DATA_ROOT, csv_path, folder_list)
    from ..services.population import network_disruption_atlas
    return JSONResponse(network_disruption_atlas(stats),
                        headers=cache_headers(stats.fingerprint))


@router.get("/api/cohort/dfc-states")
def api_cohort_dfc_states(
    csv_path: str = Query(..., description="Relative path to metadata CSV"),
    scan_folders: str = Query(..., description="Comma-separated relative folder paths"),
    k: int = Query(default=4, description="Number of dFC states"),
    window: int = Query(default=30, description="Sliding-window length (TRs)"),
    step: int = Query(default=3, description="Sliding-window step (TRs)"),
):
    """
    Dynamic FC state distributions per cohort.

    Reads the cached payload produced by `precompute._stage_dfc()`. If the
    warmup hasn't produced a cache yet, returns `available=False` with a
    hint to run/await the warmup job.
    """
    import hashlib
    import json as _json

    abs_csv = os.path.join(DATA_ROOT, csv_path)
    if not os.path.exists(abs_csv):
        return JSONResponse({"error": f"CSV not found: {csv_path}"}, status_code=404)

    folder_list = [f.strip() for f in scan_folders.split(",") if f.strip()]
    h = hashlib.sha1()
    h.update(csv_path.encode())
    for f in sorted(folder_list):
        h.update(b"\x00")
        h.update(f.encode())
    cache_key = h.hexdigest()[:20]
    cache_path = DASHBOARD_CACHE_ROOT / "dfc" / f"dfc_{cache_key}_k{k}_w{window}_s{step}.json"

    if cache_path.exists():
        try:
            return JSONResponse(_json.loads(cache_path.read_text()))
        except Exception as e:
            return JSONResponse({"available": False, "note": f"dFC cache unreadable: {e}"})

    # Surface the active warmup job (if any) so the frontend can render a progress bar.
    job = get_status(canonical_job_id(csv_path, folder_list), DASHBOARD_CACHE_ROOT)
    job_status = str(job.get("status") or "").lower()
    return JSONResponse({
        "available": False,
        "note": (
            "Dynamic FC is not computed yet for this dataset. Trigger the cohort "
            "warmup (or wait for it to finish) — Stage 5 parcellates BOLD .nii.gz "
            "and fits state distributions automatically."
        ),
        "job": job if job_status not in ("unknown", "") else None,
    })
