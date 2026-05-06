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
from ..services.qc import _ensure_qc_reduce as _ensure_qc_mean

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
    """
    Longitudinal clinical biomarker trajectory + A/T/N classification per visit
    (NIA-AA 2024 criteria). Returns the legacy schema with an extra ``atn``
    array so the frontend can render badges without a second round-trip.
    """
    abs_csv = os.path.join(DATA_ROOT, csv_path)
    if not os.path.exists(abs_csv):
        return JSONResponse({"error": f"CSV not found: {csv_path}"}, status_code=404)
    df = load_metadata(abs_csv)
    result = get_patient_clinical_trajectory(df, subject_id)

    # A/T/N classification per visit
    try:
        from ..services.atn import classify_visits
        result["atn"] = classify_visits(result)
    except Exception as e:
        result["atn"] = []
        result["atn_error"] = str(e)

    return JSONResponse(result)


@router.get("/api/patient/{subject_id}/staging")
async def api_patient_staging(
    subject_id: str,
    csv_path: str = Query(..., description="Relative path to metadata CSV"),
    scan_folders: str = Query(..., description="Comma-separated relative folder paths"),
):
    """
    Cohort-aware staging payload: per-visit EBM stage + brain-age gap +
    per-patient time-shift on the cohort-mean disease curve.
    """
    abs_csv = os.path.join(DATA_ROOT, csv_path)
    if not os.path.exists(abs_csv):
        return JSONResponse({"error": f"CSV not found: {csv_path}"}, status_code=404)
    folder_list = [f.strip() for f in scan_folders.split(",") if f.strip()]
    stats = get_cohort_stats(DATA_ROOT, csv_path, folder_list)

    # ── Per-visit fMRI biomarkers (re-uses cached stream) ────────────────
    visits_payload: list[dict] = []
    fmri_records = find_subject_npz_files(DATA_ROOT, folder_list, subject_id)
    fmri_records_by_visit = {str(r["visit"]).upper(): r for r in fmri_records}

    df = load_metadata(abs_csv)
    clinical = get_patient_clinical_trajectory(df, subject_id)

    from ..services.atn import classify_visits
    from ..services.ebm import stage_visit
    from ..services.brain_age import predict_brain_age
    from ..services.time_shift import estimate_patient_time_shift

    atn_records = classify_visits(clinical)
    atn_by_visit = {str(r["visit"]).upper(): r for r in atn_records}

    # Patient chronological age (use baseline row in the CSV)
    age = None
    if "subject_id" in df.columns and "age" in df.columns:
        sub = df[df["subject_id"] == subject_id]
        if not sub.empty:
            try:
                age = float(sub["age"].dropna().iloc[0])
            except (IndexError, ValueError, TypeError):
                age = None

    # Build merged per-visit dict for EBM + time-shift
    all_visits = sorted(
        set(fmri_records_by_visit.keys()) | set(atn_by_visit.keys()),
        key=lambda v: int(v.lstrip("M")) if v.lstrip("M").isdigit() else 9999,
    )

    ebm = stats.ebm or {}
    brain_age_model = stats.brain_age_model

    for v in all_visits:
        merged: dict = {"visit": v}
        # fMRI biomarkers via cached matrix load
        rec = fmri_records_by_visit.get(v)
        bag = None
        if rec is not None:
            try:
                from ..biomarkers import compute_fmri_biomarkers, load_correlation_matrix
                matrix = load_correlation_matrix(rec["abs_path"])
                is_dmn = matrix.shape[0] <= 50
                bms = compute_fmri_biomarkers(matrix, is_dmn_only=is_dmn)
                for k in ("global_fc", "dmn_fc", "modularity", "density",
                          "pos_fc_ratio", "system_segregation"):
                    if bms.get(k) is not None:
                        merged[k] = bms[k]
                if bms.get("network_fc"):
                    merged["network_fc"] = bms["network_fc"]
                if brain_age_model is not None and age is not None:
                    bag = predict_brain_age(brain_age_model, matrix, age)
                    merged["brain_age"] = bag
            except Exception:
                pass

        # Clinical biomarkers
        atn = atn_by_visit.get(v)
        if atn is not None:
            merged["atn"] = atn
            merged["abeta42"] = atn.get("abeta42")
            merged["p_tau"] = atn.get("p_tau")
            merged["total_tau"] = atn.get("total_tau")
        # Cognition (read directly from clinical trajectory)
        if clinical.get("visits"):
            try:
                idx = clinical["visits"].index(v)
                cog = clinical.get("cognitive", {}) or {}
                for col_key, alias in (("mmse", "mmse_total"), ("cdr", "cdr_global"),
                                       ("pacc5", "pacc5")):
                    arr = cog.get(col_key) or []
                    if idx < len(arr) and arr[idx] is not None:
                        merged[alias] = arr[idx]
            except ValueError:
                pass

        # EBM stage uses the merged dict directly
        if ebm:
            merged["ebm_stage"] = stage_visit(merged, ebm)

        visits_payload.append(merged)

    # Time-shift: needs the per-visit mergedVisits over the patient
    time_shift_payload = {"tau_months": None, "n_obs": 0}
    if stats.time_shift_model is not None:
        try:
            time_shift_payload = estimate_patient_time_shift(
                stats.time_shift_model, visits_payload
            )
        except Exception:
            pass

    return JSONResponse({
        "subject_id": subject_id,
        "age": age,
        "visits": visits_payload,
        "time_shift": time_shift_payload,
        "ebm_sequence": (ebm.get("sequence") or []),
        "brain_age_summary": {
            "available": brain_age_model is not None,
            "cv_mae": getattr(brain_age_model, "cv_mae", None),
            "cv_r2": getattr(brain_age_model, "cv_r2", None),
            "n_train": getattr(brain_age_model, "n_train", 0),
        },
    })


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


@router.get("/api/patient/{subject_id}/conversion-risk")
async def api_patient_conversion_risk(
    subject_id: str,
    csv_path: str = Query(..., description="Relative path to metadata CSV"),
):
    """
    Return 1-year, 3-year and 5-year conversion-to-AD probability for a patient
    by reading off the cohort-level KM curve at the patient's current follow-up
    duration. Returns null values for subjects not at-risk (non-MCI/converter).
    """
    abs_csv = os.path.join(DATA_ROOT, csv_path)
    if not os.path.exists(abs_csv):
        return JSONResponse({"error": f"CSV not found"}, status_code=404)

    df = load_metadata(abs_csv)
    from ..services.survival import time_to_conversion_table, kaplan_meier, _is_apoe4_carrier

    table = time_to_conversion_table(df)
    if table.empty:
        return JSONResponse({"available": False, "reason": "No at-risk subjects in dataset"})

    # Locate this subject
    row = table[table["subject_id"] == subject_id]
    if row.empty:
        return JSONResponse({"available": False, "reason": "Subject not in at-risk group"})

    patient_apoe4 = row.iloc[0]["apoe4"]
    patient_duration = float(row.iloc[0]["duration"])

    # Fit the KM for the patient's APOE4 stratum (or all-at-risk if unknown)
    try:
        from lifelines import KaplanMeierFitter
    except ImportError:
        return JSONResponse({"available": False, "reason": "lifelines not installed"})

    if patient_apoe4 is True:
        sub = table[table["apoe4"] == True]
        label = "APOE4+"
    elif patient_apoe4 is False:
        sub = table[table["apoe4"] == False]
        label = "APOE4−"
    else:
        sub = table
        label = "All at-risk"

    if len(sub) < 3:
        sub = table
        label = "All at-risk"

    kmf = KaplanMeierFitter()
    kmf.fit(durations=sub["duration"].values,
            event_observed=sub["event_observed"].values)

    def _risk_at(months: int) -> float:
        sf = kmf.survival_function_at_times([months]).iloc[0]
        return round(float(1 - sf), 3)

    return JSONResponse({
        "available": True,
        "subject_id": subject_id,
        "stratum": label,
        "n_stratum": int(len(sub)),
        "patient_followup_months": int(patient_duration),
        "risk_1yr": _risk_at(12),
        "risk_3yr": _risk_at(36),
        "risk_5yr": _risk_at(60),
        "note": "Derived from cohort KM curve — not a validated clinical prediction.",
    })
