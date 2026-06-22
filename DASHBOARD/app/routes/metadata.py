import os

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from ..config import DATA_ROOT
from ..metadata_parser import load_metadata, compute_metadata_metrics
from ..scanner import scan_selected_folders

router = APIRouter()


@router.get("/api/metadata")
def api_metadata(
    csv_path: str = Query(..., description="Relative path to metadata CSV"),
    scan_folders: str = Query(default="", description="Comma-separated relative folder paths"),
    cohort: str = Query(default=None, description="Optional diagnosis cohort to filter by"),
):
    """
    Parse a metadata CSV and compute aggregate metrics.
    Cross-references with scan folders to determine which subjects have files on disk.
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
