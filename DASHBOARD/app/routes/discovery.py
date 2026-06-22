from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from ..biomarkers import index_nifti_by_subject, index_npz_by_subject
from ..config import DATA_ROOT
from ..scanner import discover_csvs, discover_scan_folders, scan_selected_folders

router = APIRouter()


@router.get("/api/discover")
def api_discover():
    """Discover all available CSVs and scan folders under DATA_ROOT."""
    csvs = discover_csvs(DATA_ROOT)
    folders = discover_scan_folders(DATA_ROOT)
    return JSONResponse({
        "data_root": DATA_ROOT,
        "csvs": csvs,
        "scan_folders": folders,
    })


@router.get("/api/scan")
def api_scan(folders: str = Query(..., description="Comma-separated relative folder paths")):
    """
    Scan selected folders for fMRI/parcellated files.
    Also pre-warms the .npz / .nii.gz subject indices.
    """
    folder_list = [f.strip() for f in folders.split(",") if f.strip()]
    result = scan_selected_folders(DATA_ROOT, folder_list)
    try:
        index_npz_by_subject(DATA_ROOT, folder_list)
        index_nifti_by_subject(DATA_ROOT, folder_list)
    except Exception:
        pass
    return JSONResponse(result)
