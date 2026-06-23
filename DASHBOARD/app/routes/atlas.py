import json

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from ..config import STATIC_DIR

router = APIRouter()


@router.get("/api/atlas/schaefer/coords")
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
        raise HTTPException(status_code=500, detail=f"Failed to read coords: {e}") from e
    return JSONResponse(
        data,
        headers={"Cache-Control": "public, max-age=86400, immutable"},
    )
