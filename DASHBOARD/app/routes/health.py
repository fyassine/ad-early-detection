import os

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..config import DATA_ROOT

router = APIRouter()


@router.get("/api/health")
async def health():
    """Health check endpoint."""
    data_exists = os.path.isdir(DATA_ROOT)
    return JSONResponse({
        "status": "ok",
        "data_root": DATA_ROOT,
        "data_accessible": data_exists,
    })
