"""
main.py — FastAPI application factory for the fMRI Data Dashboard.
"""

from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .config import STATIC_DIR
from .routes import discovery, metadata, cohort, patient, atlas, health

app = FastAPI(title="fMRI Data Dashboard", version="2.0.0")
app.add_middleware(GZipMiddleware, minimum_size=1024)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

for _router in [discovery, metadata, cohort, patient, atlas, health]:
    app.include_router(_router.router)


@app.get("/")
async def index():
    """Serve the dashboard frontend (built by Vite into static/dist/)."""
    return FileResponse(str(STATIC_DIR / "dist" / "index.html"))
