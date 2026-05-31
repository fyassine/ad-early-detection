"""
main.py — FastAPI application factory for the fMRI Data Dashboard.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import STATIC_DIR, DATA_ROOT, DASHBOARD_CACHE_ROOT
from .routes import discovery, metadata, cohort, patient, atlas, health, population
from .services import job_manager


def _auto_warmup() -> None:
    """
    On server startup, relaunch precompute jobs for any workspace in
    watched_workspaces.json whose precompute job is not currently running.

    If the disk cache fingerprint still matches (matrices unchanged), the
    precompute script will detect the hit in Stage 1 and skip the heavy
    computation — so this is essentially free on subsequent restarts.
    """
    workspaces = job_manager.load_watched_workspaces(DASHBOARD_CACHE_ROOT)
    if not workspaces:
        return
    print(f"[startup] {len(workspaces)} watched workspace(s); checking precompute jobs…")
    launched = 0
    for ws in workspaces:
        csv_path = ws.get("csv_path", "")
        scan_folders = ws.get("scan_folders", [])
        if not csv_path:
            continue
        if job_manager.is_running(csv_path, scan_folders, DASHBOARD_CACHE_ROOT):
            print(f"[startup]   already running: {csv_path}")
            continue
        job_id, already = job_manager.start_job(
            csv_path=csv_path,
            scan_folders=scan_folders,
            data_root=DATA_ROOT,
            cache_root=DASHBOARD_CACHE_ROOT,
        )
        if not already:
            print(f"[startup]   launched job {job_id} for {csv_path}")
            launched += 1
    if launched:
        print(f"[startup] {launched} precompute job(s) launched")
    else:
        print("[startup] all workspaces already cached or running — no new jobs needed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Kill any precompute that exceeded the wall-clock or stall budget while
    # the server was down. Must run before _auto_warmup() so a zombie isn't
    # treated as "already running".
    try:
        summary = job_manager.sweep_runaway_jobs(DASHBOARD_CACHE_ROOT)
        if summary["killed_runaway"] or summary["killed_stale"] or summary["cleaned_stale_pid"]:
            print(f"[startup] sweep: {summary}")
    except Exception as e:
        print(f"[startup] sweep error (ignored): {e}")

    # Live watchdog: every WATCHDOG_INTERVAL_S, re-run the same checks.
    try:
        job_manager.start_watchdog(DASHBOARD_CACHE_ROOT)
    except Exception as e:
        print(f"[startup] watchdog start error (ignored): {e}")

    # On startup: relaunch precompute jobs for watched workspaces.
    try:
        _auto_warmup()
    except Exception as e:
        # Never block startup on a warmup failure.
        print(f"[startup] auto-warmup error (ignored): {e}")
    yield
    # On shutdown: nothing — precompute jobs are detached and keep running.


app = FastAPI(title="fMRI Data Dashboard", version="2.0.0", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1024)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

for _router in [discovery, metadata, cohort, patient, atlas, health, population]:
    app.include_router(_router.router)

@app.get("/")
async def index():
    """Serve the dashboard frontend (built by Vite into static/dist/)."""
    return FileResponse(str(STATIC_DIR / "dist" / "index.html"))
