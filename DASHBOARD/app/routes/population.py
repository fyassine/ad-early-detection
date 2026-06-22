"""
population.py — Population-tier API routes.

Provides four endpoints that back the Population top-tab:

  GET /api/population/summary       multi-cohort aggregation (totals, demographics)
  GET /api/population/epidemiology  Fang 2025 lifetime-risk reference table
  GET /api/population/network-atlas per-Schaefer-7-network effect-size matrix
  GET /api/population/model-card    GELSTM deployed-model performance (Phase 2)
"""

from __future__ import annotations

import os

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from ..cohort_stats import get_cohort_stats, is_stats_cached
from ..config import DATA_ROOT
from ..metadata_parser import load_metadata
from ..services.gelstm import get_gelstm_service
from ..services.population import (
    cohort_demographic_summary,
    fang_epidemiology_table,
    network_disruption_atlas,
)
from ..services.utils import cache_headers

router = APIRouter()


def _parse_folders(scan_folders: str) -> list[str]:
    return [f.strip() for f in (scan_folders or "").split(",") if f.strip()]


@router.get("/api/population/summary")
def api_population_summary(
    csv_path: str = Query(..., description="Relative path to metadata CSV"),
    scan_folders: str = Query("", description="Comma-separated relative folder paths"),
):
    """Cross-cohort aggregation: totals, demographics, conversion rate, sites."""
    abs_csv = os.path.join(DATA_ROOT, csv_path)
    if not os.path.exists(abs_csv):
        return JSONResponse({"error": f"CSV not found: {csv_path}"}, status_code=404)

    df = load_metadata(abs_csv)
    return JSONResponse(cohort_demographic_summary(df))


@router.get("/api/population/epidemiology")
def api_population_epidemiology():
    """Fang et al. 2025 lifetime-risk reference table (static)."""
    return JSONResponse(fang_epidemiology_table())


@router.get("/api/population/network-atlas")
def api_population_network_atlas(
    csv_path: str = Query(..., description="Relative path to metadata CSV"),
    scan_folders: str = Query(..., description="Comma-separated relative folder paths"),
):
    """Per-Schaefer-7-network effect-size matrix between cohort pairs."""
    abs_csv = os.path.join(DATA_ROOT, csv_path)
    if not os.path.exists(abs_csv):
        return JSONResponse({"error": f"CSV not found: {csv_path}"}, status_code=404)

    folder_list = _parse_folders(scan_folders)
    if not folder_list:
        return JSONResponse(
            {"error": "scan_folders is required for the network atlas"},
            status_code=400,
        )
    if not is_stats_cached(DATA_ROOT, csv_path, folder_list):
        return JSONResponse({
            "available": False,
            "note": "Cohort warmup is still running for this dataset.",
        })
    stats = get_cohort_stats(DATA_ROOT, csv_path, folder_list)
    return JSONResponse(network_disruption_atlas(stats),
                        headers=cache_headers(stats.fingerprint))


@router.get("/api/population/model-card")
def api_population_model_card():
    """
    GELSTM deployed-model performance card. Returns ``available: false``
    when no checkpoints are deployed at $GELSTM_CHECKPOINT_DIR — the
    frontend renders a placeholder in that case.
    """
    return JSONResponse(get_gelstm_service().model_card_metrics())
