"""
cohort_stats.py — Cohort-level reference data for the dashboard.

Computes baseline biomarker statistics (for normative bands) and a 2-D UMAP
embedding of all baseline subjects (for the manifold view), once per
``(csv_path, sorted scan_folders)`` key. Results are cached in process memory.

Exposes a single high-level function:

    get_cohort_stats(data_root, csv_path, scan_folders) -> CohortStats

Cohort labels expected in the CSV's ``diagnosis`` column:
    healthy, scd, mci (= MCI non-converters), converter (= MCI-C baseline), ad

Only converters are longitudinal in this dataset; everyone else is baseline-only
and acts as a fixed anchor cloud in the manifold.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional

import numpy as np
import pandas as pd

from .biomarkers import (
    compute_fmri_biomarkers,
    find_subject_npz_files,
    index_npz_by_subject,
    load_correlation_matrix,
)
from .metadata_parser import load_metadata, _get_baseline


# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #

# Cohorts we expose in the manifold + normative bands (in display order).
COHORTS = ["healthy", "scd", "mci", "converter", "ad"]

# UMAP hyper-parameters — tuned for ~hundreds of subjects with thousands of edges.
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_RANDOM_STATE = 42

# Biomarker fields tracked in normative bands.
BIOMARKER_KEYS = ["global_fc", "dmn_fc", "modularity", "density", "pos_fc_ratio"]


# --------------------------------------------------------------------------- #
# Dataclass                                                                   #
# --------------------------------------------------------------------------- #

@dataclass
class CohortStats:
    """All cohort-level reference data the frontend needs."""
    # Per-cohort biomarker stats: {cohort: {metric: {mean, std, n}}}
    biomarker_stats: dict
    # Manifold scatter
    points: list  # [{x, y, cohort, subject_id}]
    centroids: dict  # {cohort: {x, y, n}}
    conversion_axis: dict  # {origin: {x,y}, direction: {x,y}}
    # Per-cohort mean correlation matrix (n_rois × n_rois). Used by the Brain
    # View "vs CN" comparison mode. Stored as float32 to halve the cache cost.
    cohort_means: dict = field(default_factory=dict)  # {cohort: np.ndarray}
    # Internal — kept for transform() of new visits
    umap_mapper: object = field(default=None, repr=False)
    edge_dim: int = 0  # length of the upper-triangle feature vector
    n_rois: int = 0


# --------------------------------------------------------------------------- #
# Cache                                                                       #
# --------------------------------------------------------------------------- #

_CACHE: dict[tuple, CohortStats] = {}
_LOCK = Lock()


def _cache_key(csv_path: str, scan_folders: list[str]) -> tuple:
    return (csv_path, tuple(sorted(scan_folders)))


def clear_cache() -> None:
    """Invalidate every cached CohortStats — useful for tests / data refresh."""
    with _LOCK:
        _CACHE.clear()


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _safe_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if not math.isfinite(f) else f


def _upper_triangle(matrix: np.ndarray) -> np.ndarray:
    """Return the upper-triangle (k=1) of a square matrix as a 1-D vector."""
    n = matrix.shape[0]
    idx = np.triu_indices(n, k=1)
    return matrix[idx]


def _select_baseline_subjects(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pick one row per subject at baseline for the cohorts we care about.
    Falls back to first-row-per-subject if no explicit baseline marker exists.
    """
    if "diagnosis" not in df.columns or "subject_id" not in df.columns:
        return df.iloc[0:0]

    baseline = _get_baseline(df)
    baseline = baseline.copy()
    baseline["diagnosis"] = baseline["diagnosis"].astype(str).str.strip().str.lower()
    baseline = baseline[baseline["diagnosis"].isin(COHORTS)]
    baseline = baseline.drop_duplicates(subset="subject_id", keep="first")
    return baseline


def _pick_baseline_npz(records: list[dict]) -> Optional[dict]:
    """
    From a subject's .npz records, pick the baseline visit.
    Prefers M0; falls back to the chronologically earliest available visit.
    """
    if not records:
        return None
    # find_subject_npz_files already sorts chronologically, so records[0] is the earliest.
    for r in records:
        if str(r.get("visit", "")).upper() == "M0":
            return r
    return records[0]


# --------------------------------------------------------------------------- #
# Stage 1 — collect baseline FC vectors                                       #
# --------------------------------------------------------------------------- #

def _collect_baseline_fc_vectors(
    data_root: str,
    df: pd.DataFrame,
    scan_folders: list[str],
) -> tuple[np.ndarray, list[str], list[str], int]:
    """
    For every baseline subject in ``df``, find their baseline .npz, load the
    correlation matrix, and stack the upper-triangle as a feature row.

    Returns ``(features (N, n_edges), cohorts, subject_ids, n_rois)``.
    Subjects without a baseline .npz are silently skipped.
    The matrix size is locked to the first subject — others with mismatching
    shape are dropped (atlas mismatch).
    """
    baseline = _select_baseline_subjects(df)
    if baseline.empty:
        return np.empty((0, 0)), [], [], 0

    feature_rows: list[np.ndarray] = []
    cohorts: list[str] = []
    subject_ids: list[str] = []
    n_rois: int = 0

    # Build the file index once — avoids walking the folders once per subject.
    npz_index = index_npz_by_subject(data_root, scan_folders)

    for _, row in baseline.iterrows():
        sid = str(row.get("subject_id", "")).strip()
        cohort = str(row.get("diagnosis", "")).strip().lower()
        if not sid or cohort not in COHORTS:
            continue

        records = npz_index.get(sid, [])
        rec = _pick_baseline_npz(records)
        if rec is None:
            continue

        try:
            matrix = load_correlation_matrix(rec["abs_path"])
        except Exception:
            continue
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            continue

        if n_rois == 0:
            n_rois = matrix.shape[0]
        elif matrix.shape[0] != n_rois:
            # Atlas mismatch — keep the dominant size, drop the rest
            continue

        vec = _upper_triangle(matrix).astype(np.float32, copy=False)
        # Replace NaN/inf with 0 so UMAP doesn't choke
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        feature_rows.append(vec)
        cohorts.append(cohort)
        subject_ids.append(sid)

    if not feature_rows:
        return np.empty((0, 0)), [], [], 0

    return np.vstack(feature_rows), cohorts, subject_ids, n_rois


# --------------------------------------------------------------------------- #
# Stage 2 — biomarker statistics for normative bands                          #
# --------------------------------------------------------------------------- #

def _cohort_mean_matrices(
    features: np.ndarray,
    cohorts: list[str],
    n_rois: int,
) -> dict:
    """
    Reconstruct the per-cohort *mean* correlation matrix from upper-triangle
    feature rows. Returns ``{cohort: ndarray (n_rois, n_rois) float32}``.
    """
    if features.size == 0 or n_rois == 0:
        return {}
    iu = np.triu_indices(n_rois, k=1)
    out: dict = {}
    for cohort in COHORTS:
        idx = [i for i, c in enumerate(cohorts) if c == cohort]
        if not idx:
            continue
        mean_upper = features[idx].mean(axis=0)
        m = np.zeros((n_rois, n_rois), dtype=np.float32)
        m[iu] = mean_upper
        m = m + m.T
        np.fill_diagonal(m, 1.0)  # self-correlation = 1
        out[cohort] = m
    return out


def _biomarker_stats_from_features(
    features: np.ndarray,
    cohorts: list[str],
    n_rois: int,
) -> dict:
    """
    Reconstruct the symmetric matrix from each feature row, run
    ``compute_fmri_biomarkers``, then group by cohort to return
    {cohort: {metric: {mean, std, n}}}.
    """
    if features.size == 0:
        return {c: {} for c in COHORTS}

    iu = np.triu_indices(n_rois, k=1)
    is_dmn_only = n_rois <= 50

    per_subject: list[dict] = []
    for row in features:
        m = np.zeros((n_rois, n_rois), dtype=np.float32)
        m[iu] = row
        m = m + m.T  # symmetrize
        try:
            per_subject.append(compute_fmri_biomarkers(m, is_dmn_only=is_dmn_only))
        except Exception:
            per_subject.append({})

    out: dict = {}
    for cohort in COHORTS:
        idx = [i for i, c in enumerate(cohorts) if c == cohort]
        if not idx:
            out[cohort] = {}
            continue

        cohort_stats: dict = {}
        for key in BIOMARKER_KEYS:
            vals = [per_subject[i].get(key) for i in idx]
            vals = [v for v in vals if v is not None and math.isfinite(float(v))]
            if not vals:
                continue
            arr = np.asarray(vals, dtype=np.float64)
            cohort_stats[key] = {
                "mean": _safe_float(arr.mean()),
                "std": _safe_float(arr.std(ddof=0)),
                "n": int(arr.size),
            }
        out[cohort] = cohort_stats

    return out


# --------------------------------------------------------------------------- #
# Stage 3 — UMAP embedding + centroids + conversion axis                      #
# --------------------------------------------------------------------------- #

def _fit_manifold(
    features: np.ndarray,
    cohorts: list[str],
    subject_ids: list[str],
):
    """
    Fit UMAP on all baseline FC vectors. Returns ``(mapper, points, centroids)``.

    UMAP needs at least ``n_neighbors + 1`` rows; below that we fall back to a
    deterministic PCA-style projection so the endpoint still returns something
    visualizable on tiny datasets.
    """
    n = features.shape[0]
    if n == 0:
        return None, [], {}

    n_neighbors = min(UMAP_N_NEIGHBORS, max(2, n - 1))

    mapper = None
    if n >= 4:
        try:
            from umap import UMAP

            mapper = UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=UMAP_MIN_DIST,
                random_state=UMAP_RANDOM_STATE,
                metric="euclidean",
            )
            embedding = mapper.fit_transform(features)
        except Exception:
            mapper = None
            embedding = None
    else:
        embedding = None

    if embedding is None:
        # PCA fallback (also used when UMAP isn't available).
        try:
            from sklearn.decomposition import PCA

            n_components = min(2, max(1, n - 1))
            embedding = PCA(n_components=n_components).fit_transform(features)
            if embedding.shape[1] == 1:
                embedding = np.hstack([embedding, np.zeros_like(embedding)])
        except Exception:
            # Last resort — just zero the embedding so callers don't crash.
            embedding = np.zeros((n, 2), dtype=np.float32)

    points = [
        {
            "x": _safe_float(embedding[i, 0]),
            "y": _safe_float(embedding[i, 1]),
            "cohort": cohorts[i],
            "subject_id": subject_ids[i],
        }
        for i in range(n)
    ]

    centroids: dict = {}
    for cohort in COHORTS:
        idx = [i for i, c in enumerate(cohorts) if c == cohort]
        if not idx:
            continue
        cx = float(embedding[idx, 0].mean())
        cy = float(embedding[idx, 1].mean())
        centroids[cohort] = {"x": _safe_float(cx), "y": _safe_float(cy), "n": len(idx)}

    return mapper, points, centroids


def _compute_disease_axes(centroids: dict) -> dict:
    """
    Build the conversion axis. Prefers MCI-NC → AD (most discriminative for
    converter studies); falls back to CN → AD or MCI-NC → Converter if either
    endpoint is missing.
    """
    pairs = [
        ("mci", "ad"),
        ("healthy", "ad"),
        ("mci", "converter"),
    ]

    for origin_key, target_key in pairs:
        if origin_key not in centroids or target_key not in centroids:
            continue
        ox, oy = centroids[origin_key]["x"], centroids[origin_key]["y"]
        tx, ty = centroids[target_key]["x"], centroids[target_key]["y"]
        if None in (ox, oy, tx, ty):
            continue
        dx, dy = tx - ox, ty - oy
        norm = math.hypot(dx, dy)
        if norm <= 1e-9:
            continue
        return {
            "origin": {"x": ox, "y": oy, "cohort": origin_key},
            "target": {"x": tx, "y": ty, "cohort": target_key},
            "direction": {"x": dx / norm, "y": dy / norm},
            "length": norm,
        }

    return {}


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #

def get_cohort_stats(
    data_root: str,
    csv_path: str,
    scan_folders: list[str],
    force_refresh: bool = False,
) -> CohortStats:
    """
    Returns a populated ``CohortStats`` for the given dataset selection.
    Cached per ``(csv_path, sorted scan_folders)``.
    """
    key = _cache_key(csv_path, scan_folders)
    with _LOCK:
        if not force_refresh and key in _CACHE:
            return _CACHE[key]

    abs_csv = os.path.join(data_root, csv_path)
    df = load_metadata(abs_csv)

    features, cohorts, subject_ids, n_rois = _collect_baseline_fc_vectors(
        data_root, df, scan_folders
    )

    biomarker_stats = _biomarker_stats_from_features(features, cohorts, n_rois)
    mapper, points, centroids = _fit_manifold(features, cohorts, subject_ids)
    conversion_axis = _compute_disease_axes(centroids)
    cohort_means = _cohort_mean_matrices(features, cohorts, n_rois)

    stats = CohortStats(
        biomarker_stats=biomarker_stats,
        points=points,
        centroids=centroids,
        conversion_axis=conversion_axis,
        cohort_means=cohort_means,
        umap_mapper=mapper,
        edge_dim=features.shape[1] if features.size else 0,
        n_rois=n_rois,
    )

    with _LOCK:
        _CACHE[key] = stats
    return stats


def project_visits(
    stats: CohortStats,
    matrices: list[np.ndarray],
) -> list[dict]:
    """
    Project a patient's longitudinal correlation matrices into the cached
    UMAP space. Returns a list of ``{x, y, conversion_score}`` dicts in the
    same order as ``matrices``.

    Skips matrices whose shape doesn't match the cached ``n_rois``.
    """
    if stats.umap_mapper is None or stats.edge_dim == 0 or stats.n_rois == 0:
        return [{"x": None, "y": None, "conversion_score": None} for _ in matrices]

    iu = np.triu_indices(stats.n_rois, k=1)
    rows: list[np.ndarray] = []
    keep: list[int] = []
    for i, m in enumerate(matrices):
        if m is None or m.ndim != 2 or m.shape[0] != stats.n_rois:
            continue
        v = m[iu].astype(np.float32, copy=False)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        rows.append(v)
        keep.append(i)

    out: list[dict] = [
        {"x": None, "y": None, "conversion_score": None} for _ in matrices
    ]
    if not rows:
        return out

    try:
        emb = stats.umap_mapper.transform(np.vstack(rows))
    except Exception:
        return out

    axis = stats.conversion_axis or {}
    origin = axis.get("origin")
    direction = axis.get("direction")
    length = axis.get("length", 0.0) or 0.0

    for k, idx in enumerate(keep):
        x, y = float(emb[k, 0]), float(emb[k, 1])
        score: Optional[float] = None
        if origin and direction and length > 1e-9:
            dx = x - origin["x"]
            dy = y - origin["y"]
            raw = dx * direction["x"] + dy * direction["y"]
            score = _safe_float(raw / length)  # normalised: 0=origin, 1=target
        out[idx] = {
            "x": _safe_float(x),
            "y": _safe_float(y),
            "conversion_score": score,
        }

    return out
