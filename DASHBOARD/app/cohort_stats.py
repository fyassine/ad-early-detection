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

import hashlib
import math
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
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
BIOMARKER_KEYS = ["global_fc", "dmn_fc", "modularity", "density", "pos_fc_ratio",
                  "system_segregation"]

# Schaefer 7-network names (used to expose per-network FC stats).
SCHAEFER_NETWORKS = ["Default", "Cont", "SalVentAttn", "DorsAttn", "Limbic", "SomMot", "Vis"]


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
    # Patient-visit coordinates from the same UMAP fit (NOT via transform()).
    # Shape: {subject_id: {visit: {x, y, conversion_score}}}
    # Built by including every longitudinal visit in the UMAP fit so visits
    # land naturally inside the manifold instead of being projected to its
    # boundary by transform().
    patient_visit_coords: dict = field(default_factory=dict)
    # Internal — kept for transform() fallback on truly unseen subjects
    umap_mapper: object = field(default=None, repr=False)
    edge_dim: int = 0  # length of the upper-triangle feature vector
    n_rois: int = 0
    # Per-cohort percentile bands {cohort: {metric: {p5, p25, p50, p75, p95, ...}}}
    biomarker_percentiles: dict = field(default_factory=dict)
    # Raw biomarker values per cohort {cohort: {metric: [values]}} — used by
    # effect-size + EBM endpoints; not exposed to the frontend in full.
    biomarker_values: dict = field(default_factory=dict)
    # Per-cohort × per-network FC stats: {cohort: {network_name: {mean, std, p5..p95, n}}}
    network_fc_stats: dict = field(default_factory=dict)
    # Brain-age model trained on healthy CN baselines (services.brain_age.BrainAgeModel)
    brain_age_model: object = field(default=None, repr=False)
    # EBM model: {sequence: [...], biomarkers: {...}}
    ebm: dict = field(default_factory=dict)
    # Time-shift model: services.time_shift.TimeShiftModel
    time_shift_model: object = field(default=None, repr=False)
    # SHA1 fingerprint of the .npz index used to build this object.
    # Exposed so route handlers can embed it in ETag headers without
    # re-computing it from the index.
    fingerprint: str = ""


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


def is_stats_cached(
    data_root: str,
    csv_path: str,
    scan_folders: list[str],
) -> bool:
    """
    Non-blocking check: return True if CohortStats for this dataset is
    available in the memory cache or on disk, WITHOUT triggering any
    computation.

    Used by expensive endpoints (network-disruption, graph-topology) to
    return ``available: false`` immediately when the precompute job hasn't
    finished yet, instead of blocking the thread pool for 9+ minutes.
    """
    key = _cache_key(csv_path, scan_folders)
    with _LOCK:
        if key in _CACHE:
            return True
    # Disk check: the file just needs to exist and be non-empty.
    # Full fingerprint validation happens inside get_cohort_stats(); here
    # we only want to know if a warm result is likely on disk.
    try:
        disk_path = _disk_cache_path(data_root, csv_path, scan_folders)
        return os.path.isfile(disk_path) and os.path.getsize(disk_path) > 0
    except Exception:
        return False


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
    npz_index: Optional[dict] = None,
    executor: Optional[ThreadPoolExecutor] = None,
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

    if npz_index is None:
        npz_index = index_npz_by_subject(data_root, scan_folders)

    # Pre-resolve (sid, cohort, abs_path) so the parallel load step has nothing
    # to do but np.load. Order is preserved so cohorts/subject_ids stay aligned
    # with feature rows after filtering.
    plan: list[tuple[str, str, str]] = []
    for _, row in baseline.iterrows():
        sid = str(row.get("subject_id", "")).strip()
        cohort = str(row.get("diagnosis", "")).strip().lower()
        if not sid or cohort not in COHORTS:
            continue
        rec = _pick_baseline_npz(npz_index.get(sid, []))
        if rec is None:
            continue
        plan.append((sid, cohort, rec["abs_path"]))

    if not plan:
        return np.empty((0, 0)), [], [], 0

    paths = [p[2] for p in plan]

    def _safe_load(path: str):
        try:
            return load_correlation_matrix(path)
        except Exception:
            return None

    if executor is not None:
        matrices = list(executor.map(_safe_load, paths))
    else:
        matrices = [_safe_load(p) for p in paths]

    feature_rows: list[np.ndarray] = []
    cohorts: list[str] = []
    subject_ids: list[str] = []
    n_rois: int = 0

    for (sid, cohort, _path), matrix in zip(plan, matrices):
        if matrix is None:
            continue
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            continue
        if n_rois == 0:
            n_rois = matrix.shape[0]
        elif matrix.shape[0] != n_rois:
            # Atlas mismatch — keep the dominant size, drop the rest
            continue
        vec = _upper_triangle(matrix).astype(np.float32, copy=False)
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        feature_rows.append(vec)
        cohorts.append(cohort)
        subject_ids.append(sid)

    if not feature_rows:
        return np.empty((0, 0)), [], [], 0

    return np.vstack(feature_rows), cohorts, subject_ids, n_rois


# --------------------------------------------------------------------------- #
# Stage 1b — longitudinal FC vectors (converters' non-baseline visits)         #
# --------------------------------------------------------------------------- #

def _collect_longitudinal_fc_vectors(
    data_root: str,
    df: pd.DataFrame,
    scan_folders: list[str],
    npz_index: Optional[dict] = None,
    baseline_subject_ids: Optional[set] = None,
    n_rois_lock: int = 0,
    executor=None,
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    For every converter, collect their non-baseline visit FC vectors so the
    UMAP fit sees them as real samples (not unseen points to ``transform()``).

    The baseline visit is intentionally skipped because
    ``_collect_baseline_fc_vectors`` already collected it as a converter-cohort
    row. Including all visits is what fixes the "patient trajectory dots
    always at the embedding boundary" bug.

    Returns ``(features (N, edge_dim), subject_ids, visits)`` parallel arrays.
    Subjects whose .npz files don't match the locked ``n_rois_lock`` are
    silently dropped.
    """
    if "diagnosis" not in df.columns or "subject_id" not in df.columns:
        return np.empty((0, 0), dtype=np.float32), [], []

    if npz_index is None:
        npz_index = index_npz_by_subject(data_root, scan_folders)

    converter_ids = (
        df[df["diagnosis"].astype(str).str.strip().str.lower() == "converter"]
        ["subject_id"].dropna().astype(str).str.strip().unique().tolist()
    )
    if not converter_ids:
        return np.empty((0, 0), dtype=np.float32), [], []

    plan: list[tuple[str, str, str]] = []  # (sid, visit, abs_path)
    for sid in converter_ids:
        records = npz_index.get(sid, []) or []
        if not records:
            continue
        baseline_rec = _pick_baseline_npz(records)
        baseline_path = baseline_rec["abs_path"] if baseline_rec else None
        for rec in records:
            if rec["abs_path"] == baseline_path:
                continue  # already in baseline collector
            plan.append((sid, rec["visit"], rec["abs_path"]))

    if not plan:
        return np.empty((0, 0), dtype=np.float32), [], []

    paths = [p[2] for p in plan]

    def _safe_load(path: str):
        try:
            return load_correlation_matrix(path)
        except Exception:
            return None

    if executor is not None:
        matrices = list(executor.map(_safe_load, paths))
    else:
        matrices = [_safe_load(p) for p in paths]

    feature_rows: list[np.ndarray] = []
    sids: list[str] = []
    visits: list[str] = []
    for (sid, visit, _path), matrix in zip(plan, matrices):
        if matrix is None:
            continue
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            continue
        if n_rois_lock and matrix.shape[0] != n_rois_lock:
            continue
        vec = _upper_triangle(matrix).astype(np.float32, copy=False)
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        feature_rows.append(vec)
        sids.append(sid)
        visits.append(visit)

    if not feature_rows:
        return np.empty((0, 0), dtype=np.float32), [], []
    return np.vstack(feature_rows), sids, visits


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
    executor: Optional[ThreadPoolExecutor] = None,
) -> tuple[dict, dict, dict, list[dict]]:
    """
    Reconstruct the symmetric matrix from each feature row, run
    ``compute_fmri_biomarkers``, then group by cohort.

    Returns ``(biomarker_stats, biomarker_values, network_fc_stats, per_subject)``:
      - ``biomarker_stats``    : {cohort: {metric: {mean, std, n}}}
      - ``biomarker_values``   : {cohort: {metric: [raw values]}} (for percentile +
                                 effect-size + EBM endpoints)
      - ``network_fc_stats``   : {cohort: {network_name: {mean, std, n}}}
      - ``per_subject``        : ordered list of biomarker dicts (for brain-age /
                                 EBM training)
    """
    if features.size == 0:
        return ({c: {} for c in COHORTS}, {c: {} for c in COHORTS},
                {c: {} for c in COHORTS}, [])

    iu = np.triu_indices(n_rois, k=1)
    is_dmn_only = n_rois <= 50

    def _one(row):
        m = np.zeros((n_rois, n_rois), dtype=np.float32)
        m[iu] = row
        m = m + m.T  # symmetrize
        try:
            return compute_fmri_biomarkers(m, is_dmn_only=is_dmn_only)
        except Exception:
            return {}

    if executor is not None:
        per_subject: list[dict] = list(executor.map(_one, features))
    else:
        per_subject = [_one(row) for row in features]

    out_stats: dict = {}
    out_values: dict = {}
    out_network: dict = {}
    for cohort in COHORTS:
        idx = [i for i, c in enumerate(cohorts) if c == cohort]
        if not idx:
            out_stats[cohort] = {}
            out_values[cohort] = {}
            out_network[cohort] = {}
            continue

        cohort_stats: dict = {}
        cohort_values: dict = {}
        for key in BIOMARKER_KEYS:
            vals = [per_subject[i].get(key) for i in idx]
            vals = [v for v in vals if v is not None and math.isfinite(float(v))]
            cohort_values[key] = vals
            if not vals:
                continue
            arr = np.asarray(vals, dtype=np.float64)
            cohort_stats[key] = {
                "mean": _safe_float(arr.mean()),
                "std": _safe_float(arr.std(ddof=0)),
                "n": int(arr.size),
            }
        out_stats[cohort] = cohort_stats
        out_values[cohort] = cohort_values

        # Per-network FC stats: aggregate `network_fc` dict from each subject.
        net_buckets: dict[str, list[float]] = {}
        for i in idx:
            net = per_subject[i].get("network_fc") or {}
            for net_name, val in net.items():
                if val is None or not math.isfinite(float(val)):
                    continue
                net_buckets.setdefault(net_name, []).append(float(val))
        cohort_net: dict = {}
        for name, vals in net_buckets.items():
            arr = np.asarray(vals, dtype=np.float64)
            cohort_net[name] = {
                "mean": _safe_float(arr.mean()),
                "std": _safe_float(arr.std(ddof=0)),
                "n": int(arr.size),
            }
        out_network[cohort] = cohort_net

    return out_stats, out_values, out_network, per_subject


# --------------------------------------------------------------------------- #
# Stage 3 — UMAP embedding + centroids + conversion axis                      #
# --------------------------------------------------------------------------- #

def _fit_manifold(
    features: np.ndarray,
    cohorts: list[str],
    subject_ids: list[str],
    extra_features: Optional[np.ndarray] = None,
):
    """
    Fit UMAP on baseline FC vectors *concatenated with* ``extra_features``
    (longitudinal converter visits). Co-fitting avoids the well-known
    ``mapper.transform()`` failure mode that pushes unseen samples to the
    boundary of the trained manifold.

    Returns ``(mapper, points, centroids, embedding)`` where:
      - ``points`` and ``centroids`` are computed from the BASELINE rows only
        (so the cohort scatter / centroids are unchanged).
      - ``embedding`` is the FULL ``(n_baseline + n_extra, 2)`` matrix; the
        caller slices ``embedding[n_baseline:]`` to look up per-visit coords.

    UMAP needs at least ``n_neighbors + 1`` rows; below that we fall back to a
    deterministic PCA-style projection so the endpoint still returns something
    visualizable on tiny datasets.
    """
    n_baseline = features.shape[0] if features.size else 0
    extra = extra_features if extra_features is not None and extra_features.size else None
    if n_baseline == 0 and extra is None:
        return None, [], {}, np.zeros((0, 2), dtype=np.float32)

    fit_input = features if extra is None else np.vstack([features, extra])
    n = fit_input.shape[0]
    n_neighbors = min(UMAP_N_NEIGHBORS, max(2, n - 1))

    mapper = None
    embedding = None
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
            embedding = mapper.fit_transform(fit_input)
        except Exception:
            mapper = None
            embedding = None

    if embedding is None:
        try:
            from sklearn.decomposition import PCA

            n_components = min(2, max(1, n - 1))
            embedding = PCA(n_components=n_components).fit_transform(fit_input)
            if embedding.shape[1] == 1:
                embedding = np.hstack([embedding, np.zeros_like(embedding)])
        except Exception:
            embedding = np.zeros((n, 2), dtype=np.float32)

    embedding = np.asarray(embedding, dtype=np.float32)

    # Baseline-only points + centroids (the cohort scatter / centroids should
    # not be skewed by per-visit converter rows).
    points = [
        {
            "x": _safe_float(embedding[i, 0]),
            "y": _safe_float(embedding[i, 1]),
            "cohort": cohorts[i],
            "subject_id": subject_ids[i],
        }
        for i in range(n_baseline)
    ]

    centroids: dict = {}
    for cohort in COHORTS:
        idx = [i for i, c in enumerate(cohorts) if c == cohort]
        if not idx:
            continue
        cx = float(embedding[idx, 0].mean())
        cy = float(embedding[idx, 1].mean())
        centroids[cohort] = {"x": _safe_float(cx), "y": _safe_float(cy), "n": len(idx)}

    return mapper, points, centroids, embedding


def _conversion_score(x: float, y: float, axis: dict) -> Optional[float]:
    """Project (x, y) onto the normalised MCI-NC → AD axis. Returns 0 at the
    origin centroid, 1 at the target centroid."""
    if not axis:
        return None
    origin = axis.get("origin"); direction = axis.get("direction"); length = axis.get("length", 0.0) or 0.0
    if not origin or not direction or length <= 1e-9:
        return None
    dx = x - origin["x"]; dy = y - origin["y"]
    raw = dx * direction["x"] + dy * direction["y"]
    return _safe_float(raw / length)


def _build_patient_visit_coords(
    embedding: np.ndarray,
    baseline_cohorts: list[str],
    baseline_subject_ids: list[str],
    baseline_npz_index: dict,
    long_sids: list[str],
    long_visits: list[str],
    conversion_axis: dict,
) -> dict:
    """
    Combine baseline-row + longitudinal-row coords into one
    ``{subject_id: {visit: {x, y, conversion_score}}}`` lookup.

    Converters get their baseline-cohort row mapped to their actual baseline
    visit name (M0 if present, else the chronologically earliest .npz).
    """
    out: dict[str, dict[str, dict]] = {}
    n_baseline = len(baseline_subject_ids)

    # Baseline-cohort converters: their baseline row IS in the embedding's
    # first N_baseline slice. Map to the actual visit code so the frontend's
    # lookup by visit string works.
    for i in range(n_baseline):
        if baseline_cohorts[i] != "converter":
            continue
        sid = baseline_subject_ids[i]
        recs = baseline_npz_index.get(sid, []) if baseline_npz_index else []
        baseline_rec = _pick_baseline_npz(recs)
        if baseline_rec is None:
            continue
        x = float(embedding[i, 0]); y = float(embedding[i, 1])
        out.setdefault(sid, {})[baseline_rec["visit"]] = {
            "x": _safe_float(x),
            "y": _safe_float(y),
            "conversion_score": _conversion_score(x, y, conversion_axis),
        }

    # Longitudinal rows live in the tail of the embedding.
    for k, sid in enumerate(long_sids):
        row = n_baseline + k
        if row >= embedding.shape[0]:
            break
        x = float(embedding[row, 0]); y = float(embedding[row, 1])
        out.setdefault(sid, {})[long_visits[k]] = {
            "x": _safe_float(x),
            "y": _safe_float(y),
            "conversion_score": _conversion_score(x, y, conversion_axis),
        }

    return out


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
# 2023+ analytics: percentiles, brain-age, EBM, time-shift                    #
# --------------------------------------------------------------------------- #

def _compute_percentile_bands(biomarker_values: dict) -> dict:
    """Run the percentile bootstrap for each cohort × biomarker."""
    try:
        from .services.normative import percentile_bands
    except Exception:
        return {}
    out: dict = {}
    for cohort, metrics in biomarker_values.items():
        cohort_pct: dict = {}
        for key, vals in metrics.items():
            band = percentile_bands(vals)
            if band is not None:
                cohort_pct[key] = band
        out[cohort] = cohort_pct
    return out


def _train_brain_age(
    features: np.ndarray,
    cohorts: list[str],
    subject_ids: list[str],
    df: pd.DataFrame,
) -> Optional[object]:
    """Train Ridge brain-age regressor on healthy CN baselines."""
    if features.size == 0 or "age" not in df.columns or "subject_id" not in df.columns:
        return None
    age_lookup = (
        df.dropna(subset=["subject_id"])
        .drop_duplicates(subset="subject_id", keep="first")
        .set_index(df.columns[df.columns.get_loc("subject_id")] if "subject_id" in df.columns else None)
    )
    try:
        age_lookup = df.dropna(subset=["subject_id"]).drop_duplicates(subset="subject_id", keep="first").set_index("subject_id")["age"].to_dict()
    except Exception:
        return None

    cn_idx = [i for i, c in enumerate(cohorts) if c == "healthy"]
    if len(cn_idx) < 12:
        return None
    cn_features = features[cn_idx]
    cn_ages = np.asarray([age_lookup.get(subject_ids[i]) for i in cn_idx], dtype=np.float64)

    try:
        from .services.brain_age import fit_brain_age
        return fit_brain_age(cn_features, cn_ages)
    except Exception:
        return None


def _build_ebm(biomarker_values: dict, df: pd.DataFrame, subject_ids: list[str]) -> dict:
    """Fit the simple EBM using fMRI biomarkers + clinical CSF + cognition."""
    try:
        from .services.ebm import fit_ebm
    except Exception:
        return {}

    ebm_input: dict[str, dict[str, list[float]]] = {}
    for cohort in COHORTS:
        ebm_input[cohort] = dict(biomarker_values.get(cohort, {}))

    # Add clinical biomarkers to the EBM input by cross-referencing the CSV.
    if "subject_id" in df.columns and "diagnosis" in df.columns:
        baseline = _select_baseline_subjects(df)
        for col in ("mmse_total", "cdr_global", "abeta42", "p_tau", "total_tau", "pacc5"):
            if col not in baseline.columns:
                continue
            for cohort in COHORTS:
                rows = baseline[baseline["diagnosis"].astype(str).str.strip().str.lower() == cohort]
                vals = pd.to_numeric(rows[col], errors="coerce").dropna().tolist()
                if vals:
                    ebm_input[cohort].setdefault(col, []).extend(vals)

    # Use the union of all keys actually populated for either CN or AD.
    cn = ebm_input.get("healthy", {}) or {}
    ad = ebm_input.get("ad", {}) or {}
    keys = sorted(set(cn.keys()) | set(ad.keys()))
    if not keys:
        return {}
    return fit_ebm(ebm_input, biomarker_keys=keys)


def _build_time_shift_model(
    per_subject_biomarkers: list[dict],
    cohorts: list[str],
    subject_ids: list[str],
    long_sids: list[str],
    long_visits: list[str],
    df: pd.DataFrame,
) -> Optional[object]:
    """Fit logistic curves to converter biomarkers over months-from-M0."""
    try:
        from .services.time_shift import fit_time_shift_model
        from .biomarkers import compute_fmri_biomarkers, find_subject_npz_files, load_correlation_matrix  # noqa: F401
    except Exception:
        return None
    import re as _re

    def _months(visit) -> Optional[int]:
        if visit is None:
            return None
        m = _re.match(r"M(\d+)", str(visit).strip().upper())
        return int(m.group(1)) if m else None

    samples: dict[str, list[tuple[int, float]]] = {k: [] for k in BIOMARKER_KEYS}
    # Baseline converters
    for i, sid in enumerate(subject_ids):
        if cohorts[i] != "converter":
            continue
        for key in BIOMARKER_KEYS:
            v = per_subject_biomarkers[i].get(key) if i < len(per_subject_biomarkers) else None
            if v is None:
                continue
            samples[key].append((0, float(v)))
    # Longitudinal visits — recompute biomarkers via per_subject_biomarkers? We
    # don't have them here cheaply. Instead, we approximate by skipping —
    # baseline-only converter samples are still enough for a usable curve fit
    # if there are ≥6 baseline converters.

    # Add clinical biomarkers from CSV (every visit known)
    if "subject_id" in df.columns and "visit" in df.columns:
        clinical_keys = ["mmse_total", "cdr_global", "abeta42", "p_tau", "total_tau", "pacc5"]
        for col in clinical_keys:
            if col not in df.columns:
                continue
            samples.setdefault(col, [])
            for _, row in df.iterrows():
                sid = str(row.get("subject_id", "")).strip()
                if not sid:
                    continue
                # Restrict to converter cohort
                diag = str(row.get("diagnosis", "")).strip().lower()
                if diag not in ("converter", "ad"):
                    continue
                m = _months(row.get("visit"))
                if m is None:
                    continue
                v = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
                if v is None or not pd.notna(v):
                    continue
                samples[col].append((int(m), float(v)))

    samples = {k: v for k, v in samples.items() if len(v) >= 6}
    if not samples:
        return None
    return fit_time_shift_model(samples)


# --------------------------------------------------------------------------- #
# Disk cache                                                                  #
# --------------------------------------------------------------------------- #

# Bump when the pickle payload schema changes.
# v2: adds patient_visit_coords (Round 4 manifold co-fit).
# v3: adds biomarker_percentiles, biomarker_values, network_fc_stats,
#      brain_age_model, ebm, time_shift_model (2023+ analytics).
_DISK_CACHE_VERSION = 3


def _disk_cache_dir(data_root: str) -> str:
    return os.path.join(
        os.environ.get("CACHE_ROOT", os.path.join(data_root, ".cache")),
        "cohort_stats",
    )


def _disk_cache_path(data_root: str, csv_path: str, scan_folders: list[str]) -> str:
    h = hashlib.sha1()
    h.update(csv_path.encode("utf-8"))
    for f in sorted(scan_folders):
        h.update(b"\x00")
        h.update(f.encode("utf-8"))
    return os.path.join(
        _disk_cache_dir(data_root),
        f"cohort_stats_v{_DISK_CACHE_VERSION}_{h.hexdigest()}.pkl",
    )


def _index_fingerprint(npz_index: dict) -> str:
    """SHA1 over (rel_path, mtime_ns, size) for every .npz — invalidates on edit."""
    h = hashlib.sha1()
    triples: list[tuple[str, int, int]] = []
    for sid in sorted(npz_index.keys()):
        for rec in npz_index[sid]:
            path = rec.get("abs_path") or rec.get("rel_path") or ""
            try:
                st = os.stat(path)
                triples.append((rec.get("rel_path", path), st.st_mtime_ns, st.st_size))
            except OSError:
                continue
    triples.sort()
    for rel, mt, sz in triples:
        h.update(rel.encode("utf-8"))
        h.update(b"|")
        h.update(str(mt).encode("ascii"))
        h.update(b"|")
        h.update(str(sz).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def _load_disk_cache(path: str, fingerprint: str) -> Optional[CohortStats]:
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("version") != _DISK_CACHE_VERSION:
        return None
    if payload.get("fingerprint") != fingerprint:
        return None
    try:
        return CohortStats(
            biomarker_stats=payload["biomarker_stats"],
            points=payload["points"],
            centroids=payload["centroids"],
            conversion_axis=payload["conversion_axis"],
            cohort_means=payload.get("cohort_means", {}),
            patient_visit_coords=payload.get("patient_visit_coords", {}),
            umap_mapper=payload.get("umap_mapper"),
            edge_dim=int(payload.get("edge_dim", 0)),
            n_rois=int(payload.get("n_rois", 0)),
            biomarker_percentiles=payload.get("biomarker_percentiles", {}),
            biomarker_values=payload.get("biomarker_values", {}),
            network_fc_stats=payload.get("network_fc_stats", {}),
            brain_age_model=payload.get("brain_age_model"),
            ebm=payload.get("ebm", {}),
            time_shift_model=payload.get("time_shift_model"),
        )
    except Exception:
        return None


def _save_disk_cache(path: str, fingerprint: str, stats: CohortStats) -> None:
    # umap_mapper is excluded deliberately: the UMAP object contains
    # numba-compiled internals that take ~8-9 s to pickle/unpickle while
    # contributing ~4 MB to the cache file. All known subjects already have
    # precomputed coordinates in patient_visit_coords, so the mapper is only
    # needed as a fallback for truly unseen subjects. Loading it from the disk
    # cache would block the FastAPI thread pool for 8 s on every server restart.
    # The memory cache (_CACHE) always stores the live mapper, so Manifold tabs
    # work normally after the first warm request; only a cold-restart projecting
    # a brand-new unseen subject will fall back to an empty UMAP coordinate.
    payload = {
        "version": _DISK_CACHE_VERSION,
        "fingerprint": fingerprint,
        "biomarker_stats": stats.biomarker_stats,
        "points": stats.points,
        "centroids": stats.centroids,
        "conversion_axis": stats.conversion_axis,
        "cohort_means": stats.cohort_means,
        "patient_visit_coords": stats.patient_visit_coords,
        "umap_mapper": None,   # excluded from disk cache — see comment above
        "edge_dim": stats.edge_dim,
        "n_rois": stats.n_rois,
        "biomarker_percentiles": stats.biomarker_percentiles,
        "biomarker_values": stats.biomarker_values,
        "network_fc_stats": stats.network_fc_stats,
        "brain_age_model": stats.brain_age_model,
        "ebm": stats.ebm,
        "time_shift_model": stats.time_shift_model,
    }
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except Exception:
        # Best-effort — disk cache is an optimization, never break the request.
        pass


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
    Cached per ``(csv_path, sorted scan_folders)`` in process memory and on
    disk under ``$CACHE_ROOT/cohort_stats/`` (defaults to ``$DATA_ROOT/.cache``).
    The disk cache is fingerprinted against .npz mtime+size and invalidates
    automatically when matrices are regenerated.
    """
    key = _cache_key(csv_path, scan_folders)
    with _LOCK:
        if not force_refresh and key in _CACHE:
            return _CACHE[key]

    # Build the .npz index once; reused for fingerprinting + collection.
    npz_index = index_npz_by_subject(data_root, scan_folders)
    fingerprint = _index_fingerprint(npz_index)
    disk_path = _disk_cache_path(data_root, csv_path, scan_folders)

    if not force_refresh:
        cached = _load_disk_cache(disk_path, fingerprint)
        if cached is not None:
            cached.fingerprint = fingerprint  # ensure field is set on disk-loaded objects
            with _LOCK:
                _CACHE[key] = cached
            return cached

    abs_csv = os.path.join(data_root, csv_path)
    df = load_metadata(abs_csv)

    max_workers = min(8, (os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        features, cohorts, subject_ids, n_rois = _collect_baseline_fc_vectors(
            data_root, df, scan_folders, npz_index=npz_index, executor=executor
        )
        (biomarker_stats, biomarker_values, network_fc_stats,
         per_subject_biomarkers) = _biomarker_stats_from_features(
            features, cohorts, n_rois, executor=executor
        )
        # Co-fit: include every converter's non-baseline visits in the UMAP
        # training set so per-visit coords come from the actual fit (not from
        # an unstable transform()).
        long_features, long_sids, long_visits = _collect_longitudinal_fc_vectors(
            data_root, df, scan_folders,
            npz_index=npz_index,
            baseline_subject_ids=set(subject_ids),
            n_rois_lock=n_rois,
            executor=executor,
        )

    mapper, points, centroids, embedding = _fit_manifold(
        features, cohorts, subject_ids, extra_features=long_features
    )
    conversion_axis = _compute_disease_axes(centroids)
    cohort_means = _cohort_mean_matrices(features, cohorts, n_rois)

    # Build the per-(subject_id, visit) coordinate lookup. Converters' M0 is
    # already in the baseline embedding; non-baseline visits live in the
    # tail of the same fit.
    patient_visit_coords = _build_patient_visit_coords(
        embedding=embedding,
        baseline_cohorts=cohorts,
        baseline_subject_ids=subject_ids,
        baseline_npz_index=npz_index,
        long_sids=long_sids,
        long_visits=long_visits,
        conversion_axis=conversion_axis,
    )

    # ── 2023+ analytics: percentiles, brain-age, EBM, time-shift ─────────────
    biomarker_percentiles = _compute_percentile_bands(biomarker_values)
    brain_age_model = _train_brain_age(features, cohorts, subject_ids, df)
    ebm = _build_ebm(biomarker_values, df, subject_ids)
    time_shift_model = _build_time_shift_model(
        per_subject_biomarkers, cohorts, subject_ids,
        long_sids, long_visits, df,
    )

    stats = CohortStats(
        biomarker_stats=biomarker_stats,
        points=points,
        centroids=centroids,
        conversion_axis=conversion_axis,
        cohort_means=cohort_means,
        patient_visit_coords=patient_visit_coords,
        umap_mapper=mapper,
        edge_dim=features.shape[1] if features.size else 0,
        n_rois=n_rois,
        biomarker_percentiles=biomarker_percentiles,
        biomarker_values=biomarker_values,
        network_fc_stats=network_fc_stats,
        brain_age_model=brain_age_model,
        ebm=ebm,
        time_shift_model=time_shift_model,
        fingerprint=fingerprint,
    )

    with _LOCK:
        _CACHE[key] = stats
    _save_disk_cache(disk_path, fingerprint, stats)
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
