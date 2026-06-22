"""
biomarkers.py — Compute fMRI-derived biomarkers from correlation matrices.

All metrics are computed purely from the correlation matrix (.npz files).
No clinical scores or timeseries needed.

Metrics:
  - Global FC:  mean functional connectivity (upper triangle of corr matrix)
  - DMN FC:     mean FC within Default Mode Network ROIs
  - Modularity: greedy modularity Q from networkx (thresholded graph)
"""

import math
import os
import re
from threading import Lock

import numpy as np

# --------------------------------------------------------------------------- #
# Module-level file-index cache                                               #
#                                                                             #
# Walking thousands of .npz / .nii.gz files for every patient open is the     #
# main source of latency in the modal. We walk once per (data_root, folders)  #
# pair and cache the result for the lifetime of the process — folders don't  #
# change while the server is running. Restart the server to invalidate.       #
# --------------------------------------------------------------------------- #

_NPZ_INDEX_CACHE: dict[tuple, dict[str, list[dict]]] = {}
_NIFTI_INDEX_CACHE: dict[tuple, dict[str, list[dict]]] = {}
# Per-subject trajectory cache — modularity computation is the dominant cost
# of opening a patient (~300ms/visit), so cache the full result per
# (data_root, sorted folders, subject_id).
_TRAJECTORY_CACHE: dict[tuple, dict] = {}
# Per-visit loaded matrix cache — both /trajectory and /matrix endpoints can
# share a single np.load call per .npz path.
_MATRIX_CACHE: dict[str, np.ndarray] = {}
_INDEX_LOCK = Lock()


def _index_key(data_root: str, folder_paths: list[str]) -> tuple:
    return (data_root, tuple(sorted(folder_paths)))


def clear_index_cache() -> None:
    """Drop the cached file indices and trajectory results."""
    with _INDEX_LOCK:
        _NPZ_INDEX_CACHE.clear()
        _NIFTI_INDEX_CACHE.clear()
        _TRAJECTORY_CACHE.clear()
        _MATRIX_CACHE.clear()


def _safe_float(v) -> float | None:
    """Return None for NaN/inf so JSONResponse (allow_nan=False) never crashes."""
    if v is None:
        return None
    f = float(v)
    return None if not math.isfinite(f) else f

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


# Schaefer 200 parcels, 7-network order: Default Mode Network ROI indices
# These are the ROIs labeled "Default" in Schaefer2018_200Parcels_7Networks
# LH indices (0-based): 49-65 (17 ROIs), RH indices: 149-165 (17 ROIs)
# Total: 34 DMN ROIs out of 200
SCHAEFER_200_DMN_INDICES = list(range(49, 66)) + list(range(149, 166))

# For 46-ROI DMN-only matrices (v4), all ROIs are DMN — use full matrix
# No special indices needed


def compute_fmri_biomarkers(corr_matrix: np.ndarray, is_dmn_only: bool = False) -> dict:
    """
    Compute fMRI-derived biomarkers from a correlation matrix.

    Parameters
    ----------
    corr_matrix : np.ndarray
        (N, N) symmetric correlation matrix.
    is_dmn_only : bool
        If True, the matrix contains only DMN ROIs (e.g., 46x46).
        Global FC and DMN FC will be the same in this case.

    Returns
    -------
    dict with keys: global_fc, dmn_fc, modularity, n_rois, density
    """
    n = corr_matrix.shape[0]
    idx = np.triu_indices(n, k=1)
    upper_tri = corr_matrix[idx]

    metrics = {
        "n_rois": int(n),
        "global_fc": _safe_float(np.nanmean(upper_tri)),
    }

    # DMN FC
    if is_dmn_only or n <= 50:
        # Entire matrix is DMN
        metrics["dmn_fc"] = metrics["global_fc"]
    else:
        # Extract DMN sub-matrix
        dmn_idx = [i for i in SCHAEFER_200_DMN_INDICES if i < n]
        if len(dmn_idx) >= 2:
            dmn_sub = corr_matrix[np.ix_(dmn_idx, dmn_idx)]
            dmn_tri = np.triu_indices(len(dmn_idx), k=1)
            metrics["dmn_fc"] = _safe_float(np.nanmean(dmn_sub[dmn_tri]))
        else:
            metrics["dmn_fc"] = None

    # Modularity
    if HAS_NETWORKX:
        try:
            metrics["modularity"] = _safe_float(_compute_modularity(corr_matrix))
        except Exception:
            metrics["modularity"] = None
    else:
        metrics["modularity"] = None

    # Additional: positive/negative FC ratio
    metrics["pos_fc_ratio"] = _safe_float(np.sum(upper_tri > 0) / max(len(upper_tri), 1))

    # FC density (proportion of strong connections, |r| > 0.3)
    metrics["density"] = _safe_float(np.sum(np.abs(upper_tri) > 0.3) / max(len(upper_tri), 1))

    # ── Schaefer 7-network metrics (Setton 2023; Chan 2014 segregation) ─────
    # Only attempt when the matrix matches a Schaefer N-parcel atlas (n in
    # {100, 200, 400, 600, 800, 1000}); skipped silently for DMN-only or
    # mismatched sizes.
    if not is_dmn_only and n in (100, 200, 400, 600, 800, 1000):
        try:
            from .services.networks import per_network_fc, system_segregation
            net_fc = per_network_fc(corr_matrix, n_parcels=n)
            if net_fc:
                metrics["network_fc"] = net_fc
            seg = system_segregation(corr_matrix, n_parcels=n)
            if seg is not None:
                metrics["system_segregation"] = seg
        except Exception:
            # Network labels missing or atlas JSON unavailable — keep
            # legacy biomarkers without crashing.
            pass

    return metrics


def _compute_modularity(corr_matrix: np.ndarray) -> float:
    """Compute greedy modularity Q from thresholded correlation matrix."""
    # Threshold: keep positive edges > 0.1
    thresh = np.where(corr_matrix > 0.1, corr_matrix, 0.0)
    np.fill_diagonal(thresh, 0)

    G = nx.from_numpy_array(thresh)

    # Remove isolates for stability
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)

    if G.number_of_nodes() < 3:
        return 0.0

    communities = nx.community.greedy_modularity_communities(G)
    return float(nx.community.modularity(G, communities))


_VISIT_FILENAME_RE = re.compile(r"_(M\d+)_")
_SUBJECT_RE = re.compile(r"sub-([a-zA-Z0-9]+)")


def _detect_nifti_visit(filename: str, rel_dir: str) -> str:
    """
    Visit detection for NIfTI: filename first, then parent directory chain
    (DELCODE longitudinal puts the visit in ``Postprocessed_M12/`` etc.).
    """
    m = _VISIT_FILENAME_RE.search(filename)
    if m:
        return m.group(1)
    m = re.search(r"_(M\d+)$|_(M\d+)\.", filename)
    if m:
        return m.group(1) or m.group(2)
    for seg in rel_dir.split(os.sep):
        m = re.search(r"\bM(\d+)\b", seg)
        if m:
            return f"M{m.group(1)}"
        m = re.search(r"_M(\d+)\b", seg)
        if m:
            return f"M{m.group(1)}"
    return "unknown"


def _vkey(rec: dict) -> int:
    m = re.search(r"M(\d+)", rec.get("visit", "M999"))
    return int(m.group(1)) if m else 999


def _build_npz_index(data_root: str, folder_paths: list[str]) -> dict[str, list[dict]]:
    by_sid: dict[str, list[dict]] = {}
    for folder_rel in folder_paths:
        folder = os.path.join(data_root, folder_rel)
        if not os.path.isdir(folder):
            continue
        for dirpath, _, filenames in os.walk(folder):
            for fn in filenames:
                if not fn.endswith(".npz"):
                    continue
                if "z_transformed" in fn or "z_transform" in fn:
                    continue
                m = _SUBJECT_RE.search(fn)
                if not m:
                    continue
                sid = m.group(1)
                abs_path = os.path.join(dirpath, fn)
                vmatch = _VISIT_FILENAME_RE.search(fn)
                visit = vmatch.group(1) if vmatch else "unknown"
                by_sid.setdefault(sid, []).append({
                    "visit": visit,
                    "abs_path": abs_path,
                    "rel_path": os.path.relpath(abs_path, data_root),
                    "filename": fn,
                })
    for recs in by_sid.values():
        recs.sort(key=_vkey)
    return by_sid


def _build_nifti_index(data_root: str, folder_paths: list[str]) -> dict[str, list[dict]]:
    by_sid: dict[str, list[dict]] = {}
    for folder_rel in folder_paths:
        folder = os.path.join(data_root, folder_rel)
        if not os.path.isdir(folder):
            continue
        for dirpath, _, filenames in os.walk(folder):
            rel_dir = os.path.relpath(dirpath, data_root)
            for fn in filenames:
                if not (fn.endswith(".nii.gz") or fn.endswith(".nii")):
                    continue
                m = _SUBJECT_RE.search(fn)
                if not m:
                    continue
                sid = m.group(1)
                abs_path = os.path.join(dirpath, fn)
                by_sid.setdefault(sid, []).append({
                    "visit": _detect_nifti_visit(fn, rel_dir),
                    "abs_path": abs_path,
                    "rel_path": os.path.relpath(abs_path, data_root),
                    "filename": fn,
                })
    for recs in by_sid.values():
        recs.sort(key=_vkey)
    return by_sid


def index_npz_by_subject(
    data_root: str,
    folder_paths: list[str],
) -> dict[str, list[dict]]:
    """
    Walk each scan folder ONCE and group every .npz correlation-matrix file
    by subject_id. Cached at module level for the lifetime of the process.

    Returns ``{subject_id: [{visit, abs_path, rel_path, filename}, ...]}``,
    each list sorted chronologically by visit (M0, M12…).
    """
    key = _index_key(data_root, folder_paths)
    with _INDEX_LOCK:
        cached = _NPZ_INDEX_CACHE.get(key)
        if cached is not None:
            return cached
    built = _build_npz_index(data_root, folder_paths)
    with _INDEX_LOCK:
        _NPZ_INDEX_CACHE[key] = built
    return built


def index_nifti_by_subject(
    data_root: str,
    folder_paths: list[str],
) -> dict[str, list[dict]]:
    """Cached counterpart of ``index_npz_by_subject`` for ``.nii.gz`` volumes."""
    key = _index_key(data_root, folder_paths)
    with _INDEX_LOCK:
        cached = _NIFTI_INDEX_CACHE.get(key)
        if cached is not None:
            return cached
    built = _build_nifti_index(data_root, folder_paths)
    with _INDEX_LOCK:
        _NIFTI_INDEX_CACHE[key] = built
    return built


def find_subject_npz_files(
    data_root: str,
    folder_paths: list[str],
    subject_id: str,
) -> list[dict]:
    """Look up a single subject's .npz files via the cached folder index."""
    return index_npz_by_subject(data_root, folder_paths).get(subject_id, [])


def find_subject_nifti_files(
    data_root: str,
    folder_paths: list[str],
    subject_id: str,
) -> list[dict]:
    """Look up a single subject's .nii.gz files via the cached folder index."""
    return index_nifti_by_subject(data_root, folder_paths).get(subject_id, [])


def load_correlation_matrix(npz_path: str) -> np.ndarray:
    """
    Load a correlation matrix from an .npz file (uses first key).
    Cached by absolute path — the same matrix can be requested by /trajectory,
    /manifold, /matrix, and the Brain View edge plot in quick succession.
    """
    cached = _MATRIX_CACHE.get(npz_path)
    if cached is not None:
        return cached
    data = np.load(npz_path)
    key = list(data.keys())[0]
    arr = data[key]
    # Cap cache size to avoid memory blowup over a long session.
    with _INDEX_LOCK:
        if len(_MATRIX_CACHE) > 256:
            _MATRIX_CACHE.clear()
        _MATRIX_CACHE[npz_path] = arr
    return arr


def get_subject_trajectory_stream(
    data_root: str,
    folder_paths: list[str],
    subject_id: str,
    prioritize_visit: str | None = None,
):
    """
    For a given subject, find all their .npz correlation matrix files
    across the selected scan folders, compute biomarkers per session,
    and return time-ordered metrics.

    Result is cached by ``(data_root, sorted_folders, subject_id)`` — the
    biomarker computation is the dominant cost of opening a patient and
    is fully deterministic, so caching is safe for the lifetime of the
    process.

    Yields JSON-serializable dictionaries for streaming progress.
    """
    cache_key = (*_index_key(data_root, folder_paths), subject_id)
    with _INDEX_LOCK:
        cached = _TRAJECTORY_CACHE.get(cache_key)
    if cached is not None:
        yield {"type": "complete", "data": cached}
        return

    sessions = []
    files_to_process = find_subject_npz_files(data_root, folder_paths, subject_id)
    if prioritize_visit:
        preferred = prioritize_visit.strip().upper()
        if preferred:
            files_to_process = sorted(
                files_to_process,
                key=lambda rec: (
                    str(rec.get("visit", "")).upper() != preferred,
                    _vkey(rec),
                ),
            )
    total_files = len(files_to_process)

    for idx, rec in enumerate(files_to_process):
        visit = rec["visit"]
        yield {"type": "progress", "visit": visit, "current": idx + 1, "total": total_files}

        try:
            matrix = load_correlation_matrix(rec["abs_path"])
            is_dmn = matrix.shape[0] <= 50
            biomarkers = compute_fmri_biomarkers(matrix, is_dmn_only=is_dmn)
        except Exception as e:
            biomarkers = {"error": str(e)}

        sessions.append({
            "visit": visit,
            "file": rec["rel_path"],
            "filename": rec["filename"],
            **biomarkers,
        })

    result = {
        "subject_id": subject_id,
        "total_sessions": len(sessions),
        "sessions": sorted(sessions, key=_vkey),
    }
    with _INDEX_LOCK:
        _TRAJECTORY_CACHE[cache_key] = result

    yield {"type": "complete", "data": result}
