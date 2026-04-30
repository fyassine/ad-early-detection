"""
biomarkers.py — Compute fMRI-derived biomarkers from correlation matrices.

All metrics are computed purely from the correlation matrix (.npz files).
No clinical scores or timeseries needed.

Metrics:
  - Global FC:  mean functional connectivity (upper triangle of corr matrix)
  - DMN FC:     mean FC within Default Mode Network ROIs
  - Modularity: greedy modularity Q from networkx (thresholded graph)
"""

import os
import re
import math
import numpy as np


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


def get_subject_trajectory(
    data_root: str,
    folder_paths: list[str],
    subject_id: str,
) -> dict:
    """
    For a given subject, find all their .npz correlation matrix files
    across the selected scan folders, compute biomarkers per session,
    and return time-ordered metrics.

    Parameters
    ----------
    data_root : str
        Root data directory.
    folder_paths : list[str]
        Relative paths to scan folders.
    subject_id : str
        Subject pseudonym (hex string without 'sub-' prefix).

    Returns
    -------
    dict with:
      - subject_id
      - sessions: list of dicts, each with visit, file, and biomarker values
    """
    sessions = []

    for folder_rel in folder_paths:
        folder = os.path.join(data_root, folder_rel)
        if not os.path.isdir(folder):
            continue

        for dirpath, _, filenames in os.walk(folder):
            for fn in filenames:
                if not fn.endswith(".npz"):
                    continue
                if f"sub-{subject_id}" not in fn:
                    continue
                # Skip z-transformed variants to avoid double-counting
                if "z_transformed" in fn or "z_transform" in fn:
                    continue

                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, data_root)

                # Extract visit
                visit_match = re.search(r"_(M\d+)_", fn)
                visit = visit_match.group(1) if visit_match else "unknown"

                # Load and compute
                try:
                    data = np.load(full)
                    key = list(data.keys())[0]  # Usually 'array'
                    matrix = data[key]
                    is_dmn = matrix.shape[0] <= 50
                    biomarkers = compute_fmri_biomarkers(matrix, is_dmn_only=is_dmn)
                except Exception as e:
                    biomarkers = {"error": str(e)}

                sessions.append({
                    "visit": visit,
                    "file": rel,
                    "filename": fn,
                    **biomarkers,
                })

    # Sort by visit (M0, M12, M24...)
    def visit_sort_key(s):
        m = re.search(r"M(\d+)", s.get("visit", "M999"))
        return int(m.group(1)) if m else 999

    sessions.sort(key=visit_sort_key)

    return {
        "subject_id": subject_id,
        "total_sessions": len(sessions),
        "sessions": sessions,
    }
