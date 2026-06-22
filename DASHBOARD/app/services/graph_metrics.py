"""
graph_metrics.py — Graph-theoretic topology metrics for FC networks.

Implements the network-science measures the Phase 3 Cohort and Patient
tiers visualise:

  - small-worldness        σ = (C/C_rand) / (L/L_rand)   (Humphries 2008)
  - modularity Q           Louvain greedy (already computed in biomarkers.py;
                           re-exposed here for completeness)
  - clustering coefficient mean local clustering
  - characteristic path length
  - global efficiency
  - DomiRank centrality    (Engsig 2024 — dominance-driven centrality)
  - top-k hub ROIs         (by DomiRank)

All measures operate on a density-thresholded binary graph (default top-20%
edges by |r|) to control for differences in global FC across subjects.
This matches the Sanz-Arigita 2010 / Schultz 2013 / Li 2022 / DomiRank
2025 papers cited in the planning doc.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import networkx as nx  # type: ignore
    _HAS_NX = True
except Exception:
    nx = None  # type: ignore
    _HAS_NX = False


# --------------------------------------------------------------------------- #
# Adjacency construction                                                       #
# --------------------------------------------------------------------------- #

def _threshold_to_binary(corr: np.ndarray, density: float = 0.20) -> np.ndarray:
    """
    Density-thresholded binary adjacency. Keeps the top ``density`` fraction
    of edges by absolute correlation. Diagonal removed. Symmetric.
    """
    n = corr.shape[0]
    if n < 3:
        return np.zeros_like(corr, dtype=np.uint8)
    iu = np.triu_indices(n, k=1)
    weights = np.abs(corr[iu])
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    n_keep = max(1, int(round(density * weights.size)))
    if weights.size <= n_keep:
        thr = 0.0
    else:
        thr = float(np.partition(weights, -n_keep)[-n_keep])
    adj = (np.abs(corr) >= thr).astype(np.uint8)
    np.fill_diagonal(adj, 0)
    adj = ((adj + adj.T) > 0).astype(np.uint8)
    return adj


def _to_graph(adj: np.ndarray):
    if not _HAS_NX:
        return None
    return nx.from_numpy_array(adj)


# --------------------------------------------------------------------------- #
# Per-subject scalar metrics                                                   #
# --------------------------------------------------------------------------- #

def _safe_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if not math.isfinite(f) else f


def clustering_coefficient(adj: np.ndarray) -> Optional[float]:
    if not _HAS_NX:
        return None
    g = _to_graph(adj)
    try:
        return _safe_float(nx.average_clustering(g))
    except Exception:
        return None


def characteristic_path_length(adj: np.ndarray) -> Optional[float]:
    if not _HAS_NX:
        return None
    g = _to_graph(adj)
    try:
        if g.number_of_nodes() == 0:
            return None
        if not nx.is_connected(g):
            comps = list(nx.connected_components(g))
            largest = g.subgraph(max(comps, key=len))
            if largest.number_of_nodes() < 2:
                return None
            return _safe_float(nx.average_shortest_path_length(largest))
        return _safe_float(nx.average_shortest_path_length(g))
    except Exception:
        return None


def global_efficiency(adj: np.ndarray) -> Optional[float]:
    if not _HAS_NX:
        return None
    try:
        g = _to_graph(adj)
        return _safe_float(nx.global_efficiency(g))
    except Exception:
        return None


def small_worldness(
    adj: np.ndarray,
    n_random: int = 5,
    seed: int = 42,
    preserve_degree: bool = False,
) -> Optional[float]:
    """
    Humphries & Gurney (2008) σ = (C/C_rand) / (L/L_rand).

    By default uses Erdős–Rényi references matched on edge density
    (``nx.fast_gnp_random_graph``) — fast and standard in the small-world
    literature (Watts & Strogatz 1998). Set ``preserve_degree=True`` to use
    edge-preserving rewiring (``nx.random_reference``), which is more
    rigorous but ~100× slower and prone to long rejection loops on graphs
    with disconnected components.
    """
    if not _HAS_NX:
        return None
    g = _to_graph(adj)
    try:
        if g.number_of_edges() < 5 or g.number_of_nodes() < 5:
            return None
        c = nx.average_clustering(g)
        if not nx.is_connected(g):
            comps = list(nx.connected_components(g))
            largest = g.subgraph(max(comps, key=len))
            if largest.number_of_nodes() < 5:
                return None
            l = nx.average_shortest_path_length(largest)
        else:
            l = nx.average_shortest_path_length(g)

        n_nodes = g.number_of_nodes()
        n_edges = g.number_of_edges()
        p = (2 * n_edges) / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0

        rng = np.random.default_rng(seed)
        c_r = []
        l_r = []
        for _ in range(n_random):
            seed_i = int(rng.integers(0, 2**31 - 1))
            try:
                if preserve_degree:
                    gr = nx.random_reference(g, niter=2, seed=seed_i)
                else:
                    gr = nx.fast_gnp_random_graph(n_nodes, p, seed=seed_i)
            except Exception:
                continue
            c_r.append(nx.average_clustering(gr))
            try:
                if nx.is_connected(gr):
                    l_r.append(nx.average_shortest_path_length(gr))
                else:
                    comps = list(nx.connected_components(gr))
                    sub = gr.subgraph(max(comps, key=len))
                    if sub.number_of_nodes() > 1:
                        l_r.append(nx.average_shortest_path_length(sub))
            except Exception:
                continue
        if not c_r or not l_r:
            return None
        c_rand = float(np.median(c_r))
        l_rand = float(np.median(l_r))
        if c_rand < 1e-9 or l_rand < 1e-9 or l < 1e-9:
            return None
        return _safe_float((c / c_rand) / (l / l_rand))
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# DomiRank centrality (Engsig 2024)                                            #
# --------------------------------------------------------------------------- #

def domirank(adj: np.ndarray, alpha: float = 0.95, max_iter: int = 200,
             tol: float = 1e-6) -> Optional[np.ndarray]:
    """
    DomiRank centrality (Engsig, Bondia-Carrasco & Garas, *Nat. Commun.*
    2024). Defined by the fixed-point iteration

        γ_{k+1} = α · A · (1 − γ_k) + (1 − α) · 1

    where ``A`` is the binary adjacency matrix and ``γ ∈ [0, 1]``.
    Converges to a "competitive" centrality that penalises nodes whose
    neighbours are themselves dominant.

    Returns a length-N vector, or None on failure.
    """
    try:
        A = np.asarray(adj, dtype=np.float32)
        n = A.shape[0]
        if n < 2:
            return None
        ones = np.ones(n, dtype=np.float32)
        gamma = np.full(n, 0.5, dtype=np.float32)
        for _ in range(max_iter):
            new = alpha * A.dot(ones - gamma) + (1 - alpha)
            new = np.clip(new, 0.0, 1.0)
            if np.linalg.norm(new - gamma) < tol:
                gamma = new
                break
            gamma = new
        # Normalise to [0, 1] for comparable cohort-wise distributions.
        gmin, gmax = float(gamma.min()), float(gamma.max())
        if gmax - gmin < 1e-9:
            return gamma
        return (gamma - gmin) / (gmax - gmin)
    except Exception:
        return None


def top_k_hubs(rank_vec: np.ndarray, k: int = 10) -> list[dict]:
    """Return the top-k ROI indices by ``rank_vec`` value, descending."""
    if rank_vec is None:
        return []
    arr = np.asarray(rank_vec)
    if arr.size == 0:
        return []
    k = min(k, arr.size)
    idx = np.argsort(-arr)[:k]
    return [{"roi": int(i), "score": _safe_float(float(arr[i]))} for i in idx]


# --------------------------------------------------------------------------- #
# Aggregate per-subject                                                        #
# --------------------------------------------------------------------------- #

def subject_graph_metrics(
    corr: np.ndarray,
    density: float = 0.20,
    compute_hubs: bool = True,
    k_hubs: int = 10,
) -> dict:
    """
    Compute all per-subject graph metrics for a single correlation matrix.

    Heavy: small-worldness samples 5 random reference graphs. Budget ~0.5s
    per Schaefer-200 subject on CPU. Callers should parallelise via
    ThreadPoolExecutor when running over a whole cohort.
    """
    out: dict = {
        "density": density,
        "available": _HAS_NX,
        "small_worldness": None,
        "clustering": None,
        "path_length": None,
        "global_efficiency": None,
        "n_nodes": int(corr.shape[0]),
        "n_edges": None,
    }
    if not _HAS_NX:
        out["note"] = "networkx not installed"
        return out
    if corr is None or corr.size == 0 or corr.shape[0] != corr.shape[1]:
        out["note"] = "invalid correlation matrix"
        return out

    adj = _threshold_to_binary(corr, density=density)
    out["n_edges"] = int(adj.sum() // 2)

    out["clustering"] = clustering_coefficient(adj)
    out["path_length"] = characteristic_path_length(adj)
    out["global_efficiency"] = global_efficiency(adj)
    out["small_worldness"] = small_worldness(adj)
    if compute_hubs:
        dr = domirank(adj)
        out["domirank_top_k"] = top_k_hubs(dr, k=k_hubs)
    return out


# --------------------------------------------------------------------------- #
# Disk cache for cohort-level graph metrics                                   #
# --------------------------------------------------------------------------- #

def _gm_cache_key(csv_path: str, scan_folders: list[str]) -> str:
    h = hashlib.sha1()
    h.update(csv_path.encode("utf-8"))
    for f in sorted(scan_folders):
        h.update(b"\x00"); h.update(f.encode("utf-8"))
    return h.hexdigest()[:20]


def _gm_cache_path(cache_root: Path, csv_path: str,
                   scan_folders: list[str], density: float) -> Path:
    key = _gm_cache_key(csv_path, scan_folders)
    d = int(density * 100)
    return cache_root / "graph_metrics" / f"graph_metrics_{key}_density{d:02d}.json"


def load_graph_metrics_cache(
    cache_root: Path,
    csv_path: str,
    scan_folders: list[str],
    density: float,
) -> Optional[dict]:
    """Return the cached graph-metrics payload, or None on miss."""
    path = _gm_cache_path(cache_root, csv_path, scan_folders, density)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def save_graph_metrics_cache(
    cache_root: Path,
    csv_path: str,
    scan_folders: list[str],
    density: float,
    data: dict,
) -> None:
    """Write graph-metrics payload to disk. Best-effort; errors are silenced."""
    path = _gm_cache_path(cache_root, csv_path, scan_folders, density)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(path)
    except Exception:
        pass
