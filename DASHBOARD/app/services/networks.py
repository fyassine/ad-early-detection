"""
networks.py — Schaefer 7-network FC + system segregation index.

System segregation (Chan et al. 2014; revisited in Setton et al. 2023 Nature
Aging): one of the strongest published static-FC predictors of cognitive
decline in lifespan studies. Computed as

    seg = (W - B) / W

where W = mean within-network FC and B = mean between-network FC.

This module also exposes per-network mean FC for each Schaefer 7-network
(Default, Cont, SalVentAttn, DorsAttn, Limbic, SomMot, Vis), used to back
the small-multiples chart on the Patient Overview tab.
"""

from __future__ import annotations

import json
import math
from typing import Optional

import numpy as np

from ..config import STATIC_DIR

# Cache for loaded Schaefer atlas labels — small JSON, never invalidated at
# runtime (atlas doesn't change while the server is up).
_NETWORK_INDICES_CACHE: dict[int, dict[str, list[int]]] = {}


def _safe_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if not math.isfinite(f) else f


def get_schaefer_network_indices(n_parcels: int = 200) -> dict[str, list[int]]:
    """
    Load Schaefer N-parcel ROI -> network mapping from the static atlas JSON
    shipped at ``app/static/data/schaefer_{N}_coords.json``. Returns a dict
    mapping each Schaefer 7-network name to the list of ROI indices that
    belong to it.

    Returns an empty dict if the JSON is missing.
    """
    cached = _NETWORK_INDICES_CACHE.get(n_parcels)
    if cached is not None:
        return cached

    coord_file = STATIC_DIR / "data" / f"schaefer_{n_parcels}_coords.json"
    if not coord_file.exists():
        return {}

    try:
        with coord_file.open("r") as f:
            data = json.load(f)
    except Exception:
        return {}

    rois = data.get("rois") or []
    out: dict[str, list[int]] = {}
    for i, r in enumerate(rois):
        net = r.get("network")
        if not net:
            continue
        out.setdefault(net, []).append(i)

    _NETWORK_INDICES_CACHE[n_parcels] = out
    return out


def per_network_fc(corr_matrix: np.ndarray, n_parcels: int = 200) -> dict[str, float]:
    """
    Per-network mean within-network FC for each Schaefer 7-network.

    Returns ``{network_name: within_network_mean_fc}``. Networks with fewer
    than 2 ROIs in the matrix are dropped silently.
    """
    nets = get_schaefer_network_indices(n_parcels)
    if not nets:
        return {}

    n = corr_matrix.shape[0]
    out: dict[str, float] = {}
    for name, idxs in nets.items():
        idxs = [i for i in idxs if i < n]
        if len(idxs) < 2:
            continue
        sub = corr_matrix[np.ix_(idxs, idxs)]
        tri = np.triu_indices(len(idxs), k=1)
        out[name] = _safe_float(np.nanmean(sub[tri]))
    return out


def system_segregation(corr_matrix: np.ndarray, n_parcels: int = 200) -> Optional[float]:
    """
    System segregation index (Chan 2014; Setton 2023 Nature Aging) computed
    over the Schaefer 7-network parcellation.

        seg = (W - B) / W

    where W is the mean within-network correlation across networks and B is
    the mean between-network correlation. Returns None if no network labels
    are loaded or the matrix is too small.
    """
    nets = get_schaefer_network_indices(n_parcels)
    if not nets:
        return None
    n = corr_matrix.shape[0]

    # All ROI indices that actually live in the matrix, grouped by network.
    grouped: dict[str, list[int]] = {}
    for name, idxs in nets.items():
        valid = [i for i in idxs if i < n]
        if len(valid) >= 2:
            grouped[name] = valid
    if len(grouped) < 2:
        return None

    within_vals: list[float] = []
    between_vals: list[float] = []
    names = list(grouped.keys())
    for i, ni in enumerate(names):
        idxi = grouped[ni]
        sub = corr_matrix[np.ix_(idxi, idxi)]
        tri = np.triu_indices(len(idxi), k=1)
        within_vals.append(float(np.nanmean(sub[tri])))
        for nj in names[i + 1:]:
            idxj = grouped[nj]
            block = corr_matrix[np.ix_(idxi, idxj)]
            between_vals.append(float(np.nanmean(block)))

    if not within_vals or not between_vals:
        return None
    W = float(np.mean(within_vals))
    B = float(np.mean(between_vals))
    if abs(W) < 1e-9:
        return None
    return _safe_float((W - B) / W)
