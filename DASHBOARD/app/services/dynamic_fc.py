"""
dynamic_fc.py — Dynamic FC states and dwell times.

Sliding-window dynamic FC analysis (Allen 2014 / Fu 2020 / Diez 2023):
window-correlation matrices are vectorised and clustered with k-means;
dwell times and transition probabilities are estimated from the resulting
state-label sequence per subject.

In the current dashboard, only correlation matrices (.npz) are cached —
raw ROI time-series aren't routinely stored. The full pipeline therefore
requires .nii.gz volumes (loaded on demand) plus an atlas masker.

For v1 we ship a clean degraded path: if a subject only has static
correlation matrices, the service returns ``available=False`` with a
descriptive note. Phase 3+ work can either:
  (a) Precompute time-series during the QC pre-warm pass, or
  (b) Cache windowed FC vectors alongside the static matrix at ingest.

The k-means + dwell-time math is implemented now so the moment time-series
are available the rest of the pipeline lights up.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

try:
    from sklearn.cluster import KMeans  # type: ignore
    _HAS_SKLEARN = True
except Exception:
    KMeans = None  # type: ignore
    _HAS_SKLEARN = False


def _vectorize_upper_tri(corr: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(corr.shape[0], k=1)
    return corr[iu]


def sliding_window_corr(
    timeseries: np.ndarray,
    window: int = 30,
    step: int = 3,
) -> np.ndarray:
    """
    Sliding-window correlation matrices.

    Parameters
    ----------
    timeseries : (T, N) array of T timepoints × N ROIs.
    window : window size in timepoints.
    step : stride.

    Returns
    -------
    Stacked upper-triangle vectors of shape (W, N*(N-1)/2).
    """
    T, N = timeseries.shape
    if T < window + 1 or N < 3:
        return np.empty((0, 0), dtype=np.float32)
    iu = np.triu_indices(N, k=1)
    out: list[np.ndarray] = []
    for start in range(0, T - window + 1, step):
        seg = timeseries[start:start + window]
        # Avoid divide-by-zero on constant ROIs by adding tiny noise.
        sd = seg.std(axis=0)
        seg = seg + np.where(sd < 1e-9, 1e-6, 0.0)
        c = np.corrcoef(seg, rowvar=False)
        out.append(np.nan_to_num(c[iu], nan=0.0).astype(np.float32))
    if not out:
        return np.empty((0, 0), dtype=np.float32)
    return np.stack(out, axis=0)


def fit_dfc_states(
    windowed_vectors: np.ndarray,
    k: int = 4,
    seed: int = 42,
) -> dict:
    """
    K-means clustering of windowed FC vectors → state centroids + labels.
    """
    if not _HAS_SKLEARN:
        return {"available": False, "note": "scikit-learn not installed"}
    if windowed_vectors.size == 0 or windowed_vectors.shape[0] < k:
        return {"available": False, "note": "not enough windows"}
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = km.fit_predict(windowed_vectors)
    return {
        "available": True,
        "centroids": km.cluster_centers_.tolist(),
        "labels": labels.tolist(),
        "k": k,
    }


def dwell_and_transitions(labels: list[int], k: int) -> dict:
    """
    Per-state dwell fraction + k×k transition probability matrix.
    """
    if not labels:
        return {"dwell": {}, "transitions": [[0.0] * k for _ in range(k)]}
    arr = np.asarray(labels, dtype=int)
    counts = np.bincount(arr, minlength=k).astype(np.float32)
    dwell = (counts / max(counts.sum(), 1)).tolist()

    trans = np.zeros((k, k), dtype=np.float32)
    for i in range(len(arr) - 1):
        trans[arr[i], arr[i + 1]] += 1
    row_sums = trans.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    trans = trans / row_sums

    return {
        "dwell": {f"state_{i}": float(d) for i, d in enumerate(dwell)},
        "transitions": trans.tolist(),
    }


# ──────────────────────────────────────────────────────────────────────────── #
# Time-series extraction (cached per-scan T×N parcellation)                    #
# ──────────────────────────────────────────────────────────────────────────── #

_MASKER = None


def _get_schaefer_masker():
    """Lazily build (and cache) the NiftiLabelsMasker used by the processing pipeline."""
    global _MASKER
    if _MASKER is not None:
        return _MASKER
    from nilearn import datasets
    from nilearn.maskers import NiftiLabelsMasker
    schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7)
    _MASKER = NiftiLabelsMasker(labels_img=schaefer.maps, standardize="zscore_sample")
    return _MASKER


def extract_timeseries_from_nii(nii_path: str) -> np.ndarray:
    """Parcellate a 4-D BOLD .nii.gz with the Schaefer-200 atlas → (T, N)."""
    masker = _get_schaefer_masker()
    ts = masker.fit_transform(nii_path)
    return np.asarray(ts, dtype=np.float32)


def load_or_extract_timeseries(
    nii_path: str,
    cache_root,
    subject_id: str,
    visit: str,
) -> Optional[np.ndarray]:
    """Return cached (T, N) array for a subject/visit; parcellate + cache on miss."""
    from pathlib import Path
    cache_dir = Path(cache_root) / "timeseries"
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_visit = str(visit).replace("/", "_")
    cache_path = cache_dir / f"{subject_id}_{safe_visit}.npz"
    if cache_path.exists():
        try:
            return np.load(cache_path)["ts"].astype(np.float32)
        except Exception:
            pass
    try:
        ts = extract_timeseries_from_nii(nii_path)
    except Exception:
        return None
    try:
        np.savez_compressed(cache_path, ts=ts)
    except Exception:
        pass
    return ts


# ──────────────────────────────────────────────────────────────────────────── #
# Cohort-level aggregation                                                     #
# ──────────────────────────────────────────────────────────────────────────── #

def compute_cohort_dfc(
    timeseries_by_cohort: dict[str, list[np.ndarray]],
    k: int = 4,
    window: int = 30,
    step: int = 3,
    seed: int = 42,
) -> dict:
    """Fit a single shared k-means on windowed FC across all subjects, then
    return per-cohort dwell-time histograms.

    Parameters
    ----------
    timeseries_by_cohort : dict cohort_name -> list of (T, N) arrays
    """
    if not _HAS_SKLEARN:
        return {"available": False, "note": "scikit-learn not installed"}

    # 1. Build windowed FC vectors per subject; remember per-subject window count for re-split.
    per_subject_windows: list[tuple[str, np.ndarray]] = []
    for cohort, ts_list in timeseries_by_cohort.items():
        for ts in ts_list:
            if ts is None or ts.ndim != 2 or ts.shape[0] < window + 1:
                continue
            w = sliding_window_corr(ts, window=window, step=step)
            if w.size:
                per_subject_windows.append((cohort, w))
    if not per_subject_windows:
        return {
            "available": False,
            "note": "No subjects have enough time-series data to fit dFC states yet. Run the cohort warmup to parcellate time-series.",
        }

    all_windows = np.concatenate([w for _, w in per_subject_windows], axis=0)
    if all_windows.shape[0] < k:
        return {"available": False, "note": f"Only {all_windows.shape[0]} windows total (need ≥ {k})."}

    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    global_labels = km.fit_predict(all_windows)

    # 2. Per-subject dwell fractions; aggregate per cohort.
    per_cohort: dict[str, dict] = {}
    offset = 0
    for cohort, w in per_subject_windows:
        n = w.shape[0]
        labels = global_labels[offset:offset + n].tolist()
        offset += n
        d = dwell_and_transitions(labels, k=k)
        bucket = per_cohort.setdefault(cohort, {"dwell": [[] for _ in range(k)], "n": 0})
        for i in range(k):
            bucket["dwell"][i].append(d["dwell"][f"state_{i}"])
        bucket["n"] += 1

    cohorts_out: dict[str, dict] = {}
    histograms: dict[str, dict] = {}
    for cohort, b in per_cohort.items():
        means = [float(np.mean(vals)) if vals else 0.0 for vals in b["dwell"]]
        stds  = [float(np.std(vals, ddof=0)) if len(vals) > 1 else 0.0 for vals in b["dwell"]]
        cohorts_out[cohort] = {"n": b["n"], "dwell_mean": means, "dwell_std": stds}
        histograms[cohort] = {
            "n": b["n"],
            "counts": means,  # frontend expects counts; we expose mean dwell fraction
        }

    return {
        "available": True,
        "k": k,
        "window": window,
        "step": step,
        "centroids": km.cluster_centers_.tolist(),
        "n_windows_total": int(all_windows.shape[0]),
        "cohorts": list(cohorts_out.keys()),
        "per_cohort": cohorts_out,
        "histograms": histograms,
        "state_labels": [f"State {i + 1}" for i in range(k)],
    }


def subject_dynamic_fc(
    timeseries: Optional[np.ndarray],
    k: int = 4,
    window: int = 30,
    step: int = 3,
) -> dict:
    """
    End-to-end per-subject dynamic FC analysis.

    Returns ``available=False`` when timeseries are missing or too short.
    Callers should not error on this — the cohort dashboard renders a
    "dFC requires raw timeseries (.nii.gz) — none cached" placeholder.
    """
    if timeseries is None or not isinstance(timeseries, np.ndarray):
        return {
            "available": False,
            "note": "Dynamic FC requires raw ROI time-series; only static "
                    "correlation matrices are currently cached.",
        }
    if timeseries.ndim != 2 or timeseries.shape[0] < window + 1:
        return {
            "available": False,
            "note": f"Need at least {window + 1} timepoints (got {timeseries.shape[0] if timeseries.ndim == 2 else 'invalid shape'}).",
        }
    windowed = sliding_window_corr(timeseries, window=window, step=step)
    if windowed.size == 0:
        return {"available": False, "note": "no usable sliding windows"}
    fit = fit_dfc_states(windowed, k=k)
    if not fit.get("available"):
        return fit
    extra = dwell_and_transitions(fit["labels"], k=k)
    return {
        "available": True,
        "k": k,
        "window": window,
        "step": step,
        "n_windows": int(windowed.shape[0]),
        "labels": fit["labels"],
        **extra,
    }
