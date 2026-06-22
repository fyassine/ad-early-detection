"""
common/pipeline_trace.py — trace one subject's raw scan → FC matrix → graph.

Powers the "data journey" section of ``EXPLAIN_COMMON_DELCODE.ipynb``: for a single
subject/visit it reconstructs every preprocessing stage so each can be visualised —
the raw BOLD volume, the Schaefer-200 timeseries, the 200×200 functional-connectivity
matrix, and the kNN brain graph fed to the GAAE encoder.

Model-agnostic (no model imports, per ``.claude/rules/architecture.md``): the
``.nii → timeseries → FC`` step reuses the documented preprocessing
(``DATA/src/processing/process_using_schaeffer_atlas.py``), imported lazily so this
module stays cheap to import and nilearn is only required when a raw scan is actually
processed. The ``FC → PyG Data`` step uses an inlined kNN adjacency identical to
``model/GAAE/utils.py::knn_binary_adjacency_matrix_no_diag`` (kept here so ``common``
imports no model code).

If the raw ``.nii`` is unavailable, ``nii_to_fc_to_graph`` falls back to the stored
``.npz`` FC matrix (``bold_img`` / ``timeseries`` are then ``None``) so the trace
still completes.
"""
from __future__ import annotations

import glob
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FMRI_ROOT = _REPO_ROOT / "DATA" / "DELCODE" / "__fmri_wholebrain_sch200_flat__" / "fmri"
_MATRICES = _REPO_ROOT / "DATA" / "DELCODE" / "__fc_wholebrain_sch200_flat__" / "matrices"

_VARIANT_SUFFIX = {
    "z_transformed": "_whole_brain_correlation_matrix_z_transformed.npz",
    "raw": "_whole_brain_correlation_matrix.npz",
}


# --------------------------------------------------------------------------- #
# kNN adjacency (inlined copy of model/GAAE/utils.knn_binary_adjacency_matrix_no_diag)
# --------------------------------------------------------------------------- #
def knn_binary_adjacency(corr_matrix: np.ndarray, k: int) -> np.ndarray:
    """Symmetric binary kNN adjacency from |correlation|, no self-loops."""
    corr = np.asarray(corr_matrix, dtype=float).copy()
    n = corr.shape[0]
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        row = corr[i].copy()
        row[i] = -np.inf
        nearest = np.argsort(-row)[:k]
        adj[i, nearest] = 1.0
    return np.maximum(adj, adj.T)


# --------------------------------------------------------------------------- #
# File discovery
# --------------------------------------------------------------------------- #
def find_fc_matrix(
    subject_id: str,
    visit_month: int,
    *,
    matrices_dir: str | Path = _MATRICES,
    file_variant: str = "z_transformed",
) -> Optional[Path]:
    """Locate the stored FC ``.npz`` for one subject/visit, or None."""
    suffix = _VARIANT_SUFFIX.get(file_variant, _VARIANT_SUFFIX["z_transformed"])
    hits = glob.glob(os.path.join(str(matrices_dir), f"sub-{subject_id}_*{suffix}"))
    for f in hits:
        m = re.search(r"_(M\d+)_", os.path.basename(f))
        if m and int(m.group(1)[1:]) == int(visit_month):
            return Path(f)
    return None


def find_nii(
    subject_id: str,
    visit_month: int,
    *,
    fmri_root: str | Path = _FMRI_ROOT,
) -> Optional[Path]:
    """Locate the raw rest-BOLD ``.nii(.gz)`` for one subject/visit, or None."""
    sub_dir = Path(fmri_root) / f"sub-{subject_id}"
    if not sub_dir.is_dir():
        return None
    for f in sorted(sub_dir.iterdir()):
        name = f.name.lower()
        if not (name.endswith(".nii") or name.endswith(".nii.gz")):
            continue
        if "bold" not in name or "task-rest" not in name:
            continue
        if f"_m{int(visit_month)}_" in name:
            return f
    return None


# --------------------------------------------------------------------------- #
# Trace
# --------------------------------------------------------------------------- #
def nii_to_fc_to_graph(
    subject_id: str,
    visit_month: int,
    *,
    fmri_root: str | Path = _FMRI_ROOT,
    matrices_dir: str | Path = _MATRICES,
    file_variant: str = "z_transformed",
    adjacency_k: int = 8,
) -> Dict[str, Any]:
    """Reconstruct every preprocessing stage for one subject/visit.

    Returns a dict with (any of which may be ``None`` when the raw scan is absent):
    ``nii_path``, ``bold_img`` (nibabel image), ``timeseries`` (T, 200),
    ``fc_matrix`` (200, 200 Pearson), ``z_fc`` (200, 200 Fisher-z), ``adjacency``
    (200, 200 binary), ``edge_index`` (2, E numpy), ``x`` (200, 200 node features),
    and ``source`` ("nii" or "fc_npz").
    """
    out: Dict[str, Any] = {
        "subject_id": subject_id, "visit_month": int(visit_month),
        "nii_path": None, "bold_img": None, "timeseries": None,
        "fc_matrix": None, "z_fc": None, "source": None,
    }

    nii = find_nii(subject_id, visit_month, fmri_root=fmri_root)
    if nii is not None:
        try:
            import nibabel as nib  # type: ignore
            from nilearn.connectome import ConnectivityMeasure  # type: ignore

            from DATA.src.processing.process_using_schaeffer_atlas import (
                build_masker,
                compute_connectivity_matrices,
            )

            masker = build_masker()
            measure = ConnectivityMeasure(kind="correlation", standardize="zscore_sample")
            corr, zmat = compute_connectivity_matrices(nii, masker, measure)
            out["nii_path"] = str(nii)
            out["bold_img"] = nib.load(str(nii))
            out["timeseries"] = masker.transform(str(nii))
            out["fc_matrix"] = corr
            out["z_fc"] = zmat
            out["source"] = "nii"
        except Exception as exc:  # nilearn/nibabel hiccup → fall back to stored FC
            print(f"[pipeline_trace] raw-scan processing failed ({exc}); using stored FC.")

    if out["fc_matrix"] is None:
        fc_path = find_fc_matrix(subject_id, visit_month, matrices_dir=matrices_dir,
                                 file_variant=file_variant)
        if fc_path is None:
            raise FileNotFoundError(
                f"No raw .nii and no stored FC .npz for subject {subject_id!r} "
                f"visit M{visit_month}. Checked {fmri_root} and {matrices_dir}."
            )
        arr = np.nan_to_num(np.load(fc_path)["array"], nan=0.0, posinf=0.0, neginf=0.0)
        out["z_fc" if "z_transformed" in file_variant else "fc_matrix"] = arr
        out["source"] = "fc_npz"
        if out["fc_matrix"] is None:
            out["fc_matrix"] = arr  # display fallback

    feat = out["z_fc"] if out["z_fc"] is not None else out["fc_matrix"]
    feat = np.nan_to_num(np.asarray(feat, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    adj = knn_binary_adjacency(np.abs(feat), adjacency_k)
    rows, cols = np.nonzero(adj)
    out["x"] = feat
    out["adjacency"] = adj
    out["edge_index"] = np.vstack([rows, cols]).astype(np.int64)
    return out


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_brain_slice(bold_img: Any, *, title: str = "Mean BOLD") -> Any:
    """Glass/anatomical render of the mean BOLD volume (nilearn ``plot_epi``)."""
    import matplotlib.pyplot as plt
    from nilearn import image as nlimg  # type: ignore
    from nilearn import plotting as nlplt  # type: ignore

    mean_img = nlimg.mean_img(bold_img)
    fig = plt.figure(figsize=(10, 3.5))
    nlplt.plot_epi(mean_img, figure=fig, title=title, display_mode="ortho",
                   cmap="gray", colorbar=True)
    return fig


def plot_fc_heatmap(fc_matrix: np.ndarray, z_fc: Optional[np.ndarray] = None) -> Any:
    """Heatmap(s) of the FC matrix (Pearson, and Fisher-z if provided)."""
    import matplotlib.pyplot as plt

    mats = [("Pearson FC", fc_matrix)]
    if z_fc is not None:
        mats.append(("Fisher-z FC", z_fc))
    fig, axes = plt.subplots(1, len(mats), figsize=(6.2 * len(mats), 5.4))
    if len(mats) == 1:
        axes = [axes]
    for ax, (name, mat) in zip(axes, mats):
        mat = np.asarray(mat, dtype=float)
        vmax = float(np.nanpercentile(np.abs(mat), 99)) or 1.0
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
        ax.set_title(f"{name}  {mat.shape[0]}×{mat.shape[1]}")
        ax.set_xlabel("ROI")
        ax.set_ylabel("ROI")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_brain_graph(
    edge_index: np.ndarray,
    atlas: Sequence[Dict[str, Any]],
    *,
    node_values: Optional[np.ndarray] = None,
    title: str = "kNN brain graph",
    max_edges: int = 1500,
) -> Any:
    """Draw the kNN graph with nodes at their MNI (x, y) coordinates (networkx)."""
    import matplotlib.pyplot as plt
    import networkx as nx

    from CLASSIFIER.common.explain import atlas_coords, atlas_networks, NETWORK_COLORS

    coords = atlas_coords(atlas)
    networks = atlas_networks(atlas)
    pos = {i: (coords[i, 0], coords[i, 1]) for i in range(len(atlas))}

    G = nx.Graph()
    G.add_nodes_from(range(len(atlas)))
    ei = np.asarray(edge_index)
    edges = list(zip(ei[0].tolist(), ei[1].tolist()))
    if len(edges) > max_edges:  # subsample purely for legibility
        step = max(1, len(edges) // max_edges)
        edges = edges[::step]
    G.add_edges_from(edges)

    if node_values is not None:
        nv = np.abs(np.asarray(node_values, dtype=float))
        nv = nv / (nv.max() or 1.0)
        node_color = nv
        cmap = "autumn_r"
    else:
        node_color = [NETWORK_COLORS.get(n, "#888888") for n in networks]
        cmap = None

    fig, ax = plt.subplots(figsize=(8, 7))
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.12, width=0.5)
    nodes = nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=45, node_color=node_color, cmap=cmap, linewidths=0.3,
        edgecolors="white",
    )
    if node_values is not None and nodes is not None:
        fig.colorbar(nodes, ax=ax, fraction=0.046, pad=0.04, label="importance")
    ax.set_title(f"{title}  ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges shown)")
    ax.set_xlabel("MNI x")
    ax.set_ylabel("MNI y")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


__all__ = [
    "knn_binary_adjacency",
    "find_fc_matrix",
    "find_nii",
    "nii_to_fc_to_graph",
    "plot_brain_slice",
    "plot_fc_heatmap",
    "plot_brain_graph",
]
