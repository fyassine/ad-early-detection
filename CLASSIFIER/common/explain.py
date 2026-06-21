"""
common/explain.py — model-agnostic explainability helpers for the EXPLAIN notebook.

Pure utilities used by ``notebooks/EXPLAIN/EXPLAIN_COMMON_DELCODE.ipynb`` and the
explain adapters (``adapters/explain.py``). This module is deliberately
**model-agnostic**: it imports no model code (per ``.claude/rules/architecture.md``,
``common/**`` has no model-specific imports). It covers

  * the Schaefer-200 atlas (region coordinates + network labels),
  * 2-D embedding of a latent matrix (UMAP / PCA / t-SNE),
  * the shared latent-space scatter (cohort colours + centroids + conversion axis),
  * region-importance renders (nilearn glass-brain markers + a per-network bar chart),
  * figure saving (png + pdf) into a run dir.

Heavy optional dependencies (``umap``, ``nilearn``) are imported lazily inside the
functions that need them, so importing this module is cheap and never fails when an
optional backend is absent — the caller gets a clear error only if it actually uses
that path.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# The Schaefer-200 coordinate table lives with the dashboard's static assets; it is
# the single source of truth for ROI index → (network, hemisphere, MNI xyz).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ATLAS_JSON = _REPO_ROOT / "DASHBOARD" / "app" / "static" / "data" / "schaefer_200_coords.json"

# Schaefer 7-network palette (stable colours across every figure).
NETWORK_ORDER: List[str] = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
NETWORK_COLORS: Dict[str, str] = {
    "Vis": "#781286",
    "SomMot": "#4682B4",
    "DorsAttn": "#00760E",
    "SalVentAttn": "#C43AFA",
    "Limbic": "#DCF8A4",
    "Cont": "#E69422",
    "Default": "#CD3E4E",
}


# --------------------------------------------------------------------------- #
# Atlas
# --------------------------------------------------------------------------- #
def load_schaefer_atlas(path: str | Path | None = None) -> List[Dict[str, Any]]:
    """Return the 200 Schaefer ROIs as a list of dicts (index/label/network/hemisphere/xyz).

    Each entry: ``{"index", "label", "network", "hemisphere", "x_mni", "y_mni", "z_mni"}``.
    Sorted by ``index`` so positional indexing matches the FC matrix / node order.
    """
    atlas_path = Path(path) if path is not None else _ATLAS_JSON
    if not atlas_path.is_file():
        raise FileNotFoundError(
            f"Schaefer atlas coordinates not found at {atlas_path}. Expected the "
            "dashboard asset DASHBOARD/app/static/data/schaefer_200_coords.json."
        )
    data = json.loads(atlas_path.read_text())
    rois = data["rois"] if isinstance(data, dict) else data
    return sorted(rois, key=lambda r: int(r["index"]))


def atlas_coords(atlas: Sequence[Dict[str, Any]]) -> np.ndarray:
    """(N, 3) MNI coordinate array in ROI-index order."""
    return np.array([[r["x_mni"], r["y_mni"], r["z_mni"]] for r in atlas], dtype=float)


def atlas_networks(atlas: Sequence[Dict[str, Any]]) -> List[str]:
    """Per-ROI 7-network label in ROI-index order."""
    return [str(r["network"]) for r in atlas]


# --------------------------------------------------------------------------- #
# Latent-space embedding + plot
# --------------------------------------------------------------------------- #
def embed_2d(
    X: np.ndarray,
    method: str = "umap",
    *,
    seed: int = 42,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> np.ndarray:
    """Reduce ``X`` (N, D) to (N, 2) with a deterministic UMAP / PCA / t-SNE.

    ``method`` is one of ``"umap" | "pca" | "tsne"``. UMAP is imported lazily; PCA
    and t-SNE come from scikit-learn. Always seeded for reproducibility.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"embed_2d expects a 2-D array, got shape {X.shape}.")
    if X.shape[0] < 3:
        raise ValueError(f"embed_2d needs at least 3 rows, got {X.shape[0]}.")
    method = method.lower()
    if method == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=seed).fit_transform(X)
    if method == "tsne":
        from sklearn.manifold import TSNE

        perplexity = min(30, max(5, (X.shape[0] - 1) // 3))
        return TSNE(n_components=2, random_state=seed, perplexity=perplexity,
                    init="pca").fit_transform(X)
    if method == "umap":
        try:
            import umap  # type: ignore
        except ImportError as exc:  # pragma: no cover - env without umap
            raise ImportError(
                "embed_2d(method='umap') needs umap-learn. Install it "
                "(pip install -r CLASSIFIER/requirements-explain.txt) or use method='pca'."
            ) from exc
        n_neighbors = int(min(n_neighbors, max(2, X.shape[0] - 1)))
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                            random_state=seed)
        return reducer.fit_transform(X)
    raise ValueError(f"Unknown embed_2d method {method!r}; expected umap | pca | tsne.")


def plot_latent_space(
    emb2d: np.ndarray,
    labels: Sequence[int],
    *,
    title: str = "",
    method_name: str = "UMAP",
    centroids: bool = True,
    conversion_axis: bool = True,
) -> Any:
    """Scatter a 2-D latent embedding coloured by cohort, with optional centroids/axis.

    ``labels`` is 1 = converter, 0 = stable MCI. When ``conversion_axis`` is set and
    both cohorts are present, draws the arrow from the stable centroid to the
    converter centroid — the data-driven "conversion direction" in the 2-D map.
    Returns the matplotlib ``Figure``.
    """
    import matplotlib.pyplot as plt

    emb2d = np.asarray(emb2d, dtype=float)
    labels = np.asarray(labels, dtype=int)
    fig, ax = plt.subplots(figsize=(7, 6))
    palette = {1: "#F44336", 0: "#2196F3"}
    names = {1: "converter", 0: "stable MCI"}
    for lab in (0, 1):
        m = labels == lab
        if m.any():
            ax.scatter(emb2d[m, 0], emb2d[m, 1], s=28, alpha=0.7,
                       color=palette[lab], label=f"{names[lab]} (n={int(m.sum())})",
                       edgecolors="white", linewidths=0.4)

    cents: Dict[int, np.ndarray] = {}
    if centroids:
        for lab in (0, 1):
            m = labels == lab
            if m.any():
                c = emb2d[m].mean(0)
                cents[lab] = c
                ax.scatter(*c, s=320, marker="X", color=palette[lab],
                           edgecolors="black", linewidths=1.3, zorder=5)

    if conversion_axis and 0 in cents and 1 in cents:
        c0, c1 = cents[0], cents[1]
        ax.annotate("", xy=tuple(c1), xytext=tuple(c0),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=2.0))
        ax.text(*(0.5 * (c0 + c1)), "  conversion axis", fontsize=9, style="italic")

    ax.set_xlabel(f"{method_name}-1")
    ax.set_ylabel(f"{method_name}-2")
    ax.set_title(title or f"Latent space ({method_name})")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Region importance
# --------------------------------------------------------------------------- #
def _normalize_importance(values: np.ndarray) -> np.ndarray:
    v = np.abs(np.asarray(values, dtype=float))
    vmax = float(v.max()) if v.size and v.max() > 0 else 1.0
    return v / vmax


def plot_region_importance_glassbrain(
    values: np.ndarray,
    atlas: Sequence[Dict[str, Any]],
    *,
    title: str = "Region importance",
    top_k: int = 30,
    cmap: str = "autumn_r",
) -> Any:
    """Glass-brain markers sized/coloured by per-ROI importance (nilearn).

    ``values`` is a length-200 array of per-region scores (any sign; magnitude is
    used). Only the ``top_k`` most important ROIs are drawn to keep the render
    legible. Returns the matplotlib ``Figure``.
    """
    import matplotlib.pyplot as plt

    try:
        from nilearn import plotting as nlplt  # type: ignore
    except ImportError as exc:  # pragma: no cover - env without nilearn
        raise ImportError(
            "plot_region_importance_glassbrain needs nilearn. Install it "
            "(pip install -r CLASSIFIER/requirements-explain.txt)."
        ) from exc

    coords = atlas_coords(atlas)
    mag = _normalize_importance(values)
    if coords.shape[0] != mag.shape[0]:
        raise ValueError(
            f"values length {mag.shape[0]} != atlas size {coords.shape[0]}."
        )
    keep = np.argsort(mag)[::-1][: int(top_k)]
    fig = plt.figure(figsize=(11, 4))
    nlplt.plot_markers(
        node_values=mag[keep],
        node_coords=coords[keep],
        node_size="auto",
        node_cmap=cmap,
        node_vmin=0.0,
        node_vmax=1.0,
        display_mode="lyrz",
        title=title,
        figure=fig,
        colorbar=True,
    )
    return fig


def plot_region_importance_bars(
    values: np.ndarray,
    atlas: Sequence[Dict[str, Any]],
    *,
    title: str = "Importance by network",
    top_k: int = 20,
) -> Any:
    """Two-panel bar chart: top-K ROIs (left) and mean importance per network (right)."""
    import matplotlib.pyplot as plt

    mag = _normalize_importance(values)
    networks = atlas_networks(atlas)
    labels = [str(r["label"]).replace("7Networks_", "") for r in atlas]
    if len(networks) != mag.shape[0]:
        raise ValueError(f"values length {mag.shape[0]} != atlas size {len(networks)}.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    order = np.argsort(mag)[::-1][: int(top_k)]
    ax = axes[0]
    bar_colors = [NETWORK_COLORS.get(networks[i], "#888888") for i in order]
    ax.barh(range(len(order)), mag[order][::-1], color=bar_colors[::-1])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([labels[i] for i in order][::-1], fontsize=7)
    ax.set_xlabel("normalised importance")
    ax.set_title(f"Top {len(order)} ROIs")
    ax.grid(alpha=0.25, axis="x")

    ax = axes[1]
    per_net = []
    present = [n for n in NETWORK_ORDER if n in set(networks)]
    for n in present:
        idx = [i for i, nn in enumerate(networks) if nn == n]
        per_net.append(float(np.mean(mag[idx])) if idx else 0.0)
    ax.bar(range(len(present)), per_net,
           color=[NETWORK_COLORS.get(n, "#888888") for n in present])
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(present, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("mean normalised importance")
    ax.set_title("Mean importance per network")
    ax.grid(alpha=0.25, axis="y")

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def network_importance_summary(
    values: np.ndarray, atlas: Sequence[Dict[str, Any]]
) -> Dict[str, float]:
    """Mean normalised importance per 7-network (for the JSON summary)."""
    mag = _normalize_importance(values)
    networks = atlas_networks(atlas)
    out: Dict[str, float] = {}
    for n in NETWORK_ORDER:
        idx = [i for i, nn in enumerate(networks) if nn == n]
        if idx:
            out[n] = float(np.mean(mag[idx]))
    return out


# --------------------------------------------------------------------------- #
# Classification diagnostics
# --------------------------------------------------------------------------- #
def plot_classification_diagnostics(
    targets: Sequence[int],
    probs: Sequence[float],
    threshold: float,
    *,
    title: str = "",
    n_bins: int = 10,
) -> Any:
    """Three-panel diagnostics: ROC, confusion matrix, reliability/calibration curve.

    Operates on a single set of predictions (e.g. the reloaded test set). Returns the
    matplotlib ``Figure``.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

    targets = np.asarray(targets, dtype=int)
    probs = np.asarray(probs, dtype=float)
    preds = (probs >= threshold).astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    if len(np.unique(targets)) > 1:
        fpr, tpr, _ = roc_curve(targets, probs)
        auc = roc_auc_score(targets, probs)
        ax.plot(fpr, tpr, lw=2, color="#F44336", label=f"AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", alpha=0.5)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    cm = confusion_matrix(targets, preds, labels=[0, 1])
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(int(v)), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=12)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["stable", "converter"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["stable", "converter"])
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Confusion @ thr={threshold:.3f}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[2]
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1]), 0, n_bins - 1)
    xs, ys = [], []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            xs.append(float(probs[m].mean()))
            ys.append(float(targets[m].mean()))
    ax.plot([0, 1], [0, 1], "--", color="grey", alpha=0.6, label="perfect")
    ax.plot(xs, ys, "o-", color="#2196F3", label="model")
    ax.set_xlabel("mean predicted P"); ax.set_ylabel("observed frequency")
    ax.set_title("Calibration (reliability)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(); ax.grid(alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Figure saving
# --------------------------------------------------------------------------- #
def save_fig(fig: Any, run_dir: str | Path | None, name: str, *, dpi: int = 150) -> List[Path]:
    """Save ``fig`` as ``<name>.png`` and ``<name>.pdf`` under ``run_dir/figures/``.

    Returns the written paths (empty list when ``run_dir`` is None, e.g. a purely
    interactive session). Never raises on a missing dir — it is created.
    """
    if run_dir is None:
        return []
    out_dir = Path(run_dir) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for ext in ("png", "pdf"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        paths.append(p)
    return paths


__all__ = [
    "load_schaefer_atlas",
    "atlas_coords",
    "atlas_networks",
    "embed_2d",
    "plot_latent_space",
    "plot_region_importance_glassbrain",
    "plot_region_importance_bars",
    "network_importance_summary",
    "plot_classification_diagnostics",
    "save_fig",
    "NETWORK_ORDER",
    "NETWORK_COLORS",
]
