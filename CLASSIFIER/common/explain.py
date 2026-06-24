"""
common/explain.py — model-agnostic explainability helpers for the EXPLAIN notebook.

Pure utilities used by ``notebooks/EXPLAIN/EXPLAIN_COMMON_DELCODE.ipynb`` and the
explain adapters (``adapters/explain.py``). This module is deliberately
**model-agnostic**: it imports no model code (per ``.claude/rules/architecture.md``,
``common/**`` has no model-specific imports). It covers

  * the Schaefer-200 atlas (region coordinates + network labels),
  * 2-D / 3-D embedding of a latent matrix (UMAP / PCA / t-SNE),
  * the shared latent-space scatter (cohort colours + centroids + conversion axis),
  * latent-space separability (silhouette score, per-dimension Fisher Discriminant
    Ratio, KDE distributions of the most-discriminative dimensions),
  * the disease-axis projection (logistic-regression steering direction + residual
    PCA, 2-D and 3-D, with optional longitudinal trajectory arrows),
  * region-importance renders (nilearn glass-brain markers + a per-network bar chart),
  * figure saving (png + pdf) into a run dir.

These only consume a pooled embedding matrix ``X`` (and labels ``y``), so they work
for any adapter's ``latent_embeddings()`` output — not just GAAE/VGAE.

Heavy optional dependencies (``umap``, ``nilearn``, ``plotly``) are imported lazily
inside the functions that need them, so importing this module is cheap and never
fails when an optional backend is absent — the caller gets a clear error only if it
actually uses that path.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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
    labels_arr: np.ndarray = np.asarray(labels, dtype=int)
    fig, ax = plt.subplots(figsize=(7, 6))
    palette = {1: "#F44336", 0: "#2196F3"}
    names = {1: "converter", 0: "stable MCI"}
    for lab in (0, 1):
        m = labels_arr == lab
        if m.any():
            ax.scatter(emb2d[m, 0], emb2d[m, 1], alpha=0.7, s=28,
                       color=palette[lab], label=f"{names[lab]} (n={int(m.sum())})",
                       edgecolors="white", linewidths=0.4)

    cents: Dict[int, np.ndarray] = {}
    if centroids:
        for lab in (0, 1):
            m = labels_arr == lab
            if m.any():
                c = emb2d[m].mean(0)
                cents[lab] = c
                ax.scatter(float(c[0]), float(c[1]), s=320, marker="X", color=palette[lab],
                           edgecolors="black", linewidths=1.3, zorder=5)

    if conversion_axis and 0 in cents and 1 in cents:
        c0, c1 = cents[0], cents[1]
        ax.annotate("", xy=tuple(c1), xytext=tuple(c0),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=2.0))
        mid = 0.5 * (c0 + c1)
        ax.text(float(mid[0]), float(mid[1]), "  conversion axis", fontsize=9, style="italic")

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

    targets_arr: np.ndarray = np.asarray(targets, dtype=int)
    probs_arr: np.ndarray = np.asarray(probs, dtype=float)
    preds = (probs_arr >= threshold).astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    if len(np.unique(targets_arr)) > 1:
        fpr, tpr, _ = roc_curve(targets_arr, probs_arr)
        auc = roc_auc_score(targets_arr, probs_arr)
        ax.plot(fpr, tpr, lw=2, color="#F44336", label=f"AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", alpha=0.5)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    cm = confusion_matrix(targets_arr, preds, labels=[0, 1])
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(int(v)), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=12)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["stable", "converter"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["stable", "converter"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Confusion @ thr={threshold:.3f}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[2]
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(probs_arr, edges[1:-1]), 0, n_bins - 1)
    xs, ys = [], []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            xs.append(float(probs_arr[m].mean()))
            ys.append(float(targets_arr[m].mean()))
    ax.plot([0, 1], [0, 1], "--", color="grey", alpha=0.6, label="perfect")
    ax.plot(xs, ys, "o-", color="#2196F3", label="model")
    ax.set_xlabel("mean predicted P")
    ax.set_ylabel("observed frequency")
    ax.set_title("Calibration (reliability)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Figure saving
# --------------------------------------------------------------------------- #
def latent_dim_separability(X: np.ndarray, y: Sequence[int]) -> Dict[str, Any]:
    """Silhouette score + per-dimension Fisher Discriminant Ratio for a 2-class latent space.

    ``y`` is 1 = converter, 0 = stable MCI. FDR_j = (mu1_j - mu0_j)^2 / (var1_j + var0_j)
    for each latent dimension j. Returns ``{"silhouette", "fdr" (D,), "ranked_dims"
    (D,) descending by FDR}``. Pure numpy/sklearn — no model coupling.
    """
    from sklearn.metrics import silhouette_score

    X = np.asarray(X, dtype=float)
    y_arr: np.ndarray = np.asarray(y, dtype=int)
    silhouette = float(silhouette_score(X, y_arr)) if len(np.unique(y_arr)) > 1 else float("nan")

    x1, x0 = X[y_arr == 1], X[y_arr == 0]
    mu1, mu0 = x1.mean(axis=0), x0.mean(axis=0)
    var1, var0 = x1.var(axis=0) + 1e-8, x0.var(axis=0) + 1e-8
    fdr = (mu1 - mu0) ** 2 / (var1 + var0)
    ranked_dims = np.argsort(fdr)[::-1]
    return {"silhouette": silhouette, "fdr": fdr, "ranked_dims": ranked_dims}


def plot_latent_dim_distributions(
    X: np.ndarray,
    y: Sequence[int],
    fdr: np.ndarray,
    *,
    top_n: int = 16,
    title: str = "Latent dimension distributions (top by FDR)",
) -> Any:
    """KDE (falling back to histogram) per top-FDR latent dimension, converter vs stable."""
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    X = np.asarray(X, dtype=float)
    y_arr: np.ndarray = np.asarray(y, dtype=int)
    ranked = np.argsort(fdr)[::-1][: min(int(top_n), X.shape[1])]
    n = len(ranked)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 2.8))
    axes_flat = np.atleast_1d(axes).flatten()

    x1, x0 = X[y_arr == 1], X[y_arr == 0]
    for i, dim_j in enumerate(ranked):
        ax = axes_flat[i]
        v1, v0 = x1[:, dim_j], x0[:, dim_j]
        lo, hi = min(v1.min(), v0.min()), max(v1.max(), v0.max())
        xs = np.linspace(lo, hi, 200)
        try:
            ax.fill_between(xs, gaussian_kde(v1)(xs), alpha=0.45, color="#F44336", label="converter")
            ax.fill_between(xs, gaussian_kde(v0)(xs), alpha=0.45, color="#2196F3", label="stable MCI")
        except Exception:
            ax.hist(v1, bins=20, alpha=0.5, color="#F44336", density=True, label="converter")
            ax.hist(v0, bins=20, alpha=0.5, color="#2196F3", density=True, label="stable MCI")
        ax.set_title(f"dim_{dim_j}  FDR={fdr[dim_j]:.3f}", fontsize=9)
        ax.set_yticks([])
        if i == 0:
            ax.legend(fontsize=7)

    for ax in axes_flat[n:]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def embed_3d(
    X: np.ndarray,
    *,
    seed: int = 42,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> np.ndarray:
    """Reduce ``X`` (N, D) to (N, 3) with a deterministic UMAP (3-D has no PCA/t-SNE
    fallback here — :func:`embed_2d` covers those for the 2-D case)."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"embed_3d expects a 2-D array, got shape {X.shape}.")
    if X.shape[0] < 4:
        raise ValueError(f"embed_3d needs at least 4 rows, got {X.shape[0]}.")
    try:
        import umap  # type: ignore
    except ImportError as exc:  # pragma: no cover - env without umap
        raise ImportError(
            "embed_3d needs umap-learn. Install it "
            "(pip install -r CLASSIFIER/requirements-explain.txt)."
        ) from exc
    n_neighbors = int(min(n_neighbors, max(2, X.shape[0] - 1)))
    reducer = umap.UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=min_dist,
                        random_state=seed)
    return reducer.fit_transform(X)


def plot_latent_space_3d(
    emb3d: np.ndarray,
    labels: Sequence[int],
    *,
    title: str = "Latent space (3-D UMAP)",
) -> Any:
    """3-D Plotly scatter of a latent embedding coloured by cohort (converter/stable)."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - env without plotly
        raise ImportError(
            "plot_latent_space_3d needs plotly. Install it "
            "(pip install -r CLASSIFIER/requirements-explain.txt)."
        ) from exc

    emb3d = np.asarray(emb3d, dtype=float)
    labels_arr: np.ndarray = np.asarray(labels, dtype=int)
    palette = {1: "#F44336", 0: "#2196F3"}
    names = {1: "Converter", 0: "Stable MCI"}

    fig = go.Figure()
    for lab in (0, 1):
        m = labels_arr == lab
        if m.any():
            fig.add_trace(go.Scatter3d(
                x=emb3d[m, 0], y=emb3d[m, 1], z=emb3d[m, 2],
                mode="markers", marker=dict(size=4, color=palette[lab], opacity=0.75),
                name=f"{names[lab]} (n={int(m.sum())})",
            ))
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="UMAP 1", yaxis_title="UMAP 2", zaxis_title="UMAP 3",
                   bgcolor="rgb(15,15,25)",
                   xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"), zaxis=dict(gridcolor="#333")),
        paper_bgcolor="rgb(15,15,25)", plot_bgcolor="rgb(15,15,25)",
        font=dict(color="white"), width=800, height=600,
    )
    return fig


def disease_axis_projection(X: np.ndarray, y: Sequence[int], *, seed: int = 42) -> Dict[str, Any]:
    """Fit the LR steering direction (disease axis) + residual PCA for a latent space.

    Fits ``StandardScaler`` + ``LogisticRegression(class_weight="balanced")`` on
    ``(X, y)`` (caller is responsible for passing only train+val data — see
    ``.claude/rules/evaluation.md``), then projects every row onto the unit weight
    vector ``w_hat`` (the disease score ``s``) and computes a 2-component PCA of the
    residual orthogonal to ``w_hat``. Pure numpy/sklearn; both the 2-D and 3-D plots
    are built from this one fit so the axis is identical across views.

    Returns ``{"w_hat" (D,), "scaler", "clf", "scores" (N,), "residual_pc" (N, 2)}``.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, dtype=float)
    y_arr: np.ndarray = np.asarray(y, dtype=int)
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                             random_state=seed).fit(scaler.transform(X), y_arr)

    w = clf.coef_.ravel()
    w_hat = w / np.linalg.norm(w)
    z_std = scaler.transform(X)
    scores = z_std @ w_hat
    residual = z_std - np.outer(scores, w_hat)
    residual_pc = PCA(n_components=2, random_state=seed).fit_transform(residual)
    return {"w_hat": w_hat, "scaler": scaler, "clf": clf, "scores": scores, "residual_pc": residual_pc}


def plot_disease_axis(
    proj: Dict[str, Any],
    y: Sequence[int],
    *,
    visit_groups: Optional[Dict[str, List[int]]] = None,
    title: str = "Disease axis projection",
) -> Any:
    """2-D matplotlib view of a :func:`disease_axis_projection` fit.

    ``visit_groups`` (optional) maps ``subject_id -> ordered row indices`` into
    ``proj``'s arrays; when given, draws a trajectory arrow per multi-visit subject.
    """
    import matplotlib.pyplot as plt

    s = proj["scores"]
    pc1 = proj["residual_pc"][:, 0]
    y_arr: np.ndarray = np.asarray(y, dtype=int)
    palette = {1: "#F44336", 0: "#2196F3"}
    names = {1: "Converter", 0: "Stable MCI"}

    fig, ax = plt.subplots(figsize=(10, 6.5))
    for lab in (0, 1):
        m = y_arr == lab
        if m.any():
            ax.scatter(s[m], pc1[m], c=palette[lab], alpha=0.45, s=22,
                       edgecolors="none", label=f"{names[lab]} (n={int(m.sum())})", zorder=2)

    centroids: Dict[int, tuple] = {}
    for lab in (0, 1):
        m = y_arr == lab
        if m.any():
            c = (float(s[m].mean()), float(pc1[m].mean()))
            centroids[lab] = c
            ax.scatter(*c, marker="*", s=300, c=palette[lab],
                       edgecolors="black", linewidths=0.8, zorder=5)
            ax.annotate(names[lab].upper(), c, textcoords="offset points",
                       xytext=(6, 4), fontsize=9, fontweight="bold")

    if visit_groups:
        for _sid, idx in visit_groups.items():
            if len(idx) < 2:
                continue
            xs, ys_ = s[idx], pc1[idx]
            lab = int(y_arr[idx[0]])
            col = palette.get(lab, "#9E9E9E")
            ax.plot(xs, ys_, color=col, alpha=0.35, lw=1.2, zorder=1)
            ax.annotate("", xy=(xs[-1], ys_[-1]), xytext=(xs[-2], ys_[-2]),
                       arrowprops=dict(arrowstyle="->", color=col, lw=1.0, alpha=0.6))

    ax.axvline(0, color="black", lw=1.2, linestyle="--", alpha=0.6, label="Decision boundary (s=0)")
    ax.set_xlabel("Disease score  (s = Z ŵ)  →  conversion direction", fontsize=11)
    ax.set_ylabel("Residual PC1  (orthogonal variation)", fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_disease_axis_3d(
    proj: Dict[str, Any],
    y: Sequence[int],
    *,
    visit_groups: Optional[Dict[str, List[int]]] = None,
    title: str = "Disease axis + residual PCA (3-D)",
) -> Any:
    """3-D Plotly view of a :func:`disease_axis_projection` fit (centroids, decision
    boundary plane, and per-subject trajectory lines when ``visit_groups`` is given)."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - env without plotly
        raise ImportError(
            "plot_disease_axis_3d needs plotly. Install it "
            "(pip install -r CLASSIFIER/requirements-explain.txt)."
        ) from exc

    s = proj["scores"]
    pc1, pc2 = proj["residual_pc"][:, 0], proj["residual_pc"][:, 1]
    y_arr: np.ndarray = np.asarray(y, dtype=int)
    palette = {1: "#F44336", 0: "#2196F3"}
    names = {1: "Converter", 0: "Stable MCI"}

    fig = go.Figure()
    for lab in (0, 1):
        m = y_arr == lab
        if m.any():
            fig.add_trace(go.Scatter3d(
                x=s[m], y=pc1[m], z=pc2[m], mode="markers",
                marker=dict(size=4, color=palette[lab], opacity=0.7),
                name=f"{names[lab]} (n={int(m.sum())})",
            ))
            fig.add_trace(go.Scatter3d(
                x=[float(s[m].mean())], y=[float(pc1[m].mean())], z=[float(pc2[m].mean())],
                mode="markers+text", marker=dict(size=12, color=palette[lab], symbol="diamond",
                                                 line=dict(color="white", width=1)),
                text=[names[lab].upper()], textposition="top center",
                name=f"{names[lab]} centroid",
            ))

    if visit_groups:
        for _sid, idx in visit_groups.items():
            if len(idx) < 2:
                continue
            lab = int(y_arr[idx[0]])
            col = palette.get(lab, "#888")
            fig.add_trace(go.Scatter3d(
                x=s[idx], y=pc1[idx], z=pc2[idx], mode="lines",
                line=dict(color=col, width=2), opacity=0.35,
                showlegend=False, hoverinfo="skip",
            ))

    pc1_range = [float(pc1.min()), float(pc1.max())]
    pc2_range = [float(pc2.min()), float(pc2.max())]
    fig.add_trace(go.Surface(
        x=[[0, 0], [0, 0]], y=[pc1_range, pc1_range],
        z=[[pc2_range[0], pc2_range[1]], [pc2_range[0], pc2_range[1]]],
        opacity=0.15, colorscale=[[0, "grey"], [1, "grey"]],
        showscale=False, name="Decision boundary (s=0)", showlegend=True,
    ))

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="Disease score (ŵ · z)",
                   yaxis_title="Residual PC1", zaxis_title="Residual PC2",
                   bgcolor="rgb(15,15,25)",
                   xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"), zaxis=dict(gridcolor="#333")),
        paper_bgcolor="rgb(15,15,25)", plot_bgcolor="rgb(15,15,25)",
        font=dict(color="white"), width=900, height=650,
    )
    return fig


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
    "embed_3d",
    "plot_latent_space",
    "plot_latent_space_3d",
    "latent_dim_separability",
    "plot_latent_dim_distributions",
    "disease_axis_projection",
    "plot_disease_axis",
    "plot_disease_axis_3d",
    "plot_region_importance_glassbrain",
    "plot_region_importance_bars",
    "network_importance_summary",
    "plot_classification_diagnostics",
    "save_fig",
    "NETWORK_ORDER",
    "NETWORK_COLORS",
]
