#!/usr/bin/env python3
"""
generate_fc_pipeline_assets.py

Generates the eight inset panel PDFs for the illustrated functional-connectivity
graph-construction pipeline figure (DOCS/plots_tkiz/fc_pipeline_diagram/).

All panel content is SYNTHETIC and SCHEMATIC (illustrative only, not measured data).

Compliant with .claude/rules/plots.md:
- Typography: Courier New monospace, correct pt sizes
- Vector export: PDF (TrueType 42)
- Colours: TUM corporate palette (matches THESIS/settings.tex) plus Okabe-Ito style
  diverging map for connectivity data (blue-white-red, not red-green)
"""

from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

MM = 1 / 25.4

plt.rcParams.update(
    {
        "font.family": "Courier New",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "DOCS" / "plots_tkiz" / "fc_pipeline_diagram"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TUM_BLUE = "#0065BD"
TUM_BLUE2 = "#003359"
TUM_ORANGE = "#E37222"
TUM_LTBLUE = "#98C6EA"
TUM_GRAY = "#808080"

RNG = np.random.default_rng(0)

N_ROI = 200
N_NETWORKS = 7
L_FRAMES = 180


def save(fig, name):
    path = OUTPUT_DIR / name
    fig.savefig(path, transparent=True, edgecolor="none")
    plt.close(fig)
    print(f"wrote {path}")


# -------------------------------------------------------------
# Shared synthetic data & 16-node representative network
# -------------------------------------------------------------
def make_network_labels(n_roi=N_ROI, n_networks=N_NETWORKS):
    boundaries = np.linspace(0, n_roi, n_networks + 1).astype(int)
    labels = np.zeros(n_roi, dtype=int)
    for k in range(n_networks):
        labels[boundaries[k] : boundaries[k + 1]] = k
    return labels


def make_fc_matrix(n_roi=N_ROI, n_networks=N_NETWORKS, rng=RNG):
    labels = make_network_labels(n_roi, n_networks)
    within_strength = rng.uniform(0.40, 0.60, n_networks)
    base = rng.normal(0, 0.07, (n_roi, n_roi))
    for k in range(n_networks):
        mask = labels == k
        idx = np.where(mask)[0]
        base[np.ix_(idx, idx)] += within_strength[k]
    corr = (base + base.T) / 2
    np.fill_diagonal(corr, 0.0)
    corr = np.clip(corr, -0.95, 0.95)
    z = np.arctanh(corr)
    return z, labels


def make_timeseries(n_roi=N_ROI, l_frames=L_FRAMES, rng=RNG):
    t = np.linspace(0, 8 * np.pi, l_frames)
    n_latent = 7
    latent = np.stack(
        [np.sin(t * (0.4 + 0.12 * k) + rng.uniform(0, 2 * np.pi)) for k in range(n_latent)]
    )
    loadings = rng.normal(0, 1, (n_roi, n_latent))
    signal = loadings @ latent
    signal += rng.normal(0, 0.5, (n_roi, l_frames))
    signal = (signal - signal.mean(axis=1, keepdims=True)) / signal.std(axis=1, keepdims=True)
    return signal


# -------------------------------------------------------------
# 1. fmri_input.pdf -- Clear brain slice with parcellation motif
# -------------------------------------------------------------
def plot_fmri_input():
    fig, ax = plt.subplots(figsize=(42 * MM, 26 * MM))
    
    # Generate an axial-like synthetic brain slice
    res = 128
    y, x = np.mgrid[-1.1:1.1:complex(res), -1.0:1.0:complex(res)]
    
    # Anatomical outer shape (oval with frontal/occipital contour)
    dist = (x / 0.72) ** 2 + ((y + 0.05 * x**2) / 0.88) ** 2
    brain_mask = dist <= 1.0
    
    # Ventricles mask
    ventricle_mask = ((x / 0.12) ** 2 + ((y - 0.05) / 0.30) ** 2 <= 0.35) & (abs(x) > 0.02)
    
    # Cortical ribbon mask (outer ring of brain)
    cortex_mask = brain_mask & (dist >= 0.65)
    
    # Background anatomical intensity
    rng = np.random.default_rng(42)
    base_tex = rng.normal(0.65, 0.08, (res, res))
    # Smooth
    for _ in range(4):
        base_tex = 0.5 * base_tex + 0.25 * (np.roll(base_tex, 1, axis=0) + np.roll(base_tex, 1, axis=1))
    
    img = np.where(brain_mask, base_tex, np.nan)
    img = np.where(ventricle_mask, 0.2, img)
    
    ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0, extent=[-1, 1, -1.1, 1.1])
    
    # Overlay distinct colored Schaefer parcellation patches on cortex
    angles = np.arctan2(y, x)
    # Define 7 distinct angle sectors on cortex
    sector_colors = [
        "#0065BD", "#E37222", "#A2AD00", "#98C6EA", 
        "#64A0C8", "#005293", "#CC79A7"
    ]
    
    for k in range(7):
        th0 = -np.pi + k * (2 * np.pi / 7)
        th1 = -np.pi + (k + 1) * (2 * np.pi / 7)
        sec = cortex_mask & (angles >= th0) & (angles < th1)
        # Sub-sample to show individual discrete parcels
        p_mask = sec & ((np.sin(angles * 28) > -0.2))
        ax.contourf(x, y, p_mask, levels=[0.5, 1.5], colors=[sector_colors[k]], alpha=0.85)
        ax.contour(x, y, p_mask, levels=[0.5], colors=["white"], linewidths=0.5, alpha=0.9)
    
    # Clean outer boundary
    ax.contour(x, y, brain_mask, levels=[0.5], colors=["#333333"], linewidths=0.8)
    
    ax.set_xlim(-0.85, 0.85)
    ax.set_ylim(-0.95, 0.95)
    ax.axis("off")
    fig.tight_layout(pad=0.1)
    save(fig, "fmri_input.pdf")


# -------------------------------------------------------------
# 2. regional_timeseries.pdf -- 200 x L heatmap without tiny tick junk
# -------------------------------------------------------------
def plot_regional_timeseries():
    signal = make_timeseries()
    fig, ax = plt.subplots(figsize=(42 * MM, 26 * MM))
    ax.imshow(signal, aspect="auto", cmap="coolwarm", vmin=-2.2, vmax=2.2)
    ax.set_xlabel("Time (frames)", fontsize=7, labelpad=2)
    ax.set_ylabel("200 ROIs", fontsize=7, labelpad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.6)
    fig.tight_layout(pad=0.2)
    save(fig, "regional_timeseries.pdf")


# -------------------------------------------------------------
# 3. fc_matrix.pdf -- 200 x 200 Fisher-z matrix, clean block structure
# -------------------------------------------------------------
def plot_fc_matrix():
    z, labels = make_fc_matrix()
    fig, ax = plt.subplots(figsize=(30 * MM, 30 * MM))
    vmax = np.percentile(np.abs(z), 98)
    ax.imshow(z, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    
    # Draw subtle network boundaries
    boundaries = np.linspace(0, N_ROI, N_NETWORKS + 1)
    for b in boundaries[1:-1]:
        ax.axvline(b - 0.5, color="white", lw=0.4, alpha=0.7)
        ax.axhline(b - 0.5, color="white", lw=0.4, alpha=0.7)
        
    ax.set_xlabel("200 ROIs", fontsize=7, labelpad=2)
    ax.set_ylabel("200 ROIs", fontsize=7, labelpad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#555555")
    fig.tight_layout(pad=0.2)
    save(fig, "fc_matrix.pdf")


# -------------------------------------------------------------
# 4. node_features.pdf -- row extraction from FC matrix to vector
# -------------------------------------------------------------
def plot_node_features():
    z, _ = make_fc_matrix()
    roi = 68
    row = z[roi : roi + 1, :]
    
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(46 * MM, 24 * MM), 
        gridspec_kw={"width_ratios": [1.1, 1.8], "wspace": 0.35}
    )
    
    # Subplot 1: Mini FC matrix with highlighted row
    vmax = np.percentile(np.abs(z), 98)
    ax1.imshow(z, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax1.axhline(roi, color=TUM_ORANGE, lw=1.2)
    ax1.set_title("FC matrix", fontsize=6.5, pad=2)
    ax1.set_xticks([])
    ax1.set_yticks([])
    for spine in ax1.spines.values():
        spine.set_linewidth(0.5)
    
    # Subplot 2: Extracted 1D feature row
    ax2.imshow(row, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax, extent=[0, 200, 0, 1])
    ax2.set_title(r"$\mathbf{x}_j = C_{j,:}$", fontsize=7, pad=2)
    ax2.set_xlabel("200 features", fontsize=6.5, labelpad=1)
    ax2.set_yticks([])
    ax2.set_xticks([0, 100, 200])
    ax2.set_xticklabels(["1", "100", "200"], fontsize=6)
    for spine in ax2.spines.values():
        spine.set_linewidth(0.5)
        
    fig.subplots_adjust(left=0.08, right=0.96, top=0.88, bottom=0.18, wspace=0.35)
    save(fig, "node_features.pdf")


# -------------------------------------------------------------
# 5/6/7/8. 16-node representative graphs with FIXED node layout
# -------------------------------------------------------------
# Define 16 nodes grouped into 4 functional communities:
# Quad 0: Visual (TUM Blue)
# Quad 1: Default Mode (TUM Orange)
# Quad 2: Somatomotor (TUM Green)
# Quad 3: Control (TUM Light Blue)
COMMUNITY_COLORS = [TUM_BLUE, TUM_ORANGE, "#A2AD00", "#56B4E9"]
COMMUNITY_BORDER = [TUM_BLUE2, "#9E4700", "#5C6300", "#1C6B99"]

def _get_fixed_16node_layout():
    """Generates 16 fixed 2D node coordinates grouped into 4 modular clusters."""
    coords = np.zeros((16, 2))
    # 4 communities of 4 nodes each
    quad_centers = [
        np.array([0.65, 0.65]),    # Comm 0 (top-right)
        np.array([-0.65, 0.65]),   # Comm 1 (top-left)
        np.array([-0.65, -0.65]),  # Comm 2 (bottom-left)
        np.array([0.65, -0.65]),   # Comm 3 (bottom-right)
    ]
    # Fixed intra-cluster offsets
    offsets = [
        np.array([-0.22, 0.18]),
        np.array([0.22, 0.20]),
        np.array([0.20, -0.22]),
        np.array([-0.20, -0.18]),
    ]
    for c in range(4):
        for i in range(4):
            node_idx = c * 4 + i
            coords[node_idx] = quad_centers[c] + offsets[i]
    return coords


def _get_baseline_edges():
    """Returns baseline edge set for the 16-node network (~18 edges)."""
    edges = set()
    # Intra-community edges (dense within module)
    for c in range(4):
        base = c * 4
        # Ring of 4 inside module + 1 cross edge
        intra = [
            (base, base + 1), (base + 1, base + 2), 
            (base + 2, base + 3), (base + 3, base),
            (base, base + 2)
        ]
        for u, v in intra:
            edges.add((min(u, v), max(u, v)))
    
    # Inter-community edges (long-range cross-talk)
    inter = [
        (1, 4),   # Comm 0 - Comm 1
        (7, 8),   # Comm 1 - Comm 2
        (11, 14), # Comm 2 - Comm 3
        (13, 2),  # Comm 3 - Comm 0
        (0, 10),  # Cross-diagonal
    ]
    for u, v in inter:
        edges.add((min(u, v), max(u, v)))
    return edges


def _draw_network(ax, coords, edges, edge_styles=None, node_size=28):
    """Draws network with fixed nodes and specified edge styles."""
    if edge_styles is None:
        edge_styles = {}
        
    # Draw edges
    for u, v in edges:
        e = (min(u, v), max(u, v))
        style = edge_styles.get(e, {"color": TUM_GRAY, "lw": 0.8, "alpha": 0.55, "ls": "-"})
        ax.plot(
            [coords[u, 0], coords[v, 0]],
            [coords[u, 1], coords[v, 1]],
            color=style.get("color", TUM_GRAY),
            lw=style.get("lw", 0.8),
            alpha=style.get("alpha", 0.55),
            linestyle=style.get("ls", "-"),
            zorder=1,
        )
        
    # Draw nodes
    for i in range(16):
        c = i // 4
        ax.scatter(
            coords[i, 0],
            coords[i, 1],
            s=node_size,
            color=COMMUNITY_COLORS[c],
            edgecolor=COMMUNITY_BORDER[c],
            linewidth=0.6,
            zorder=3,
        )
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_sparse_graph():
    fig, ax = plt.subplots(figsize=(26 * MM, 26 * MM))
    coords = _get_fixed_16node_layout()
    edges = _get_baseline_edges()
    _draw_network(ax, coords, edges, node_size=26)
    fig.tight_layout(pad=0.1)
    save(fig, "sparse_graph.pdf")


def plot_graph_snapshots():
    coords = _get_fixed_16node_layout()
    base_edges = _get_baseline_edges()
    
    # Snapshot 1: Visit 1 (Baseline)
    fig1, ax1 = plt.subplots(figsize=(24 * MM, 24 * MM))
    _draw_network(ax1, coords, base_edges, node_size=22)
    fig1.tight_layout(pad=0.08)
    save(fig1, "graph_snapshot1.pdf")
    
    # Snapshot 2: Visit 2 (Follow-up 1)
    # 2 weakened/lost edges (dashed or removed), 2 strengthened edges (highlighted in TUM_ORANGE)
    fig2, ax2 = plt.subplots(figsize=(24 * MM, 24 * MM))
    edges_v2 = set(base_edges)
    # Weakened edge
    edges_v2.remove((0, 10)) # lost cross-diagonal
    # Add new/altered connections
    edges_v2.add((5, 13))    # altered inter-community
    edges_v2.add((1, 8))     # altered cross-talk
    
    edge_styles_v2 = {
        (5, 13): {"color": TUM_ORANGE, "lw": 1.5, "alpha": 0.95, "ls": "-"},
        (1, 8): {"color": TUM_ORANGE, "lw": 1.5, "alpha": 0.95, "ls": "-"},
    }
    _draw_network(ax2, coords, edges_v2, edge_styles=edge_styles_v2, node_size=22)
    fig2.tight_layout(pad=0.08)
    save(fig2, "graph_snapshot2.pdf")
    
    # Snapshot 3: Visit 3 (Follow-up 2)
    # Further persistent progression of dysconnectivity
    fig3, ax3 = plt.subplots(figsize=(24 * MM, 24 * MM))
    edges_v3 = set(edges_v2)
    edges_v3.remove((7, 8))   # further lost cross-talk
    edges_v3.add((4, 15))    # another altered edge
    
    edge_styles_v3 = {
        (5, 13): {"color": TUM_ORANGE, "lw": 1.5, "alpha": 0.95, "ls": "-"},
        (1, 8): {"color": TUM_ORANGE, "lw": 1.5, "alpha": 0.95, "ls": "-"},
        (4, 15): {"color": TUM_ORANGE, "lw": 1.5, "alpha": 0.95, "ls": "-"},
    }
    _draw_network(ax3, coords, edges_v3, edge_styles=edge_styles_v3, node_size=22)
    fig3.tight_layout(pad=0.08)
    save(fig3, "graph_snapshot3.pdf")


def main():
    # Ensure both output directory and assets subdirectory exist
    assets_dir = OUTPUT_DIR / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    
    plot_fmri_input()
    plot_regional_timeseries()
    plot_fc_matrix()
    plot_node_features()
    plot_sparse_graph()
    plot_graph_snapshots()

    # Also copy all 8 PDFs into assets/ subfolder for path flexibility
    import shutil
    for f in [
        "fmri_input.pdf", "regional_timeseries.pdf", "fc_matrix.pdf",
        "node_features.pdf", "sparse_graph.pdf", "graph_snapshot1.pdf",
        "graph_snapshot2.pdf", "graph_snapshot3.pdf"
    ]:
        src = OUTPUT_DIR / f
        dst = assets_dir / f
        shutil.copy2(src, dst)
    print(f"Copied all 8 assets to {assets_dir}")


if __name__ == "__main__":
    main()

