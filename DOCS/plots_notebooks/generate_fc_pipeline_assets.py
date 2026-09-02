#!/usr/bin/env python3
"""
generate_fc_pipeline_assets.py

Generates the nine inset panel PDFs for the illustrated functional-connectivity
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
# Shared synthetic data: a 7-network block-structured FC matrix,
# reused by fc_matrix, node_features, and the graph panels.
# -------------------------------------------------------------
def make_network_labels(n_roi=N_ROI, n_networks=N_NETWORKS):
    boundaries = np.linspace(0, n_roi, n_networks + 1).astype(int)
    labels = np.zeros(n_roi, dtype=int)
    for k in range(n_networks):
        labels[boundaries[k] : boundaries[k + 1]] = k
    return labels


def make_fc_matrix(n_roi=N_ROI, n_networks=N_NETWORKS, rng=RNG):
    labels = make_network_labels(n_roi, n_networks)
    within_strength = rng.uniform(0.35, 0.55, n_networks)
    base = rng.normal(0, 0.08, (n_roi, n_roi))
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
    n_latent = 8
    latent = np.stack(
        [np.sin(t * (0.5 + 0.1 * k) + rng.uniform(0, 2 * np.pi)) for k in range(n_latent)]
    )
    loadings = rng.normal(0, 1, (n_roi, n_latent))
    signal = loadings @ latent
    signal += rng.normal(0, 0.6, (n_roi, l_frames))
    signal = (signal - signal.mean(axis=1, keepdims=True)) / signal.std(axis=1, keepdims=True)
    return signal


# -------------------------------------------------------------
# 1. fmri_input.pdf -- three orthogonal synthetic brain slices
# -------------------------------------------------------------
def plot_fmri_input():
    fig, axes = plt.subplots(1, 3, figsize=(70 * MM, 28 * MM))
    for ax in axes:
        noise = RNG.normal(0, 1, (64, 64))
        for _ in range(3):
            noise = 0.6 * noise + 0.4 * np.roll(noise, 1, axis=0)
            noise = 0.6 * noise + 0.4 * np.roll(noise, 1, axis=1)
        yy, xx = np.mgrid[0:64, 0:64]
        mask = ((xx - 32) / 27) ** 2 + ((yy - 32) / 22) ** 2 <= 1
        img = np.where(mask, noise, np.nan)
        ax.imshow(img, cmap="gray", vmin=-2, vmax=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(0.5, 0.02, "4D BOLD volume (x, y, z, t)", ha="center", fontsize=7, color="black")
    fig.tight_layout(pad=0.3, rect=[0, 0.08, 1, 1])
    save(fig, "fmri_input.pdf")


# -------------------------------------------------------------
# 2. regional_timeseries.pdf -- 200 x L heatmap
# -------------------------------------------------------------
def plot_regional_timeseries():
    signal = make_timeseries()
    fig, ax = plt.subplots(figsize=(58 * MM, 32 * MM))
    im = ax.imshow(signal, aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5)
    ax.set_xlabel("Time (frames)")
    ax.set_ylabel("ROI")
    ax.set_xticks([0, L_FRAMES - 1])
    ax.set_xticklabels(["1", str(L_FRAMES)])
    ax.set_yticks([0, N_ROI - 1])
    ax.set_yticklabels(["1", str(N_ROI)])
    ax.spines[["top", "right"]].set_visible(False)
    del im
    fig.tight_layout(pad=0.3)
    save(fig, "regional_timeseries.pdf")


# -------------------------------------------------------------
# 3. fc_matrix.pdf -- 200 x 200 Fisher-z matrix, 7-network blocks
# -------------------------------------------------------------
def plot_fc_matrix():
    z, _ = make_fc_matrix()
    fig, ax = plt.subplots(figsize=(45 * MM, 42 * MM))
    vmax = np.abs(z).max()
    ax.imshow(z, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xlabel("ROI")
    ax.set_ylabel("ROI")
    ax.set_xticks([0, N_ROI // 2, N_ROI - 1])
    ax.set_xticklabels(["1", "101", "200"])
    ax.set_yticks([0, N_ROI // 2, N_ROI - 1])
    ax.set_yticklabels(["1", "101", "200"])
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.3)
    save(fig, "fc_matrix.pdf")


# -------------------------------------------------------------
# 4. node_features.pdf -- one row of the FC matrix as a feature strip
# -------------------------------------------------------------
def plot_node_features():
    z, _ = make_fc_matrix()
    roi = 72
    row = z[roi : roi + 1, :]
    fig, ax = plt.subplots(figsize=(48 * MM, 24 * MM))
    vmax = np.abs(row).max()
    ax.imshow(row, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_yticks([])
    ax.set_xlabel("ROI (feature index)")
    ax.set_xticks([0, N_ROI // 2, N_ROI - 1])
    ax.set_xticklabels(["1", "101", "200"])
    ax.axvline(roi, color=TUM_BLUE2, lw=1.2)
    ax.text(
        roi,
        1.35,
        f"ROI {roi + 1}",
        ha="center",
        va="bottom",
        fontsize=6.5,
        color=TUM_BLUE2,
        transform=ax.get_xaxis_transform(),
    )
    fig.tight_layout(pad=0.3)
    save(fig, "node_features.pdf")


# -------------------------------------------------------------
# 5/6/7/8. sparse_graph.pdf + graph_snapshot{1,2,3}.pdf
# Decimated circular-layout k-NN graph, readable at inset scale.
# -------------------------------------------------------------
def _knn_edges_by_similarity(n_nodes, k, n_communities, rng):
    """Top-k neighbours by a synthetic block-correlated similarity, not by
    on-circle spatial position, so edges appear as visible chords rather than
    collapsing onto the ring boundary."""
    labels = np.repeat(np.arange(n_communities), int(np.ceil(n_nodes / n_communities)))[:n_nodes]
    rng.shuffle(labels)
    features = rng.normal(0, 1, (n_nodes, 6))
    community_signal = rng.normal(0, 1, (n_communities, 6))
    features += 1.4 * community_signal[labels]
    sim = features @ features.T
    np.fill_diagonal(sim, -np.inf)
    edges = set()
    for i in range(n_nodes):
        nn = np.argsort(sim[i])[::-1][:k]
        for j in nn:
            edges.add((min(i, int(j)), max(i, int(j))))
    return edges


def _draw_graph_panel(ax, n_nodes, k, perturb_seed, node_size, lw):
    rng = np.random.default_rng(perturb_seed)
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    coords = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    edges = _knn_edges_by_similarity(n_nodes, k, n_communities=4, rng=rng)
    for i, j in edges:
        ax.plot(
            [coords[i, 0], coords[j, 0]],
            [coords[i, 1], coords[j, 1]],
            color=TUM_GRAY,
            lw=lw,
            alpha=0.5,
            zorder=1,
        )
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=node_size,
        color=TUM_BLUE,
        edgecolor=TUM_BLUE2,
        linewidth=0.4,
        zorder=2,
    )
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_sparse_graph():
    fig, ax = plt.subplots(figsize=(30 * MM, 30 * MM))
    _draw_graph_panel(ax, n_nodes=48, k=3, perturb_seed=1, node_size=10, lw=0.6)
    fig.tight_layout(pad=0.15)
    save(fig, "sparse_graph.pdf")


def plot_graph_snapshots():
    for idx, seed in enumerate([10, 20, 30], start=1):
        fig, ax = plt.subplots(figsize=(20 * MM, 20 * MM))
        _draw_graph_panel(ax, n_nodes=40, k=3, perturb_seed=seed, node_size=7, lw=0.5)
        fig.tight_layout(pad=0.1)
        save(fig, f"graph_snapshot{idx}.pdf")


# -------------------------------------------------------------
# 9. subject_outcome.pdf -- Stable MCI (blue) vs Converter MCI (orange)
# -------------------------------------------------------------
def plot_subject_outcome():
    fig, ax = plt.subplots(figsize=(58 * MM, 20 * MM))
    cards = [
        (0.05, "Stable MCI", "y = 0", TUM_BLUE, TUM_LTBLUE),
        (0.55, "Converter MCI", "y = 1", TUM_ORANGE, "#F3C7A5"),
    ]
    for x0, title, sub, edge, fill in cards:
        box = FancyBboxPatch(
            (x0, 0.15),
            0.40,
            0.7,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            linewidth=1.0,
            edgecolor=edge,
            facecolor=fill,
            transform=ax.transAxes,
        )
        ax.add_patch(box)
        ax.text(
            x0 + 0.20,
            0.60,
            title,
            ha="center",
            va="center",
            fontsize=7.5,
            color="black",
            transform=ax.transAxes,
            path_effects=[pe.withStroke(linewidth=2, foreground=fill)],
        )
        ax.text(
            x0 + 0.20,
            0.35,
            sub,
            ha="center",
            va="center",
            fontsize=7.5,
            style="italic",
            color="black",
            transform=ax.transAxes,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.tight_layout(pad=0.1)
    save(fig, "subject_outcome.pdf")


def main():
    plot_fmri_input()
    plot_regional_timeseries()
    plot_fc_matrix()
    plot_node_features()
    plot_sparse_graph()
    plot_graph_snapshots()
    plot_subject_outcome()


if __name__ == "__main__":
    main()
