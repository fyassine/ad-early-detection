"""
common/plots.py — shared result plots for the longitudinal notebooks.

Pure plotting on already-prepared arrays / frames: no model, no I/O, no threshold
derivation. Each function returns the ``matplotlib.figure.Figure`` so callers can
``plt.show()`` it, save it, or log it. Importing this module does not pick a
backend, so it is safe under a headless (Agg) test environment.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def plot_oof_test_roc(
    oof_targets,
    oof_probs,
    test_targets,
    test_probs,
    *,
    title: str = "",
) -> Any:
    """Two-panel ROC figure: pooled out-of-fold (left) and test set (right)."""
    import matplotlib.pyplot as plt

    oof_targets = np.asarray(oof_targets, dtype=int)
    oof_probs = np.asarray(oof_probs, dtype=float)
    test_targets = np.asarray(test_targets, dtype=int)
    test_probs = np.asarray(test_probs, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    fpr, tpr, _ = roc_curve(oof_targets, oof_probs)
    auc_oof = roc_auc_score(oof_targets, oof_probs)
    ax.plot(fpr, tpr, lw=2, color="#2196F3", label=f"OOF ROC (AUC={auc_oof:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", alpha=0.5)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("OOF Cross-Validation ROC")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    fpr_te, tpr_te, _ = roc_curve(test_targets, test_probs)
    auc_te = roc_auc_score(test_targets, test_probs)
    ax.plot(fpr_te, tpr_te, lw=2, color="#F44336", label=f"Test ROC (AUC={auc_te:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", alpha=0.5)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("Test-Set ROC")
    ax.legend()
    ax.grid(alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def plot_conversion_trajectories(traj_df, threshold: float, *, title: str = "") -> Any:
    """Per-visit P(converter) trajectories, split into converter / stable panels.

    ``traj_df`` columns: ``pid``, ``label`` (1=converter, 0=stable), ``month``, ``prob``.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    palette = {1: "#F44336", 0: "#2196F3"}
    for ax, (label, panel_title) in zip(axes, [(1, "Converters"), (0, "Stable MCI")], strict=False):
        sub = traj_df[traj_df["label"] == label]
        for _pid, grp in sub.groupby("pid"):
            ax.plot(
                grp["month"], grp["prob"], marker="o", alpha=0.6,
                color=palette[label], lw=1.5,
            )
        ax.axhline(
            threshold, color="black", lw=1.2, linestyle="--",
            label=f"Threshold={threshold:.3f}",
        )
        ax.set_xlabel("Visit month")
        ax.set_ylabel("P(converter)")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(panel_title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig
