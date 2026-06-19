from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

from .losses import compute_sample_reconstruction_error
from .utils import knn_binary_adjacency_matrix_no_diag
from CLASSIFIER.common.robustness import perturb_graph


def compute_errors_for_dataset(
    dataset,
    split_name: str,
    model,
    device,
    cohort_map: dict,
    adj_loss_weight: float,
    *,
    allowed_cohorts: set | None = None,
    adjacency_args: dict | None = None,
    noise_method: str = "none",
    noise_level: float = 0.0,
    rng=None,
) -> pd.DataFrame:
    """
    Compute GAAE reconstruction errors for every sample in dataset.

    allowed_cohorts: if given, IDs that don't map to an allowed cohort raise
    ValueError — use for train/val where every subject must be labelled.
    Without it, unmapped IDs get cohort='unknown' (use for test/robustness).

    noise_method='matrix_noise_rebuild' perturbs node features then rebuilds
    the kNN graph; requires adjacency_args.  All other methods delegate to
    common.robustness.perturb_graph.
    """
    records: list[dict] = []
    unknown_ids: list[str] = []
    model.eval()

    for i in range(len(dataset)):
        sample = dataset[i]
        patient_id = str(getattr(sample, "patient_id", f"idx_{i}")).strip()
        cohort = str(cohort_map.get(patient_id, "unknown")).lower()

        if allowed_cohorts is not None and cohort not in allowed_cohorts:
            unknown_ids.append(patient_id)
            continue

        if noise_level > 0 and noise_method != "none":
            if noise_method == "matrix_noise_rebuild" and adjacency_args is not None:
                d = perturb_graph(sample, "feature_noise", noise_level, rng=rng)
                adj_bin = knn_binary_adjacency_matrix_no_diag(
                    d.x.detach().cpu().numpy(), **adjacency_args
                )
                src, dst = np.where(adj_bin > 0)
                d.edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
                d.edge_attr = torch.ones(d.edge_index.size(1), dtype=torch.float32)
            else:
                d = perturb_graph(sample, noise_method, noise_level, rng=rng)
        else:
            d = sample

        x_err, adj_err, total_err = compute_sample_reconstruction_error(
            d, model, device, adj_loss_weight
        )
        records.append({
            "Split": split_name,
            "DatasetIndex": i,
            "PatientID": patient_id,
            "Cohort": cohort,
            "X Error": float(x_err),
            "Adj Error": float(adj_err),
            "Total Error": float(total_err),
        })

    if allowed_cohorts is not None and unknown_ids:
        unique_unknown = sorted(set(unknown_ids))
        raise ValueError(
            f"{split_name}: {len(unique_unknown)} IDs have no allowed cohort mapping. "
            f"Examples: {unique_unknown[:10]}"
        )

    return pd.DataFrame(records)


def compute_one_vs_rest_thresholds(
    val_errors_df: pd.DataFrame,
    cohorts: list[str],
) -> dict:
    """
    For each cohort derive a Youden-optimal reconstruction-error threshold
    via one-vs-rest AUROC on the val set.

    Returns a dict keyed by cohort name; each value has:
      direction ('high'/'low'), auc, threshold_error, threshold_score.
    """
    val = val_errors_df.copy()
    val["Cohort"] = val["Cohort"].astype(str).str.lower().str.strip()
    val_errors = val["Total Error"].astype(float).values
    out: dict = {}

    for cohort_name in cohorts:
        y = (val["Cohort"] == cohort_name).astype(int).values
        if y.sum() == 0 or y.sum() == len(y):
            out[cohort_name] = {
                "direction": "low", "auc": float("nan"),
                "threshold_error": float("nan"), "threshold_score": float("nan"),
            }
            continue

        auc_high = roc_auc_score(y, val_errors)
        auc_low = roc_auc_score(y, -val_errors)
        if auc_high >= auc_low:
            direction, scores, auc_used = "high", val_errors, auc_high
        else:
            direction, scores, auc_used = "low", -val_errors, auc_low

        fpr, tpr, score_thresholds = roc_curve(y, scores)
        best_idx = int(np.argmax(tpr - fpr))
        threshold_score = float(score_thresholds[best_idx])
        threshold_error = float(
            threshold_score if direction == "high" else -threshold_score
        )
        out[cohort_name] = {
            "direction": direction,
            "auc": float(auc_used),
            "threshold_error": threshold_error,
            "threshold_score": threshold_score,
        }
    return out


def is_cohort_positive(total_error: float, threshold: dict) -> int:
    """Return 1 if total_error crosses the cohort threshold dict, else 0."""
    if np.isnan(threshold["threshold_error"]):
        return 0
    if threshold["direction"] == "high":
        return int(total_error >= threshold["threshold_error"])
    return int(total_error <= threshold["threshold_error"])


def plot_cohort_errors(
    split_name: str,
    split_df: pd.DataFrame,
    cohort_order: list[str],
    palette_name: str = "Blues",
    wandb_project: str = "",
    run_name: str = "",
) -> None:
    """Swarmplot + boxplot of per-cohort reconstruction errors."""
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    plt.figure(figsize=(10, 7))
    ax = plt.gca()

    sns.swarmplot(
        data=split_df, x="Cohort", y="Total Error", order=cohort_order,
        color=".25", size=4, alpha=0.6, zorder=1, ax=ax,
    )
    boxplot = sns.boxplot(
        data=split_df, x="Cohort", y="Total Error", order=cohort_order,
        palette=palette_name, showcaps=True,
        boxprops={"edgecolor": "black", "linewidth": 2},
        medianprops={"color": "red", "linewidth": 2.5},
        whiskerprops={"color": "black", "linewidth": 2},
        capprops={"color": "black", "linewidth": 2},
        zorder=2, ax=ax,
    )
    for patch in boxplot.patches:
        fc = patch.get_facecolor()
        patch.set_facecolor((fc[0], fc[1], fc[2], 0.5))

    y_top = split_df["Total Error"].max()
    y_min = split_df["Total Error"].min()
    y_span = max(y_top - y_min, 1e-6)
    ax.set_ylim(y_min - 0.05 * y_span, y_top + 0.18 * y_span)

    for i, cohort in enumerate(cohort_order):
        vals = split_df[split_df["Cohort"] == cohort]["Total Error"]
        if vals.empty:
            continue
        std = vals.std(ddof=1) if len(vals) > 1 else 0.0
        ax.text(
            i, y_top + 0.12 * y_span,
            f"μ={vals.mean():.4f}\nσ={std:.4f}",
            ha="center", va="top", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.8),
        )

    ax.set_title(split_name, fontsize=16, fontweight="bold")
    ax.set_xlabel("Cohort", fontsize=14)
    ax.set_ylabel("Total Weighted Reconstruction Error", fontsize=14)
    ax.tick_params(axis="x", rotation=25, labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    if wandb_project or run_name:
        plt.figtext(
            0.99, 0.01, f"Project: {wandb_project}, Run: {run_name}",
            ha="right", va="bottom", fontsize=6, alpha=0.5,
        )
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.show()


def plot_robustness_sweep(
    summary_df: pd.DataFrame,
    cohorts_to_analyze: list[str],
    cohort_thresholds: dict,
    noise_methods: list[str],
) -> None:
    """Error drift + decision-stability plots for each cohort."""
    for cohort_name in cohorts_to_analyze:
        cohort_summary = summary_df[summary_df["SelectionCohort"] == cohort_name]
        if cohort_summary.empty:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        threshold_error = cohort_thresholds[cohort_name]["threshold_error"]

        for method in noise_methods:
            m = cohort_summary[cohort_summary["Method"] == method]
            axes[0].plot(m["NoiseLevelPercent"], m["MeanTotalError"], marker="o", label=method)
        axes[0].axhline(threshold_error, linestyle="--", color="black", alpha=0.6,
                        label=f"{cohort_name} threshold")
        axes[0].set_xlabel("Noise level (%)")
        axes[0].set_ylabel("Mean total reconstruction error")
        axes[0].set_title(f"Error drift under noise ({cohort_name})")
        axes[0].legend()

        for method in noise_methods:
            m = cohort_summary[cohort_summary["Method"] == method]
            axes[1].plot(m["NoiseLevelPercent"], m["CohortStabilityRate"], marker="o", label=method)
        axes[1].set_xlabel("Noise level (%)")
        axes[1].set_ylabel("Cohort-correct prediction rate")
        axes[1].set_ylim(0, 1)
        axes[1].set_title(f"Decision stability under noise ({cohort_name})")
        axes[1].legend()

        plt.tight_layout()
        plt.show()
