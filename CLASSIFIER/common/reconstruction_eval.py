"""
Model-agnostic per-sample reconstruction-error tabulation.

Why this module exists
-----------------------
``model/GAAE/evaluation.py::compute_errors_for_dataset`` is hardwired to GAAE's
two-component loss (``compute_sample_reconstruction_error(..., adj_loss_weight)``
emitting ``X Error``/``Adj Error`` columns). VGAE has no equivalent at all. This
module provides the same row-building / noise-perturbation logic parameterised
by an ``error_fn`` callable instead, so a single function works for any
reconstruction-based encoder (GAAE, VGAE, future architectures). The other GAAE
evaluation helpers (``compute_one_vs_rest_thresholds``, ``is_cohort_positive``,
``plot_cohort_errors``, ``plot_robustness_sweep``) only touch the resulting
DataFrame/dict — no GAAE coupling — so they are reused unchanged against this
function's output.
"""
from __future__ import annotations

from typing import Callable, Union

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from .robustness import perturb_graph

ErrorFn = Callable[[Data], Union[float, dict]]


def compute_errors_for_dataset(
    dataset,
    split_name: str,
    error_fn: ErrorFn,
    cohort_map: dict,
    *,
    allowed_cohorts: set | None = None,
    noise_method: str = "none",
    noise_level: float = 0.0,
    rng: np.random.Generator | None = None,
    adjacency_args: dict | None = None,
) -> pd.DataFrame:
    """
    Compute reconstruction errors for every sample in ``dataset``.

    error_fn(sample) -> float | dict
        A float is taken as the row's ``Total Error``. A dict must contain a
        ``"total_error"`` key; any other keys become extra columns (e.g.
        GAAE's ``x_error``/``adj_error``, VGAE's ``recon_error``/``kl_error``).

    allowed_cohorts: if given, IDs that don't map to an allowed cohort raise
    ValueError — use for train/val where every subject must be labelled.
    Without it, unmapped IDs get cohort='unknown' (use for test/robustness).

    noise_method='matrix_noise_rebuild' perturbs node features then rebuilds
    the kNN graph; requires adjacency_args. All other methods delegate to
    common.robustness.perturb_graph.
    """
    records: list[dict] = []
    unknown_ids: list[str] = []

    for i in range(len(dataset)):
        sample = dataset[i]
        patient_id = str(getattr(sample, "patient_id", f"idx_{i}")).strip()
        cohort = str(cohort_map.get(patient_id, "unknown")).lower()

        if allowed_cohorts is not None and cohort not in allowed_cohorts:
            unknown_ids.append(patient_id)
            continue

        if noise_level > 0 and noise_method != "none":
            if noise_method == "matrix_noise_rebuild" and adjacency_args is not None:
                from CLASSIFIER.model.GAAE.utils import knn_binary_adjacency_matrix_no_diag

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

        result = error_fn(d)
        if isinstance(result, dict):
            if "total_error" not in result:
                raise ValueError(
                    f"error_fn returned a dict without a 'total_error' key: {result!r}"
                )
            extra = {k: float(v) for k, v in result.items() if k != "total_error"}
            total_error = float(result["total_error"])
        else:
            extra = {}
            total_error = float(result)

        records.append({
            "Split": split_name,
            "DatasetIndex": i,
            "PatientID": patient_id,
            "Cohort": cohort,
            **extra,
            "Total Error": total_error,
        })

    if allowed_cohorts is not None and unknown_ids:
        unique_unknown = sorted(set(unknown_ids))
        raise ValueError(
            f"{split_name}: {len(unique_unknown)} IDs have no allowed cohort mapping. "
            f"Examples: {unique_unknown[:10]}"
        )

    return pd.DataFrame(records)
