"""
model/sequences.py — DELCODE subject record -> Brain-TokenGT ``(A_list, Nodes_list)``.

This replaces upstream ``Brain-TokenGT/datasets.py`` entirely. That file is a
synthetic-data loader and cannot be ported: its line 24 does

    FC = FC[keys].astype('int64')

which truncates correlations to integers. It is harmless upstream only because
the shipped ``synthetic_data/*.mat`` matrices are already integer 0/1; on real
z-transformed FC (every |r| < 1) it zeroes the entire matrix. Its edge budget is
likewise hardcoded to 1216 entries, which is meaningful only at 90x90.

Instead we consume the SAME ``LongitudinalSubjectDataset`` items the GELSTM
adapter consumes, so both models see byte-identical FC matrices, the same visit
filtering and the same subject set — that identity is what makes the head-to-head
comparison fair. Only the adjacency construction is Brain-TokenGT's own (upstream
uses a global top-k over the FC matrix; GELSTM uses kNN), because adjacency
construction is part of each method.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch

# Upstream edge budget: 1216 retained entries of a 90x90 FC matrix.
UPSTREAM_EDGE_DENSITY: float = 1216.0 / (90.0 * 90.0)  # ~= 0.1501

_ADJACENCY_METRICS = ("raw", "abs")


def build_adjacency(
    fc: torch.Tensor,
    *,
    edge_density: float = UPSTREAM_EDGE_DENSITY,
    metric: str = "raw",
) -> torch.Tensor:
    """Weighted top-k adjacency, generalising upstream's fixed 1216-entry budget.

    Upstream (``datasets.py:31-32``)::

        entries   = torch.topk(FC.flatten(), 1216).values
        denseAdj  = torch.where(FC >= entries[-1], 1, 0)

    i.e. a global top-k over the whole matrix at a density of 1216/90^2. Here the
    *density* transfers rather than the raw count, so the same fraction of entries
    is retained at any ROI count. Retained entries keep their FC value; the model
    binarises them when ``edge_weight_mode="binary"`` (upstream's behaviour), so
    this is a superset of upstream's output, not a change to it.

    ``metric="raw"`` reproduces upstream (top-k over signed values, i.e. the
    strongest positive correlations). ``metric="abs"`` ranks by |FC|, retaining
    strong anticorrelations too — the convention the GELSTM pipeline uses.
    """
    if metric not in _ADJACENCY_METRICS:
        raise ValueError(f"metric must be one of {_ADJACENCY_METRICS}, got {metric!r}")
    if not 0.0 < edge_density <= 1.0:
        raise ValueError(f"edge_density must be in (0, 1], got {edge_density}")

    ranked = fc.abs() if metric == "abs" else fc
    k = max(1, int(round(edge_density * ranked.numel())))
    threshold = torch.topk(ranked.flatten(), k).values[-1]
    return torch.where(ranked >= threshold, fc, torch.zeros_like(fc))


def item_to_sequence(
    item: Dict,
    *,
    edge_density: float = UPSTREAM_EDGE_DENSITY,
    metric: str = "raw",
    device=None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """One ``LongitudinalSubjectDataset`` item -> ``(A_list, Nodes_list)``.

    Node features are the FC rows themselves (upstream ``x = FC``), taken from
    ``graph.x`` unchanged so they are identical to what the GELSTM encoder sees.
    """
    A_list: List[torch.Tensor] = []
    Nodes_list: List[torch.Tensor] = []
    for graph in item["graphs"]:
        fc = graph.x.to(torch.float32)
        if device is not None:
            fc = fc.to(device)
        fc = torch.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)
        A_list.append(build_adjacency(fc, edge_density=edge_density, metric=metric))
        Nodes_list.append(fc)
    return A_list, Nodes_list


def window_item(item: Dict, *, max_visits: int | None) -> Dict:
    """Truncate a subject record to its first ``max_visits`` visits.

    Mirrors ``GELSTMAdapter.truncate_to_n_visits`` so both models are restricted
    identically. ``delta_t`` is carried through untouched: Brain-TokenGT has no
    notion of inter-visit interval (its temporal edges are weighted 1), so the
    field is preserved only for provenance.
    """
    if max_visits is None or item["n_scans"] <= max_visits:
        return item
    return {
        **item,
        "graphs": item["graphs"][:max_visits],
        "delta_t": item["delta_t"][:max_visits],
        "visit_months": item["visit_months"][:max_visits],
        "n_scans": max_visits,
    }


def count_static_edges(A_list: Sequence[torch.Tensor]) -> int:
    """Total non-zero spatial edges across the visit sequence (diagnostics)."""
    return int(sum(int((a != 0).sum()) for a in A_list))
