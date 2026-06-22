"""
model/GELSTM/explain.py — temporal attribution + forward-trace for the recurrent head.

GELSTM / GEGRU consume a sequence of pooled GAAE embeddings (one per visit). The
explanations here are about *time*:

  * ``build_sequence_input`` — reconstruct the per-visit ``[z_t ‖ Δt_t]`` input tensor
    for one subject (mirrors ``utils.encode_batch_sequences`` for a single sequence).
  * ``trace_forward`` — per-visit pooled embeddings, the RNN hidden-state trajectory
    across visits, and the final logit / probability (drives the data-journey + gauge).
  * ``hidden_state_trajectory`` — thin wrapper returning just the (T, hidden) states.
  * ``sequence_integrated_gradients`` — captum IG of the final logit w.r.t. the packed
    sequence input → per-visit and per-latent-dim importance (optional dependency).

All run on one subject ``item`` (dict from ``LongitudinalSubjectDataset``).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch


def _pool(z_nodes: torch.Tensor, graph_pool: str) -> torch.Tensor:
    if graph_pool == "max":
        return z_nodes.max(dim=0).values
    if graph_pool == "sum":
        return z_nodes.sum(dim=0)
    return z_nodes.mean(dim=0)


@torch.no_grad()
def build_sequence_input(
    model,
    item: Dict[str, Any],
    *,
    device="cpu",
    use_time_delta: bool = True,
    graph_pool: str = "mean",
    dim_filter: "np.ndarray | None" = None,
    zero_time_delta: bool = False,
) -> Dict[str, Any]:
    """Reconstruct the (T, input_dim) RNN input for one subject + the raw pooled latents.

    Returns ``{"seq_input": (T, input_dim) tensor, "visit_embeddings": (T, latent) np,
    "filtered": (T, k) np}`` — ``visit_embeddings`` are the standardised pooled latents
    before the optional FDR ``dim_filter`` / Δt are applied (for display).
    """
    model.eval()
    graphs = item["graphs"]
    deltas = item["delta_t"]
    raw_pooled: List[np.ndarray] = []
    steps: List[torch.Tensor] = []
    idx = None
    if dim_filter is not None:
        idx = torch.tensor(np.asarray(dim_filter).copy(), dtype=torch.long, device=device)

    for t, g in enumerate(graphs):
        x = g.x.to(device)
        ei = g.edge_index.to(device)
        ea = g.edge_attr.to(device) if g.edge_attr is not None else None
        if hasattr(model, "encode_visit"):
            z_t = model.encode_visit(x, ei, ea, pool=graph_pool)
        else:
            z_t = _pool(model.encode(x, ei, ea), graph_pool)
        raw_pooled.append(z_t.detach().cpu().numpy())
        if idx is not None:
            z_t = z_t[idx]
        if use_time_delta or zero_time_delta:
            dt_val = 0.0 if zero_time_delta else float(deltas[t])
            z_t = torch.cat([z_t, torch.tensor([dt_val], dtype=torch.float, device=device)], dim=0)
        steps.append(z_t)

    seq = torch.stack(steps, dim=0)  # (T, input_dim)
    filtered = seq[:, : seq.shape[1] - (1 if (use_time_delta or zero_time_delta) else 0)]
    return {
        "seq_input": seq,
        "visit_embeddings": np.stack(raw_pooled),
        "filtered": filtered.detach().cpu().numpy(),
    }


@torch.no_grad()
def trace_forward(
    model,
    item: Dict[str, Any],
    *,
    device="cpu",
    use_time_delta: bool = True,
    graph_pool: str = "mean",
    dim_filter: "np.ndarray | None" = None,
) -> Dict[str, Any]:
    """Run the RNN step-by-step for one subject; capture hidden states + final probability."""
    model.eval()
    seq = build_sequence_input(
        model, item, device=device, use_time_delta=use_time_delta,
        graph_pool=graph_pool, dim_filter=dim_filter,
    )
    inp = seq["seq_input"].unsqueeze(0)  # (1, T, input_dim)
    if getattr(model, "rnn_type", "lstm") == "gru":
        output, h_n = model.lstm(inp)
    else:
        output, (h_n, _c) = model.lstm(inp)
    hidden_states = output.squeeze(0)               # (T, hidden) per-visit hidden state
    logit = model.classifier(h_n[-1]).squeeze(-1)   # final-visit prediction
    prob = torch.sigmoid(logit).item()

    out = {
        "visit_embeddings": seq["visit_embeddings"],   # (T, latent)
        "seq_input": seq["seq_input"].detach().cpu().numpy(),
        "hidden_states": hidden_states.detach().cpu().numpy(),
        "logit": float(logit.item()),
        "prob": float(prob),
        "visit_months": list(item.get("visit_months", [])),
    }
    out["stages"] = [
        ("per-visit pooled latent", out["visit_embeddings"].shape),
        ("RNN input [z‖Δt]", out["seq_input"].shape),
        ("RNN hidden trajectory", out["hidden_states"].shape),
        ("final logit → P", (1,)),
    ]
    return out


def hidden_state_trajectory(model, item, **kwargs) -> np.ndarray:
    """(T, hidden) RNN hidden-state trajectory across the subject's visits."""
    return trace_forward(model, item, **kwargs)["hidden_states"]


def sequence_integrated_gradients(
    model,
    item: Dict[str, Any],
    *,
    device="cpu",
    use_time_delta: bool = True,
    graph_pool: str = "mean",
    dim_filter: "np.ndarray | None" = None,
    n_steps: int = 50,
) -> Dict[str, np.ndarray]:
    """captum IG of the final logit w.r.t. the (T, input_dim) sequence input.

    Returns ``{"per_visit": (T,), "per_dim": (input_dim,), "attribution": (T, input_dim)}``
    (magnitudes, max-normalised). Requires captum.
    """
    try:
        from captum.attr import IntegratedGradients  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "sequence_integrated_gradients needs captum "
            "(pip install -r CLASSIFIER/requirements-explain.txt)."
        ) from exc

    seq = build_sequence_input(
        model, item, device=device, use_time_delta=use_time_delta,
        graph_pool=graph_pool, dim_filter=dim_filter,
    )["seq_input"].unsqueeze(0).clone().requires_grad_(True)

    def _logit(inp):
        if getattr(model, "rnn_type", "lstm") == "gru":
            _o, h_n = model.lstm(inp)
        else:
            _o, (h_n, _c) = model.lstm(inp)
        return model.classifier(h_n[-1]).squeeze(-1)

    model.eval()
    ig = IntegratedGradients(_logit)
    attr = ig.attribute(seq, n_steps=n_steps).squeeze(0).detach().cpu().numpy()  # (T, input_dim)
    mag = np.abs(attr)
    per_visit = mag.sum(axis=1)
    per_dim = mag.sum(axis=0)
    return {
        "attribution": mag,
        "per_visit": per_visit / (per_visit.max() or 1.0),
        "per_dim": per_dim / (per_dim.max() or 1.0),
    }


__all__ = [
    "build_sequence_input",
    "trace_forward",
    "hidden_state_trajectory",
    "sequence_integrated_gradients",
]
