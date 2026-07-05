"""
model/GAAE/explain.py — attribution + forward-trace helpers for the GAAE encoder.

The GAAE is the only genuinely graph-level model in the stack (GEC / GELSTM consume
its pooled latents), so the region-level explanations live here:

  * ``aggregate_gat_attention`` — fold the per-layer GATv2 attention into a per-ROI
    "received attention" score.
  * ``per_node_reconstruction_error`` — per-ROI autoencoder reconstruction error.
  * ``reconstruction_quality`` — residual matrix + element-wise error metrics
    (MSE/RMSE/MAE/NRMSE) and input↔reconstruction Pearson r / R² with a fidelity band.
  * ``trace_forward`` — capture every intermediate of one forward pass (input, each
    encoder/decoder GAT layer's node embeddings, attention, latent z, pooled graph
    embedding, reconstruction) for the data-journey visualisation.
  * ``latent_dim_integrated_gradients`` — captum IG of one latent dimension w.r.t. the
    input ROI features (optional dependency; raises a clear error if captum absent).
  * ``gnn_explain_latent_dim`` — ``torch_geometric.explain`` node/edge importance for a
    chosen latent dimension's pooled value.

These operate on a single PyG ``Data`` (one visit graph). The encoder is used without
FiLM conditioning, matching how GEC / GELSTM pool embeddings (``enc.encode(x, ei, ea)``).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def _edge_attr(data) -> Optional[torch.Tensor]:
    return getattr(data, "edge_attr", None)


def aggregate_gat_attention(
    attention_weights: List[Any], n_nodes: int, *, reduce: str = "mean"
) -> np.ndarray:
    """Per-node received-attention from a list of ``(edge_index, alpha)`` GAT tuples.

    For each layer, attention ``alpha`` (E, heads) is averaged over heads and
    scatter-summed onto its destination node, then divided by the node in-degree.
    Layers are combined by ``reduce`` ("mean" | "sum" | "last"). Returns a length
    ``n_nodes`` array, max-normalised to [0, 1].
    """
    per_layer = []
    for layer in attention_weights:
        edge_index, alpha = layer
        dst = edge_index[1].detach().cpu()
        a = alpha.detach().cpu()
        if a.dim() > 1:
            a = a.mean(dim=1)
        recv = torch.zeros(n_nodes)
        cnt = torch.zeros(n_nodes)
        recv.scatter_add_(0, dst, a.float())
        cnt.scatter_add_(0, dst, torch.ones_like(a.float()))
        per_layer.append((recv / cnt.clamp_min(1.0)).numpy())
    if not per_layer:
        return np.zeros(n_nodes, dtype=float)
    stack = np.stack(per_layer)
    if reduce == "sum":
        out = stack.sum(0)
    elif reduce == "last":
        out = stack[-1]
    else:
        out = stack.mean(0)
    return out / (out.max() or 1.0)


@torch.no_grad()
def per_node_reconstruction_error(model, data, *, device="cpu") -> np.ndarray:
    """Per-ROI reconstruction error of the GAAE autoencoder for one graph."""
    model.eval()
    x = data.x.to(device)
    ei = data.edge_index.to(device)
    ea = _edge_attr(data)
    ea = ea.to(device) if ea is not None else None
    z = model.encode(x, ei, ea)
    x_rec = model.decode_features(z, ei, ea)
    err = ((x - x_rec) ** 2).mean(dim=1)  # (N,)
    return err.detach().cpu().numpy()


@torch.no_grad()
def reconstruct_features(model, data, *, device="cpu") -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(x, x̂)`` numpy arrays — input node features and the GAAE feature
    reconstruction for one graph (single forward pass, no FiLM conditioning).

    Pairs with :func:`reconstruction_quality` for cohort-level fidelity scoring.
    """
    model.eval()
    x = data.x.to(device)
    ei = data.edge_index.to(device)
    ea = _edge_attr(data)
    ea = ea.to(device) if ea is not None else None
    z = model.encode(x, ei, ea)
    x_rec = model.decode_features(z, ei, ea)
    return x.detach().cpu().numpy(), x_rec.detach().cpu().numpy()


def reconstruction_quality(x, x_recon) -> Dict[str, Any]:
    """Quantify FC-matrix reconstruction fidelity for one graph (numpy-only).

    Compares the input node-feature matrix ``x`` (the Fisher-z FC rows) with the
    decoder output ``x_recon`` and returns the residual plus element-wise error
    metrics and the input↔reconstruction agreement (Pearson r, R²).

    Returns ``{"residual" (N, F), "mse", "rmse", "mae", "nrmse", "pearson_r", "r2",
    "quality"}``. ``nrmse`` is RMSE divided by the input's standard deviation, so it
    reads as "error relative to the FC signal's own spread". ``quality`` bins the
    Pearson r with the rule-of-thumb ``r≥0.90 excellent · 0.80–0.90 good ·
    0.60–0.80 fair · <0.60 poor`` commonly used for autoencoder reconstruction of
    parcellated-fMRI functional connectivity (correlation is reported far more often
    than absolute error because the z-FC scale is dataset-specific).
    """
    a = np.asarray(x, dtype=float)
    b = np.asarray(x_recon, dtype=float)
    if a.shape != b.shape:
        raise ValueError(
            f"reconstruction_quality: shape mismatch x{a.shape} vs x_recon{b.shape}."
        )
    residual = b - a
    flat_a, flat_b = a.ravel(), b.ravel()
    mse = float(np.mean(residual ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residual)))
    input_std = float(flat_a.std())
    nrmse = float(rmse / input_std) if input_std > 0 else float("nan")
    pearson_r = (
        float(np.corrcoef(flat_a, flat_b)[0, 1])
        if flat_a.std() > 0 and flat_b.std() > 0
        else float("nan")
    )
    ss_tot = float(np.sum((flat_a - flat_a.mean()) ** 2))
    r2 = float(1.0 - np.sum(residual ** 2) / ss_tot) if ss_tot > 0 else float("nan")

    if np.isnan(pearson_r):
        quality = "undefined"
    elif pearson_r >= 0.90:
        quality = "excellent"
    elif pearson_r >= 0.80:
        quality = "good"
    elif pearson_r >= 0.60:
        quality = "fair"
    else:
        quality = "poor"

    return {
        "residual": residual,
        "mse": mse, "rmse": rmse, "mae": mae, "nrmse": nrmse,
        "pearson_r": pearson_r, "r2": r2, "quality": quality,
    }


@torch.no_grad()
def trace_forward(model, data, *, device="cpu") -> Dict[str, Any]:
    """Capture intermediates of one GAAE forward pass for the data-journey plots.

    Returns a dict of numpy arrays / shapes: ``x`` (N, F), ``gat1`` / ``gat2``
    (N, hidden*heads), ``latent`` (N, latent), ``pooled`` (latent,), ``recon`` (N, F),
    ``attention`` (list of per-node attention arrays, one per encoder GAT layer), and
    ``stages`` (ordered [(name, shape), …] for printing the transformation).
    """
    model.eval()
    x = data.x.to(device)
    ei = data.edge_index.to(device)
    ea = _edge_attr(data)
    ea = ea.to(device) if ea is not None else None
    n_nodes = x.shape[0]

    captured: Dict[str, torch.Tensor] = {}
    handles = []
    for name in ("encoder_gat1", "encoder_gat2", "encoder_gat3",
                 "decoder_gat1", "decoder_gat2", "decoder_gat3"):
        layer = getattr(model, name)

        def _hook(_mod, _inp, out, _name=name):
            captured[_name] = out[0] if isinstance(out, tuple) else out

        handles.append(layer.register_forward_hook(_hook))

    z, attn = model.encode(x, ei, ea, return_attention=True)
    pooled = z.mean(dim=0)
    x_rec = model.decode_features(z, ei, ea)
    for h in handles:
        h.remove()
    attn_nodes = [aggregate_gat_attention([layer], n_nodes) for layer in attn]

    def _np(t):
        return t.detach().cpu().numpy()

    out = {
        "x": _np(x),
        "gat1": _np(captured.get("encoder_gat1", z)),
        "gat2": _np(captured.get("encoder_gat2", z)),
        "latent": _np(z),
        "pooled": _np(pooled),
        "decoder_gat1": _np(captured.get("decoder_gat1", x_rec)),
        "decoder_gat2": _np(captured.get("decoder_gat2", x_rec)),
        "recon": _np(x_rec),
        "attention": attn_nodes,
    }
    out["stages"] = [
        ("input x (FC rows)", out["x"].shape),
        ("encoder_gat1", out["gat1"].shape),
        ("encoder_gat2", out["gat2"].shape),
        ("latent z (per-node)", out["latent"].shape),
        ("pooled graph embedding", out["pooled"].shape),
        ("decoder_gat1", out["decoder_gat1"].shape),
        ("decoder_gat2", out["decoder_gat2"].shape),
        ("reconstruction x̂", out["recon"].shape),
    ]
    return out


class _PooledLatentDim(nn.Module):
    """Wrap the GAAE encoder so it returns one pooled latent dimension (graph scalar).

    Target for captum / GNNExplainer, both of which want a model that maps node
    features → a scalar graph output.
    """

    def __init__(self, encoder, dim: int):
        super().__init__()
        self.encoder = encoder
        self.dim = int(dim)

    def forward(self, x, edge_index, edge_attr=None, **_):
        z = self.encoder.encode(x, edge_index, edge_attr)
        return z.mean(dim=0)[self.dim].reshape(1)


def latent_dim_integrated_gradients(
    model, data, dim: int, *, device="cpu", n_steps: int = 50
) -> np.ndarray:
    """captum Integrated Gradients of pooled latent ``dim`` w.r.t. input ROI features.

    Returns a per-ROI importance array (N,) = summed |attribution| over features,
    max-normalised. Requires captum (``requirements-explain.txt``).
    """
    try:
        from captum.attr import IntegratedGradients  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "latent_dim_integrated_gradients needs captum "
            "(pip install -r CLASSIFIER/requirements-explain.txt)."
        ) from exc

    wrapper = _PooledLatentDim(model, dim).to(device).eval()
    x = data.x.to(device).clone().requires_grad_(True)
    ei = data.edge_index.to(device)
    ea = _edge_attr(data)
    ea = ea.to(device) if ea is not None else None
    ig = IntegratedGradients(lambda inp: wrapper(inp, ei, ea))
    attr = ig.attribute(x, n_steps=n_steps)
    node_imp = attr.abs().sum(dim=1).detach().cpu().numpy()
    return node_imp / (node_imp.max() or 1.0)


def gnn_explain_latent_dim(
    model, data, dim: int, *, device="cpu", epochs: int = 100
) -> Dict[str, np.ndarray]:
    """``torch_geometric.explain`` node/edge importance for pooled latent ``dim``.

    Returns ``{"node_importance": (N,), "edge_importance": (E,)}`` (max-normalised).
    """
    from torch_geometric.explain import Explainer, GNNExplainer
    from torch_geometric.explain.config import ModelConfig

    wrapper = _PooledLatentDim(model, dim).to(device).eval()
    x = data.x.to(device)
    ei = data.edge_index.to(device)
    ea = _edge_attr(data)
    ea = ea.to(device) if ea is not None else None

    explainer = Explainer(
        model=wrapper,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=ModelConfig(mode="regression", task_level="graph", return_type="raw"),
    )
    explanation = explainer(x, ei, edge_attr=ea)
    node_mask = explanation.node_mask.detach().cpu().numpy()
    node_imp = np.abs(node_mask).sum(axis=1)
    edge_imp = (
        np.abs(explanation.edge_mask.detach().cpu().numpy())
        if explanation.get("edge_mask") is not None
        else np.zeros(ei.shape[1])
    )
    return {
        "node_importance": node_imp / (node_imp.max() or 1.0),
        "edge_importance": edge_imp / (edge_imp.max() or 1.0),
    }


@torch.no_grad()
def steer_along_axis(
    model, data, w_hat: np.ndarray, sigma: float, *,
    scales: Optional[np.ndarray] = None, device="cpu",
) -> Dict[str, Any]:
    """Steer one subject's baseline graph along a disease-axis direction and decode.

    ``w_hat`` is a unit steering direction in latent space (e.g. from
    ``common.explain.disease_axis_projection``); ``sigma`` is the population std of
    the pooled latent space (1 step = 1 std), passed explicitly by the caller rather
    than recomputed here — see ``.claude/rules/errors.md`` on not hiding implicit
    defaults. Returns ``{"scales", "baseline_fc", "steered_fcs"}``, where
    ``baseline_fc``/each entry of ``steered_fcs`` is the (N_nodes, in_features)
    decoder reconstruction at that scale.
    """
    if scales is None:
        scales = np.linspace(-3.0, 3.0, 11)
    model.eval()
    x = data.x.to(device)
    ei = data.edge_index.to(device)
    ea = _edge_attr(data)
    ea = ea.to(device) if ea is not None else None
    z = model.encode(x, ei, ea)
    w = torch.tensor(np.asarray(w_hat, dtype=np.float32), device=device)

    baseline_fc = model.decode_features(z, ei, ea).cpu().numpy()
    steered_fcs = []
    for scale in scales:
        shift = float(scale * sigma) * w.unsqueeze(0)
        fc = model.decode_features(z + shift, ei, ea).cpu().numpy()
        steered_fcs.append(fc)
    return {"scales": np.asarray(scales, dtype=float), "baseline_fc": baseline_fc,
            "steered_fcs": steered_fcs}


@torch.no_grad()
def reconstruct_sorted_by_score(model, subjects: List[Dict[str, Any]], *, device="cpu") -> Dict[str, Any]:
    """Encode+decode each subject's baseline graph; pair with its disease score/label.

    ``subjects`` is a list of ``{"data": <PyG Data>, "score": float, "label": int}``
    — already selected/sorted by the caller (e.g. evenly sampled across the disease-
    score range). Returns aligned ``{"gt", "recon", "scores", "labels"}`` lists for
    the ground-truth-vs-reconstruction grid plot.
    """
    model.eval()
    gt, recon, scores, labels = [], [], [], []
    for subj in subjects:
        data = subj["data"]
        x = data.x.to(device)
        ei = data.edge_index.to(device)
        ea = _edge_attr(data)
        ea = ea.to(device) if ea is not None else None
        z = model.encode(x, ei, ea)
        x_rec = model.decode_features(z, ei, ea)
        gt.append(x.cpu().numpy())
        recon.append(x_rec.cpu().numpy())
        scores.append(float(subj["score"]))
        labels.append(subj["label"])
    return {"gt": gt, "recon": recon, "scores": scores, "labels": labels}


__all__ = [
    "aggregate_gat_attention",
    "per_node_reconstruction_error",
    "reconstruct_features",
    "reconstruction_quality",
    "trace_forward",
    "latent_dim_integrated_gradients",
    "gnn_explain_latent_dim",
    "steer_along_axis",
    "reconstruct_sorted_by_score",
]
