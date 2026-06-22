"""
model/GEC/explain.py — attribution for the flattened-trajectory GEC-MLP.

The longitudinal "GEC" run is a ``LongitudinalMLP`` over a padded flat vector of
frozen-GAAE pooled latents (``[z_1..z_Nmax ‖ Δt ‖ mask]``). Its explanations are about
*which visit* and *which latent dimension* drive the prediction:

  * ``mlp_input_attribution`` — captum IG (or gradient×input fallback) of the logit
    w.r.t. the scaled flat input vector.
  * ``unpack_flat_importance`` — fold that flat importance back into per-visit and
    per-latent-dim arrays using the adapter's layout (k latent dims × max_visits,
    optional Δt block, optional visit-mask block).

The per-latent-dim importance is what the explain adapter back-projects through the
frozen GAAE encoder to obtain a brain-region map.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch


def mlp_input_attribution(
    model,
    x_vec: np.ndarray,
    *,
    device="cpu",
    n_steps: int = 50,
) -> np.ndarray:
    """Per-feature importance of the MLP logit w.r.t. the scaled flat input.

    Uses captum Integrated Gradients when available, else a gradient×input fallback
    (so the notebook still runs before captum is installed). Returns a length
    ``feat_dim`` magnitude array (not normalised — the caller unpacks then normalises).
    """
    x = torch.tensor(np.asarray(x_vec, dtype=np.float32), device=device).reshape(1, -1)
    model.eval()
    try:
        from captum.attr import IntegratedGradients  # type: ignore

        x = x.clone().requires_grad_(True)
        ig = IntegratedGradients(lambda inp: model(inp).reshape(-1))
        attr = ig.attribute(x, n_steps=n_steps).detach().cpu().numpy().reshape(-1)
        return np.abs(attr)
    except ImportError:
        x = x.clone().requires_grad_(True)
        out = model(x).reshape(-1)
        out.backward(torch.ones_like(out))
        grad = x.grad.detach().cpu().numpy().reshape(-1)
        return np.abs(grad * x.detach().cpu().numpy().reshape(-1))


def unpack_flat_importance(
    importance: np.ndarray,
    *,
    k: int,
    max_visits: int,
    use_time_delta: bool,
    append_visit_mask: bool,
) -> Dict[str, np.ndarray]:
    """Fold flat-vector importance into per-visit / per-latent-dim / Δt components.

    Layout (matching ``adapters/gec.py::_records_to_X``):
        ``[ z block: max_visits*k ][ Δt block: max_visits? ][ mask block: max_visits? ]``
    Returns ``{"per_visit": (max_visits,), "per_dim": (k,), "dt": (max_visits,) | None}``,
    each max-normalised.
    """
    importance = np.asarray(importance, dtype=float)
    z_block = importance[: max_visits * k].reshape(max_visits, k)
    per_visit = z_block.sum(axis=1)
    per_dim = z_block.sum(axis=0)
    offset = max_visits * k
    dt = None
    if use_time_delta:
        dt = importance[offset: offset + max_visits]
        per_visit = per_visit + dt  # Δt importance belongs to its visit
        offset += max_visits
    # visit-mask block (if present) is structural padding info — folded into per_visit.
    if append_visit_mask:
        mask_imp = importance[offset: offset + max_visits]
        per_visit = per_visit + mask_imp

    def _norm(a):
        a = np.asarray(a, dtype=float)
        return a / (a.max() or 1.0)

    return {
        "per_visit": _norm(per_visit),
        "per_dim": _norm(per_dim),
        "dt": _norm(dt) if dt is not None else None,
    }


__all__ = ["mlp_input_attribution", "unpack_flat_importance"]
