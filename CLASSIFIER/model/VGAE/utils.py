"""VGAE inference helpers — mirror ``model/GAAE/utils.load_gaae_for_inference``."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .models import VariationalGraphAutoencoder


def load_vgae_for_inference(
    ckpt_path: Path | str,
    in_features: int,
    config: dict,
    device: torch.device | str = "cpu",
) -> "VariationalGraphAutoencoder":
    """Instantiate and load a frozen VGAE encoder for downstream inference.

    config keys used: hidden_dim, latent_dim, conv_type, num_heads, dropout.
    Handles both the full-state checkpoint dict and a bare ``state_dict`` / module
    (see .claude/rules/checkpoints.md).
    """
    from .models import VariationalGraphAutoencoder

    model = VariationalGraphAutoencoder(
        in_features=in_features,
        hidden_dim=config.get("hidden_dim", 128),
        latent_dim=config.get("latent_dim", 64),
        conv_type=config.get("conv_type", "gcn"),
        num_heads=config.get("num_heads", 2),
        dropout=config.get("dropout", 0.3),
    )
    obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(obj, torch.nn.Module):
        model = obj
    elif isinstance(obj, dict):
        model.load_state_dict(obj.get("model_state_dict", obj))
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(obj)}.")
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
