from __future__ import annotations

import warnings

import numpy as np
import torch
from torch_geometric.data import Data


def perturb_graph(
    sample: Data,
    method: str,
    noise_level: float,
    rng: np.random.Generator | None = None,
) -> Data:
    """
    Return a perturbed copy of a PyG Data sample.

    method: 'none' | 'feature_noise' | 'edge_perturbation' | 'conditioning_noise'
    noise_level: perturbation magnitude in [0, 1].
    rng: numpy Generator for reproducibility. Falls back to global state if None,
         emitting a DeprecationWarning (matches make_batches pattern).
    """
    if rng is None:
        warnings.warn(
            "perturb_graph called without explicit rng — using global state. "
            "Pass rng=make_rng(seed) for reproducibility.",
            DeprecationWarning,
            stacklevel=2,
        )

    d = sample.clone()

    if noise_level == 0.0 or method == "none":
        return d

    if method == "feature_noise":
        base_std = float(d.x.std().item()) if d.x.numel() > 1 else 1.0
        if rng is not None:
            noise = torch.tensor(
                rng.normal(0.0, base_std * noise_level, size=tuple(d.x.shape)),
                dtype=d.x.dtype,
                device=d.x.device,
            )
        else:
            noise = torch.randn_like(d.x) * (base_std * noise_level)
        d.x = d.x + noise

    elif method == "edge_perturbation":
        edge_index = d.edge_index.detach().cpu()
        num_edges = edge_index.size(1)
        keep = max(1, int(round(num_edges * (1.0 - noise_level))))

        if rng is not None:
            keep_idx = torch.from_numpy(rng.choice(num_edges, size=keep, replace=False))
        else:
            keep_idx = torch.randperm(num_edges)[:keep]

        kept_edges = edge_index[:, keep_idx]
        add_count = int(round(num_edges * noise_level))
        n_nodes = d.x.size(0)

        if add_count > 0:
            if rng is not None:
                new_src = torch.from_numpy(rng.integers(0, n_nodes, size=add_count))
                new_dst = torch.from_numpy(rng.integers(0, n_nodes, size=add_count))
            else:
                new_src = torch.randint(0, n_nodes, (add_count,), dtype=torch.long)
                new_dst = torch.randint(0, n_nodes, (add_count,), dtype=torch.long)
            valid = new_src != new_dst
            new_src, new_dst = new_src[valid], new_dst[valid]
            if new_src.numel() > 0:
                add_edges = torch.stack([new_src.long(), new_dst.long()], dim=0)
                d.edge_index = torch.cat([kept_edges, add_edges], dim=1)
            else:
                d.edge_index = kept_edges
        else:
            d.edge_index = kept_edges

        if hasattr(d, "edge_attr") and d.edge_attr is not None:
            d.edge_attr = torch.ones(d.edge_index.size(1), dtype=d.edge_attr.dtype)

    elif method == "conditioning_noise":
        if hasattr(d, "patient_age") and d.patient_age is not None:
            age = float(d.patient_age.item()) if torch.is_tensor(d.patient_age) else float(d.patient_age)
            delta = float(rng.normal(0.0, noise_level * 0.05) if rng is not None
                          else np.random.normal(0.0, noise_level * 0.05))
            d.patient_age = torch.tensor(age + delta, dtype=torch.float32)

        if hasattr(d, "patient_sex") and d.patient_sex is not None:
            sex = float(d.patient_sex.item()) if torch.is_tensor(d.patient_sex) else float(d.patient_sex)
            delta = float(rng.normal(0.0, noise_level * 0.1) if rng is not None
                          else np.random.normal(0.0, noise_level * 0.1))
            d.patient_sex = torch.tensor(
                float(np.clip(sex + delta, 0.0, 1.0)), dtype=torch.float32
            )

    else:
        raise ValueError(
            f"Unknown perturbation method: {method!r}. "
            "Expected 'none', 'feature_noise', 'edge_perturbation', or 'conditioning_noise'."
        )

    return d
