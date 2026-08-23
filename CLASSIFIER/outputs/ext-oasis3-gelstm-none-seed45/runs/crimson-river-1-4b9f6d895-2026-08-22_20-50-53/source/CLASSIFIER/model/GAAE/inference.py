from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import unbatch

if TYPE_CHECKING:
    from CLASSIFIER.model.GAAE.models import GraphAttentionAutoencoderConditioned


def extract_embeddings(
    model: "GraphAttentionAutoencoderConditioned",
    dataset,
    device: torch.device | str,
    batch_size: int = 32,
    graph_pool: Literal["mean", "max", "sum"] = "mean",
) -> tuple[np.ndarray, list[str]]:
    """
    Encode all samples in dataset → (N, F) graph-level embeddings + raw patient IDs.

    Patient IDs are returned as-is; callers strip scan suffixes (e.g. split('_')[0])
    if subject-level grouping is needed for CV.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_emb: list[np.ndarray] = []
    all_pids: list[str] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            z = model.encode(batch.x, batch.edge_index, batch.edge_attr)
            z_list = unbatch(z, batch.batch)
            for z_g in z_list:
                if graph_pool == "mean":
                    emb = z_g.mean(0)
                elif graph_pool == "max":
                    emb = z_g.max(0).values
                else:
                    emb = z_g.sum(0)
                all_emb.append(emb.cpu().numpy())

            pids = batch.patient_id
            if isinstance(pids, (list, tuple)):
                all_pids.extend([str(p) for p in pids])
            else:
                all_pids.append(str(pids))

    return np.stack(all_emb), all_pids
