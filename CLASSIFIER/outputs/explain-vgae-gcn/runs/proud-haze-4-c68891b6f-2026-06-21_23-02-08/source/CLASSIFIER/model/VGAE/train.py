"""VGAE training loop with validation-based early stopping.

Mirrors ``model/GAAE/train.py`` (same batched block-diagonal dense-adjacency
construction, same injected ``wandb_run`` convention, same best-val-loss
early-stopping contract returning a ``state_dict``) but optimises the VGAE
objective: masked adjacency BCE + β·KL, with no feature-reconstruction term.
"""
from __future__ import annotations

import copy
import logging

import torch
from torch_geometric.utils import to_dense_adj
from tqdm.notebook import tqdm

from ..GAAE.utils import create_mask
from .losses import vgae_total_loss


def _combined_dense_adj(edge_index, batch_mask, total_nodes, device):
    """Block-diagonal dense adjacency for a batch (graphs laid out contiguously)."""
    dense_adj = to_dense_adj(edge_index, batch=batch_mask).to(device)
    combined = torch.zeros((total_nodes, total_nodes), device=device)
    start = 0
    for i in range(dense_adj.size(0)):
        n = int((batch_mask == i).sum().item())
        combined[start:start + n, start:start + n] = dense_adj[i, :n, :n]
        start += n
    return combined


def _run_epoch(model, loader, optimizer, device, beta, *, train: bool):
    model.train(train)
    total_loss = total_recon = total_kl = 0.0
    for batch in loader:
        batch = batch.to(device)
        x, edge_index, edge_attr, batch_mask = batch.x, batch.edge_index, batch.edge_attr, batch.batch

        with torch.set_grad_enabled(train):
            _z, mu, logvar, adj_reconstructed = model(x, edge_index, edge_attr)
            adj_true = _combined_dense_adj(edge_index, batch_mask, x.size(0), device)
            mask = create_mask(batch_mask)
            loss, recon, kl = vgae_total_loss(adj_true, adj_reconstructed, mask, mu, logvar, beta)

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()
    n = max(1, len(loader))
    return total_loss / n, total_recon / n, total_kl / n


def train_vgae_with_val(
    model, train_loader, val_loader, optimizer, device,
    *, beta=1.0, epochs=100, early_stopping_patience=25, wandb_run=None,
):
    """Train the VGAE; return ``(best_state_dict, history)`` (best = lowest val loss).

    ``beta`` weights the KL term (β-VGAE). Pass ``wandb_run=None`` to disable logging.
    """
    best_val_loss = float("inf")
    best_model = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "train_recon": [], "train_kl": []}

    outer_bar = tqdm(range(epochs), desc="VGAE Training")
    for epoch in outer_bar:
        tr_loss, tr_recon, tr_kl = _run_epoch(model, train_loader, optimizer, device, beta, train=True)
        va_loss, _va_recon, _va_kl = _run_epoch(model, val_loader, optimizer, device, beta, train=False)

        outer_bar.set_postfix({"Train": f"{tr_loss:.4f}", "Val": f"{va_loss:.4f}"})
        logging.info(f"Epoch {epoch}: train={tr_loss:.6f} (recon={tr_recon:.4f} kl={tr_kl:.4f}) val={va_loss:.6f}")
        if wandb_run is not None:
            wandb_run.log({"train_loss": tr_loss, "val_loss": va_loss,
                           "train_recon": tr_recon, "train_kl": tr_kl, "epoch": epoch})

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_recon"].append(tr_recon)
        history["train_kl"].append(tr_kl)

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            epochs_no_improve = 0
            best_model = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= early_stopping_patience:
            logging.info("Early stopping triggered.")
            break

    return best_model, history
