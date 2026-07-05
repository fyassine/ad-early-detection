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
from .losses import raw_kl_stats, vgae_total_loss


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


def _run_epoch(model, loader, optimizer, device, beta, *, train: bool,
               free_bits=0.0, feature_loss_weight=0.0):
    model.train(train)
    total_loss = total_recon = total_kl = total_feat = 0.0
    kl_stats_sum = {"raw_kl_min": 0.0, "raw_kl_mean": 0.0, "raw_kl_max": 0.0, "frac_dims_at_floor": 0.0}
    for batch in loader:
        batch = batch.to(device)
        x, edge_index, edge_attr, batch_mask = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        cond_vec = torch.stack([batch.patient_age, batch.patient_sex.float()], dim=1)

        with torch.set_grad_enabled(train):
            _z, mu, logvar, adj_reconstructed, x_reconstructed = model(
                x, edge_index, edge_attr, cond_vec=cond_vec, batch_mask=batch_mask
            )
            adj_true = _combined_dense_adj(edge_index, batch_mask, x.size(0), device)
            mask = create_mask(batch_mask)
            loss, recon, kl, feat = vgae_total_loss(
                adj_true, adj_reconstructed, mask, mu, logvar, beta,
                free_bits=free_bits, x_original=x, x_reconstructed=x_reconstructed,
                feature_loss_weight=feature_loss_weight,
            )

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()
        total_feat += feat.item()
        with torch.no_grad():
            for k, v in raw_kl_stats(mu, logvar, free_bits=free_bits).items():
                kl_stats_sum[k] += v
    n = max(1, len(loader))
    kl_stats = {k: v / n for k, v in kl_stats_sum.items()}
    return total_loss / n, total_recon / n, total_kl / n, total_feat / n, kl_stats


def train_vgae_with_val(
    model, train_loader, val_loader, optimizer, device,
    *, beta=1.0, beta_warmup_epochs=0, free_bits=0.0, feature_loss_weight=0.0,
    epochs=100, early_stopping_patience=25, wandb_run=None, on_epoch_end=None,
):
    """Train the VGAE; return ``(best_state_dict, history)`` (best = lowest val loss).

    ``on_epoch_end``, if given, is called as ``on_epoch_end(epoch, epoch_metrics)``
    after each epoch's history is recorded (``epoch_metrics`` is a dict with that
    epoch's ``train_loss``/``val_loss``/``train_recon``/``raw_kl_mean``/
    ``frac_dims_at_floor``/etc.). Returning ``True`` stops training immediately —
    this is a generic hook, not Optuna-specific (this module stays free of any
    tuning-library import); callers like a hyperparameter sweep can use it to bail
    out of a clearly-collapsing trial well before the epoch budget is exhausted,
    rather than paying the full cost just to discover the same thing at the end.

    ``beta`` weights the KL term (β-VGAE) — its target value once warmup completes.
    ``beta_warmup_epochs`` linearly ramps the KL weight from 0 up to ``beta`` over the
    first N epochs (0 = constant ``beta`` from epoch 0, the prior behaviour).

    Without warmup, the KL gradient (trivially satisfied by collapsing mu/logvar to the
    prior) can dominate the weaker, adjacency-only reconstruction signal before the
    encoder learns any real structure — posterior collapse. This was observed on the
    DELCODE whole-brain pretrain runs: train_kl -> ~0 within ~10-20 epochs and train_recon
    flatlines at the constant-p=0.5 BCE floor for the rest of training (see
    CLASSIFIER/outputs/explain-vgae-*/latest/run.log and the W&B train_recon/train_kl
    curves for vgae-gcn-static / vgae-gat-static).

    Additional anti-collapse knobs (both no-ops at their defaults):
      ``free_bits`` — per-latent-dimension KL floor (nats); see ``losses.kl_divergence``.
      ``feature_loss_weight`` — weight on the node-feature reconstruction MSE; requires
        the model to have been built with ``feature_decoder=True`` (otherwise
        ``x_reconstructed`` is ``None`` and the term stays zero).

    Early stopping is evaluated only once the β warmup has finished (``current_beta``
    has reached its target ``beta``). ``val_loss = recon + beta*kl + feat_weight*feat``
    rises every epoch while beta ramps up, purely from the beta multiplier, even when
    recon/kl/feat are flat or improving — comparing val_loss across warmup epochs would
    pick the first (smallest-beta, least-trained) epoch as "best" and exhaust patience
    almost immediately. While warming up, ``best_model`` tracks the latest state_dict
    (a fallback only) and ``epochs_no_improve`` does not advance. With
    ``beta_warmup_epochs=0`` warmup is already complete at epoch 0, so this is a no-op
    and behaviour for existing constant-beta configs is unchanged.

    Pass ``wandb_run=None`` to disable logging.
    """
    best_val_loss = float("inf")
    best_model = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "train_recon": [], "train_kl": [],
               "train_feat": [], "val_recon": [], "val_feat": [], "beta": [],
               "raw_kl_min": [], "raw_kl_mean": [], "raw_kl_max": [], "frac_dims_at_floor": []}

    outer_bar = tqdm(range(epochs), desc="VGAE Training")
    for epoch in outer_bar:
        current_beta = (
            beta * min(1.0, (epoch + 1) / beta_warmup_epochs) if beta_warmup_epochs > 0 else beta
        )
        tr_loss, tr_recon, tr_kl, tr_feat, tr_kl_stats = _run_epoch(
            model, train_loader, optimizer, device, current_beta, train=True,
            free_bits=free_bits, feature_loss_weight=feature_loss_weight,
        )
        va_loss, va_recon, _va_kl, va_feat, _va_kl_stats = _run_epoch(
            model, val_loader, optimizer, device, current_beta, train=False,
            free_bits=free_bits, feature_loss_weight=feature_loss_weight,
        )

        outer_bar.set_postfix({"Train": f"{tr_loss:.4f}", "Val": f"{va_loss:.4f}", "beta": f"{current_beta:.4f}"})
        logging.info(
            f"Epoch {epoch}: train={tr_loss:.6f} (recon={tr_recon:.4f} kl={tr_kl:.4f} "
            f"feat={tr_feat:.4f} beta={current_beta:.4f}) val={va_loss:.6f} "
            f"(val_recon={va_recon:.4f}) "
            f"raw_kl=[{tr_kl_stats['raw_kl_min']:.4f}, {tr_kl_stats['raw_kl_mean']:.4f}, "
            f"{tr_kl_stats['raw_kl_max']:.4f}] frac_at_floor={tr_kl_stats['frac_dims_at_floor']:.2f}"
        )
        if wandb_run is not None:
            wandb_run.log({"train_loss": tr_loss, "val_loss": va_loss,
                           "train_recon": tr_recon, "train_kl": tr_kl,
                           "train_feat": tr_feat, "val_recon": va_recon, "val_feat": va_feat,
                           "beta": current_beta, "epoch": epoch, **tr_kl_stats})

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_recon"].append(tr_recon)
        history["train_kl"].append(tr_kl)
        history["train_feat"].append(tr_feat)
        history["val_recon"].append(va_recon)
        history["val_feat"].append(va_feat)
        history["beta"].append(current_beta)
        history["raw_kl_min"].append(tr_kl_stats["raw_kl_min"])
        history["raw_kl_mean"].append(tr_kl_stats["raw_kl_mean"])
        history["raw_kl_max"].append(tr_kl_stats["raw_kl_max"])
        history["frac_dims_at_floor"].append(tr_kl_stats["frac_dims_at_floor"])

        warmed_up = current_beta >= beta
        if not warmed_up:
            # val_loss isn't comparable across epochs while beta is still ramping up
            # (the beta*kl term alone can make it rise epoch over epoch regardless of
            # model quality) — keep the latest state as a fallback, don't early-stop.
            best_model = copy.deepcopy(model.state_dict())
        elif va_loss < best_val_loss:
            best_val_loss = va_loss
            epochs_no_improve = 0
            best_model = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
        if warmed_up and epochs_no_improve >= early_stopping_patience:
            logging.info("Early stopping triggered.")
            break

        if on_epoch_end is not None:
            epoch_metrics = {
                "train_loss": tr_loss, "val_loss": va_loss, "train_recon": tr_recon,
                "train_kl": tr_kl, "train_feat": tr_feat, "val_recon": va_recon,
                "val_feat": va_feat, "beta": current_beta, "warmed_up": warmed_up,
                **tr_kl_stats,
            }
            if on_epoch_end(epoch, epoch_metrics):
                logging.info("Stopped via on_epoch_end callback.")
                break

    return best_model, history
