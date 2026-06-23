"""Tests for model.VGAE — forward shapes, encode contract, and the VGAE loss.

CPU-only, tiny synthetic graphs (no DELCODE matrices, no GAAE checkpoint). The
full training path (data loading, GAAE pretrain, notebook orchestration) is
exercised end-to-end by the experiment runner, not here — but the early-stopping
*logic* inside ``train_vgae_with_val`` is pure control flow and is unit-tested
below on tiny synthetic loaders.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dense_to_sparse, to_dense_adj

from CLASSIFIER.model.GAAE.utils import create_mask
from CLASSIFIER.model.VGAE.losses import kl_divergence, vgae_total_loss
from CLASSIFIER.model.VGAE.models import VariationalGraphAutoencoder
from CLASSIFIER.model.VGAE.train import train_vgae_with_val

IN_FEATURES = 8
N_NODES = 10
LATENT = 4
HIDDEN = 6


def _toy_graph(seed=0):
    rng = np.random.default_rng(seed)
    x = torch.tensor(rng.standard_normal((N_NODES, IN_FEATURES)), dtype=torch.float)
    # symmetric kNN-ish adjacency from a random correlation, no self loops.
    w = rng.random((N_NODES, N_NODES))
    adj = ((w + w.T) > 1.0).astype(float)
    np.fill_diagonal(adj, 0.0)
    edge_index, edge_attr = dense_to_sparse(torch.tensor(adj, dtype=torch.float))
    return x, edge_index, edge_attr


@pytest.mark.parametrize("conv_type", ["gcn", "gat"])
def test_forward_shapes(conv_type):
    x, ei, ea = _toy_graph()
    model = VariationalGraphAutoencoder(IN_FEATURES, HIDDEN, LATENT, conv_type=conv_type, num_heads=2)
    model.train()
    z, mu, logvar, adj_hat, x_hat = model(x, ei, ea)
    assert z.shape == (N_NODES, LATENT)
    assert mu.shape == (N_NODES, LATENT)
    assert logvar.shape == (N_NODES, LATENT)
    assert adj_hat.shape == (N_NODES, N_NODES)
    assert x_hat is None  # no feature decoder by default


@pytest.mark.parametrize("conv_type", ["gcn", "gat"])
def test_encode_is_drop_in_pooling(conv_type):
    """encode() returns a single [N, latent] tensor so `.mean(0)` pools to [latent]."""
    x, ei, ea = _toy_graph(1)
    model = VariationalGraphAutoencoder(IN_FEATURES, HIDDEN, LATENT, conv_type=conv_type)
    model.eval()
    z = model.encode(x, ei, ea)
    assert z.shape == (N_NODES, LATENT)
    pooled = z.mean(0)
    assert pooled.shape == (LATENT,)
    # eval encode == mu (deterministic).
    mu, _ = model.encode_dist(x, ei, ea)
    assert torch.allclose(z, mu)


def test_encode_attention_present_for_gat_empty_for_gcn():
    x, ei, ea = _toy_graph(2)
    gat = VariationalGraphAutoencoder(IN_FEATURES, HIDDEN, LATENT, conv_type="gat")
    _mu, attn = gat.encode(x, ei, ea, return_attention=True)
    assert isinstance(attn, list) and len(attn) >= 1  # GATv2 (edge_index, alpha) tuples
    gcn = VariationalGraphAutoencoder(IN_FEATURES, HIDDEN, LATENT, conv_type="gcn")
    _mu2, attn2 = gcn.encode(x, ei, ea, return_attention=True)
    assert attn2 == []  # GCN has no attention -> region_importance degrades to zeros


@pytest.mark.parametrize("conv_type", ["gcn", "gat"])
def test_film_conditioning_changes_latent_but_keeps_shape(conv_type):
    """FiLM cond_vec modulates the latent; omitting it leaves the pooling path unchanged."""
    x, ei, ea = _toy_graph(5)
    model = VariationalGraphAutoencoder(IN_FEATURES, HIDDEN, LATENT, conv_type=conv_type, num_heads=2)
    model.eval()
    cond_vec = torch.tensor([[0.7, 1.0]], dtype=torch.float)  # one graph: (age, sex)
    batch_mask = torch.zeros(N_NODES, dtype=torch.long)

    mu_plain = model.encode(x, ei, ea)
    mu_cond = model.encode(x, ei, ea, cond_vec=cond_vec, batch_mask=batch_mask)
    assert mu_cond.shape == mu_plain.shape == (N_NODES, LATENT)
    # Conditioning is the single source of truth: forward() with cond matches encode().
    _z, mu_fwd, _lv, adj_hat, _x_hat = model(x, ei, ea, cond_vec=cond_vec, batch_mask=batch_mask)
    assert torch.allclose(mu_fwd, mu_cond)
    assert adj_hat.shape == (N_NODES, N_NODES)
    # FiLM (gamma*mu+beta) actually moves the latent away from the un-conditioned one
    # (film MLPs are randomly initialised but non-degenerate).
    assert not torch.allclose(mu_cond, mu_plain)


def test_invalid_conv_type_raises():
    with pytest.raises(ValueError, match="conv_type"):
        VariationalGraphAutoencoder(IN_FEATURES, HIDDEN, LATENT, conv_type="sage")


def test_kl_divergence_zero_at_standard_normal():
    mu = torch.zeros(N_NODES, LATENT)
    logvar = torch.zeros(N_NODES, LATENT)  # var = 1
    assert torch.allclose(kl_divergence(mu, logvar), torch.tensor(0.0), atol=1e-6)
    # Non-trivial posterior -> strictly positive KL.
    assert kl_divergence(torch.ones(N_NODES, LATENT), torch.zeros(N_NODES, LATENT)) > 0


def test_free_bits_floors_per_dim_kl():
    """A near-collapsed posterior has KL clamped up to free_bits*latent_dim."""
    # mu/logvar close to the prior -> raw KL ~ 0; free_bits installs a per-dim floor.
    mu = torch.full((N_NODES, LATENT), 1e-3)
    logvar = torch.zeros(N_NODES, LATENT)
    raw = kl_divergence(mu, logvar, free_bits=0.0)
    floored = kl_divergence(mu, logvar, free_bits=0.5)
    assert raw < floored
    # Each of LATENT dims is floored at 0.5 nats -> total ~= 0.5 * LATENT.
    assert torch.allclose(floored, torch.tensor(0.5 * LATENT), atol=1e-3)
    # free_bits is a floor only: a posterior already above it is left untouched.
    big = kl_divergence(torch.ones(N_NODES, LATENT) * 5, torch.zeros(N_NODES, LATENT))
    assert torch.allclose(big, kl_divergence(torch.ones(N_NODES, LATENT) * 5,
                                             torch.zeros(N_NODES, LATENT), free_bits=0.5))


def test_vgae_total_loss_finite_and_splits():
    x, ei, ea = _toy_graph(3)
    model = VariationalGraphAutoencoder(IN_FEATURES, HIDDEN, LATENT, conv_type="gcn")
    model.train()
    _z, mu, logvar, adj_hat, _x_hat = model(x, ei, ea)
    batch_mask = torch.zeros(N_NODES, dtype=torch.long)
    adj_true = to_dense_adj(ei, batch=batch_mask).squeeze(0)
    mask = create_mask(batch_mask)
    total, recon, kl, feat = vgae_total_loss(adj_true, adj_hat, mask, mu, logvar, beta=0.5)
    assert torch.isfinite(total) and torch.isfinite(recon) and torch.isfinite(kl)
    assert feat.item() == 0.0  # no feature term without a decoder
    assert torch.allclose(total, recon + 0.5 * kl)


def test_feature_decoder_adds_reconstruction_term():
    x, ei, ea = _toy_graph(7)
    model = VariationalGraphAutoencoder(
        IN_FEATURES, HIDDEN, LATENT, conv_type="gcn", feature_decoder=True
    )
    model.train()
    _z, mu, logvar, adj_hat, x_hat = model(x, ei, ea)
    assert x_hat is not None and x_hat.shape == (N_NODES, IN_FEATURES)
    batch_mask = torch.zeros(N_NODES, dtype=torch.long)
    adj_true = to_dense_adj(ei, batch=batch_mask).squeeze(0)
    mask = create_mask(batch_mask)
    total, recon, kl, feat = vgae_total_loss(
        adj_true, adj_hat, mask, mu, logvar, beta=0.5,
        x_original=x, x_reconstructed=x_hat, feature_loss_weight=2.0,
    )
    assert feat.item() > 0.0
    assert torch.allclose(total, recon + 2.0 * feat + 0.5 * kl)


def test_loss_backward_updates_encoder():
    x, ei, ea = _toy_graph(4)
    model = VariationalGraphAutoencoder(IN_FEATURES, HIDDEN, LATENT, conv_type="gcn")
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    batch_mask = torch.zeros(N_NODES, dtype=torch.long)
    adj_true = to_dense_adj(ei, batch=batch_mask).squeeze(0)
    mask = create_mask(batch_mask)
    _z, mu, logvar, adj_hat, _x_hat = model(x, ei, ea)
    total, _r, _k, _f = vgae_total_loss(adj_true, adj_hat, mask, mu, logvar)
    opt.zero_grad()
    total.backward()
    opt.step()
    # at least one encoder parameter received a gradient
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in model.parameters())


def _toy_loader(num_graphs=2, seed=0):
    rng = np.random.default_rng(seed)
    graphs = []
    for i in range(num_graphs):
        x, ei, ea = _toy_graph(seed + i)
        graphs.append(Data(
            x=x, edge_index=ei, edge_attr=ea,
            patient_age=torch.tensor(rng.random(), dtype=torch.float),
            patient_sex=torch.tensor(rng.integers(0, 2), dtype=torch.long),
        ))
    return DataLoader(graphs, batch_size=2, shuffle=False)


def _stub_run_epoch_factory(recon=1.0, kl=2.0, feat=0.0):
    """Deterministic stand-in for ``_run_epoch``: flat recon/kl/feat every epoch,
    so the only thing that can move ``loss`` across epochs is the beta passed in —
    mirrors the observed vgae-*-anticollapse curves (recon/kl flatlining fast)."""
    def _stub(model, loader, optimizer, device, beta, *, train, free_bits=0.0,
               feature_loss_weight=0.0):
        return recon + beta * kl + feature_loss_weight * feat, recon, kl, feat
    return _stub


def test_early_stopping_waits_for_beta_warmup(monkeypatch):
    """Regression test: patience must not exhaust during the beta ramp.

    With flat recon/kl (stubbed, mirroring the observed vgae-anticollapse curves),
    val_loss = recon + beta*kl rises every epoch purely because beta is ramping up.
    Before the fix this made the first epoch (smallest beta) look like the
    unbeatable "best" and exhausted patience almost immediately — exactly what
    happened on the vgae-gcn/gat-static-anticollapse W&B runs, which stopped at
    epoch ~25-27 with beta_warmup_epochs=100, early_stopping_patience=25.
    """
    import CLASSIFIER.model.VGAE.train as vgae_train
    monkeypatch.setattr(vgae_train, "_run_epoch", _stub_run_epoch_factory())

    model = VariationalGraphAutoencoder(IN_FEATURES, HIDDEN, LATENT, conv_type="gcn")
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    train_loader = _toy_loader(seed=0)
    val_loader = _toy_loader(seed=100)

    beta_warmup_epochs = 10
    early_stopping_patience = 2  # deliberately much shorter than the warmup
    _best, history = train_vgae_with_val(
        model, train_loader, val_loader, opt, torch.device("cpu"),
        beta=1.0, beta_warmup_epochs=beta_warmup_epochs,
        epochs=beta_warmup_epochs + early_stopping_patience + 2,
        early_stopping_patience=early_stopping_patience,
    )
    # Must run through the full warmup before early stopping is even considered.
    assert len(history["val_loss"]) >= beta_warmup_epochs
    assert history["beta"][beta_warmup_epochs - 1] == pytest.approx(1.0)
    # Post-warmup, val_loss is flat (recon/kl stubbed flat) -> patience exhausts
    # exactly `early_stopping_patience` epochs after warmup completes.
    assert len(history["val_loss"]) == beta_warmup_epochs + early_stopping_patience


def test_early_stopping_unaffected_when_no_warmup(monkeypatch):
    """beta_warmup_epochs=0 (the non-anticollapse default) keeps prior behaviour:
    patience is counted from epoch 0 since beta is constant throughout."""
    import CLASSIFIER.model.VGAE.train as vgae_train
    monkeypatch.setattr(vgae_train, "_run_epoch", _stub_run_epoch_factory())

    model = VariationalGraphAutoencoder(IN_FEATURES, HIDDEN, LATENT, conv_type="gcn")
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    train_loader = _toy_loader(seed=0)
    val_loader = _toy_loader(seed=100)

    early_stopping_patience = 3
    _best, history = train_vgae_with_val(
        model, train_loader, val_loader, opt, torch.device("cpu"),
        beta=1.0, beta_warmup_epochs=0,
        epochs=50, early_stopping_patience=early_stopping_patience,
    )
    # Flat val_loss from epoch 0 (beta constant) -> stops right after `patience`
    # epochs of no improvement, long before the 50-epoch cap.
    assert len(history["val_loss"]) == early_stopping_patience + 1
    assert all(b == pytest.approx(1.0) for b in history["beta"])
