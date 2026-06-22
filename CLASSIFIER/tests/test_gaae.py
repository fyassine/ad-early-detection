"""Unit tests for GAAE encoder, decoder, and loss functions."""
import pytest
import torch
from torch_geometric.data import Data

from CLASSIFIER.model.GAAE.models import GraphAttentionAutoencoderConditioned
from CLASSIFIER.model.GAAE.losses import (
    feature_reconstruction_loss,
    adjacency_reconstruction_loss,
    total_loss_fn,
)
from CLASSIFIER.model.GAAE.utils import calculate_dense_adjacency, create_mask


# ── Fixtures ─────────────────────────────────────────────────────────────────

N_NODES    = 6
IN_FEAT    = 8
HIDDEN     = 16
LATENT     = 4
COND_DIM   = 2
NUM_HEADS  = 1
DEVICE     = torch.device("cpu")


def _make_model() -> GraphAttentionAutoencoderConditioned:
    return GraphAttentionAutoencoderConditioned(
        in_features=IN_FEAT,
        hidden_dim=HIDDEN,
        out_features=LATENT,
        cond_dim=COND_DIM,
        num_heads=NUM_HEADS,
        dropout=0.0,
    )


def _make_graph(n_nodes: int = N_NODES) -> tuple:
    """Return (x, edge_index, edge_attr, batch_mask, cond_vec) for a single graph."""
    x = torch.randn(n_nodes, IN_FEAT)
    # Fully-connected pairs (just enough edges to exercise GAT)
    src = torch.arange(n_nodes).repeat_interleave(n_nodes)
    dst = torch.arange(n_nodes).repeat(n_nodes)
    mask = src != dst
    edge_index = torch.stack([src[mask], dst[mask]])
    edge_attr  = torch.rand(edge_index.size(1))
    batch_mask = torch.zeros(n_nodes, dtype=torch.long)
    cond_vec   = torch.randn(1, COND_DIM)
    return x, edge_index, edge_attr, batch_mask, cond_vec


# ── Encoder ───────────────────────────────────────────────────────────────────

def test_encode_output_shape():
    model = _make_model()
    model.eval()
    x, ei, ea, bm, cond = _make_graph()
    with torch.no_grad():
        z = model.encode(x, ei, ea)
    assert z.shape == (N_NODES, LATENT)


def test_encode_no_nan():
    model = _make_model()
    model.eval()
    x, ei, ea, _, _ = _make_graph()
    with torch.no_grad():
        z = model.encode(x, ei, ea)
    assert not torch.isnan(z).any()


def test_encode_with_attention_returns_weights():
    model = _make_model()
    model.eval()
    x, ei, ea, _, _ = _make_graph()
    with torch.no_grad():
        z, attn = model.encode(x, ei, ea, return_attention=True)
    assert z.shape == (N_NODES, LATENT)
    assert len(attn) == 3  # one entry per encoder GAT layer


def test_condition_latent_changes_output():
    """FiLM conditioning must not be a no-op."""
    model = _make_model()
    model.eval()
    x, ei, ea, bm, cond = _make_graph()
    with torch.no_grad():
        z = model.encode(x, ei, ea)
        z_cond = model.condition_latent(z.clone(), cond, bm)
    # With random weights, conditioning should produce a different tensor.
    assert not torch.allclose(z, z_cond)


# ── Decoder ───────────────────────────────────────────────────────────────────

def test_decode_features_shape():
    model = _make_model()
    model.eval()
    x, ei, ea, bm, cond = _make_graph()
    with torch.no_grad():
        z = model.encode(x, ei, ea)
        x_hat = model.decode_features(z, ei, ea)
    assert x_hat.shape == (N_NODES, IN_FEAT)


def test_decode_adjacency_shape():
    model = _make_model()
    model.eval()
    x, ei, ea, bm, cond = _make_graph()
    with torch.no_grad():
        z = model.encode(x, ei, ea)
        adj_hat = model.decode_adjacency(z, ei)
    # InnerProductDecoder returns one scalar per edge
    assert adj_hat.shape == (ei.size(1),)
    assert ((adj_hat >= 0) & (adj_hat <= 1)).all(), "adjacency probs must be in [0,1]"


# ── Full forward ──────────────────────────────────────────────────────────────

def test_forward_output_shapes():
    model = _make_model()
    model.eval()
    x, ei, ea, bm, cond = _make_graph()
    with torch.no_grad():
        z, x_hat, adj_hat, attn = model(x, ei, ea, cond, bm)
    assert z.shape    == (N_NODES, LATENT)
    assert x_hat.shape == (N_NODES, IN_FEAT)
    assert adj_hat.shape == (ei.size(1),)
    assert "encoder" in attn and "decoder" in attn


# ── Loss functions ────────────────────────────────────────────────────────────

def test_feature_reconstruction_loss_perfect():
    x = torch.randn(N_NODES, IN_FEAT)
    loss = feature_reconstruction_loss(x, x)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_feature_reconstruction_loss_nonzero():
    x = torch.randn(N_NODES, IN_FEAT)
    x_hat = torch.randn(N_NODES, IN_FEAT)
    loss = feature_reconstruction_loss(x, x_hat)
    assert loss.item() > 0.0


def test_adjacency_reconstruction_loss_all_correct():
    """BCE loss should be ~0 when predictions perfectly match targets."""
    N = 4
    adj = torch.eye(N)
    # predictions very close to targets (but not exactly 0/1 for BCE)
    preds = adj.clamp(0.001, 0.999)
    mask  = torch.ones(N, N, dtype=torch.bool)
    loss  = adjacency_reconstruction_loss(adj, preds, mask)
    assert loss.item() < 0.01


def test_adjacency_reconstruction_loss_mask_respected():
    """Masked-out entries should not contribute to the loss."""
    N = 4
    adj   = torch.zeros(N, N)
    preds = torch.ones(N, N) * 0.5
    mask  = torch.zeros(N, N, dtype=torch.bool)
    # Only unmask the diagonal (which is 0 in adj, 0.5 in preds)
    for i in range(N):
        mask[i, i] = True
    loss_with_mask = adjacency_reconstruction_loss(adj, preds, mask)
    mask_all = torch.ones(N, N, dtype=torch.bool)
    loss_no_mask = adjacency_reconstruction_loss(adj, preds, mask_all)
    # Both compute the same BCE (preds=0.5, target=0) but on different numbers of elements
    assert abs(loss_with_mask.item() - loss_no_mask.item()) < 1e-5


def test_total_loss_fn_weight_zero():
    """adj_loss_weight=0 should reduce total_loss to just the feature loss."""
    N = 4
    x     = torch.randn(N, IN_FEAT)
    x_hat = torch.randn(N, IN_FEAT)
    adj   = torch.zeros(N, N)
    preds = torch.ones(N, N) * 0.5
    mask  = torch.ones(N, N, dtype=torch.bool)
    total, feat, adj_loss = total_loss_fn(x, x_hat, adj, preds, mask, adj_loss_weight=0.0)
    assert total.item() == pytest.approx(feat.item(), abs=1e-6)


def test_total_loss_fn_components():
    N = 4
    x     = torch.randn(N, IN_FEAT)
    x_hat = torch.randn(N, IN_FEAT)
    adj   = torch.zeros(N, N)
    preds = torch.ones(N, N) * 0.5
    mask  = torch.ones(N, N, dtype=torch.bool)
    w = 2.0
    total, feat, adj_loss = total_loss_fn(x, x_hat, adj, preds, mask, adj_loss_weight=w)
    expected = feat + w * adj_loss
    assert total.item() == pytest.approx(expected.item(), rel=1e-5)
