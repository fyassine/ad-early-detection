"""Unit tests for GEC graph encoder-classifier models."""
import pytest
import torch

from CLASSIFIER.model.GEC.models import (
    GraphEncoderClassifier,
    GraphEncoderClassifierAttention,
)

# ── Shared helpers ────────────────────────────────────────────────────────────

IN_FEAT  = 8
HIDDEN   = 16
LATENT   = 4
COND_DIM = 2
N_GRAPHS = 3
N_NODES  = 5  # nodes per graph


def _make_batch(n_nodes: int = N_NODES, n_graphs: int = N_GRAPHS):
    """Return (x, edge_index, cond_vec, batch_mask) for a toy batch."""
    total = n_nodes * n_graphs
    x = torch.randn(total, IN_FEAT)
    # No edges — still a valid graph; GAT handles isolated nodes
    edge_index = torch.zeros(2, 0, dtype=torch.long)
    cond_vec   = torch.randn(n_graphs, COND_DIM)
    batch_mask = torch.arange(n_graphs).repeat_interleave(n_nodes)
    return x, edge_index, cond_vec, batch_mask


def _make_model(cls):
    return cls(
        in_features=IN_FEAT,
        hidden_dim=HIDDEN,
        latent_dim=LATENT,
        cond_dim=COND_DIM,
        num_heads=1,
        dropout=0.0,
    )


# ── GraphEncoderClassifier ────────────────────────────────────────────────────

@pytest.mark.parametrize("cls", [GraphEncoderClassifier, GraphEncoderClassifierAttention])
def test_forward_output_shapes(cls):
    model = _make_model(cls)
    model.eval()
    x, ei, cond, bm = _make_batch()
    with torch.no_grad():
        logits, emb = model(x, ei, cond, bm)
    assert logits.shape == (N_GRAPHS,), f"logits shape {logits.shape}"
    assert emb.shape    == (N_GRAPHS, LATENT), f"embedding shape {emb.shape}"


@pytest.mark.parametrize("cls", [GraphEncoderClassifier, GraphEncoderClassifierAttention])
def test_forward_no_nan(cls):
    model = _make_model(cls)
    model.eval()
    x, ei, cond, bm = _make_batch()
    with torch.no_grad():
        logits, emb = model(x, ei, cond, bm)
    assert not torch.isnan(logits).any()
    assert not torch.isnan(emb).any()


@pytest.mark.parametrize("cls", [GraphEncoderClassifier, GraphEncoderClassifierAttention])
def test_freeze_encoder_freezes_encoder_params(cls):
    model = _make_model(cls)
    model.freeze_encoder()
    for name, p in model.named_parameters():
        enc_names = ("encoder_gat", "encoder_bn", "film_gamma", "film_beta")
        if any(name.startswith(n) for n in enc_names):
            assert not p.requires_grad, f"{name} should be frozen"
    # Classifier weights must remain trainable
    for name, p in model.named_parameters():
        if name.startswith("classifier"):
            assert p.requires_grad, f"{name} should still be trainable"


@pytest.mark.parametrize("cls", [GraphEncoderClassifier, GraphEncoderClassifierAttention])
def test_unfreeze_encoder_restores_all_params(cls):
    model = _make_model(cls)
    model.freeze_encoder()
    model.unfreeze_encoder()
    for name, p in model.named_parameters():
        assert p.requires_grad, f"{name} should be trainable after unfreeze"


@pytest.mark.parametrize("cls", [GraphEncoderClassifier, GraphEncoderClassifierAttention])
def test_get_trainable_params_after_freeze(cls):
    model = _make_model(cls)
    model.freeze_encoder()
    trainable = model.get_trainable_params()
    # Only classifier head + attention_pool (if present) should be in the list
    total_params   = sum(p.numel() for p in model.parameters())
    frozen_params  = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_numel = sum(p.numel() for p in trainable)
    assert trainable_numel == total_params - frozen_params


@pytest.mark.parametrize("cls", [GraphEncoderClassifier, GraphEncoderClassifierAttention])
def test_encode_output_shape(cls):
    model = _make_model(cls)
    model.eval()
    x, ei, _, _ = _make_batch()
    with torch.no_grad():
        z = model.encode(x, ei)
    assert z.shape == (N_NODES * N_GRAPHS, LATENT)
