"""Tests for fail-loud error paths and edge cases across CLASSIFIER utilities."""
import tempfile
import warnings

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from CLASSIFIER.common.utils import compute_class_cost_weights, compute_class_weights
from CLASSIFIER.model.GEC.models import GraphEncoderClassifier
from CLASSIFIER.model.GELSTM.utils import encode_batch_sequences

# ── encode_batch_sequences ────────────────────────────────────────────────────

class _MinimalEncoder(torch.nn.Module):
    """Minimal stub that satisfies encode_visit()."""
    def encode_visit(self, x, edge_index, edge_attr=None, pool="mean"):
        return torch.zeros(2)  # 2-dim fake embedding


def test_encode_batch_sequences_empty_batch_raises():
    """Empty batch must raise ValueError immediately (not crash on max([]))."""
    enc = _MinimalEncoder()
    with pytest.raises(ValueError, match="empty batch"):
        encode_batch_sequences([], enc, device=torch.device("cpu"))


def test_encode_batch_sequences_single_item_ok():
    """Single-item batch should not raise."""
    enc = _MinimalEncoder()
    g = Data(
        x=torch.zeros(1, 1),
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        edge_attr=None,
    )
    batch = [{"subject_id": "s1", "graphs": [g], "delta_t": [0.0],
              "visit_months": [0.0], "label": 0.0}]
    packed, labels, lengths = encode_batch_sequences(
        batch, enc, device=torch.device("cpu"), use_time_delta=False
    )
    assert labels.shape == (1,)


# ── compute_class_weights ─────────────────────────────────────────────────────

def test_compute_class_weights_normal():
    labels = [0, 0, 0, 1, 1]
    w = compute_class_weights(labels)
    assert w.item() == pytest.approx(3.0 / 2.0, rel=1e-5)


def test_compute_class_weights_all_positive_warns():
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        w = compute_class_weights([1, 1, 1])
    assert any("degenerate" in str(w.message).lower() for w in ws)
    assert w.item() == pytest.approx(1.0)


def test_compute_class_weights_all_negative_warns():
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        w = compute_class_weights([0, 0, 0])
    assert any("degenerate" in str(w.message).lower() for w in ws)
    assert w.item() == pytest.approx(1.0)


# ── compute_class_cost_weights ────────────────────────────────────────────────

def test_compute_class_cost_weights_normal():
    labels = [0, 0, 0, 1, 1]
    w = compute_class_cost_weights(labels)
    assert w.shape == (2,)
    assert (w > 0).all()


def test_compute_class_cost_weights_empty_raises():
    with pytest.raises(ValueError, match="empty label set"):
        compute_class_cost_weights([])


def test_compute_class_cost_weights_single_class_warns():
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        w = compute_class_cost_weights([1, 1, 1])
    assert any("degenerate" in str(w.message).lower() for w in ws)
    assert torch.allclose(w, torch.tensor([1.0, 1.0]))


# ── load_frozen_encoder_from_gaae ─────────────────────────────────────────────

def test_load_frozen_encoder_mismatched_dims_raises(tmp_path):
    """A checkpoint whose encoder dims don't match the GEC model must raise ValueError."""
    from CLASSIFIER.common.utils import load_frozen_encoder_from_gaae

    # Build a GEC model with latent_dim=4
    gec = GraphEncoderClassifier(
        in_features=8, hidden_dim=16, latent_dim=4, cond_dim=2, num_heads=1
    )

    # Create a fake GAAE checkpoint where encoder_gat3 has wrong output dim (8, not 4)
    fake_sd = {}
    for k, v in gec.state_dict().items():
        # Flip the latent-dim (first) for the final GAT layer weight
        if "encoder_gat3" in k and v.dim() >= 1:
            wrong_shape = list(v.shape)
            wrong_shape[0] *= 2  # double the output dim → mismatch
            fake_sd[k] = torch.zeros(wrong_shape)
        else:
            fake_sd[k] = v.clone()

    ckpt_path = tmp_path / "fake_gaae.pth"
    torch.save({"model_state_dict": fake_sd}, ckpt_path)

    with pytest.raises(ValueError, match="incompatible"):
        load_frozen_encoder_from_gaae(gec, ckpt_path)


def test_load_frozen_encoder_no_matching_keys_raises(tmp_path):
    """A checkpoint with no encoder keys at all must raise ValueError."""
    from CLASSIFIER.common.utils import load_frozen_encoder_from_gaae

    gec = GraphEncoderClassifier(
        in_features=8, hidden_dim=16, latent_dim=4, cond_dim=2, num_heads=1
    )
    # Checkpoint with only classifier keys (no encoder_gatX / film_ keys)
    fake_sd = {"classifier.0.weight": torch.zeros(64, 4)}
    ckpt_path = tmp_path / "no_encoder_keys.pth"
    torch.save({"model_state_dict": fake_sd}, ckpt_path)

    with pytest.raises(ValueError, match="No encoder keys"):
        load_frozen_encoder_from_gaae(gec, ckpt_path)
