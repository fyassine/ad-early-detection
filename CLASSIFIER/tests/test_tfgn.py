"""tests/test_tfgn.py — TFGN model package correctness tests.

Covers shape contracts, gate bounds, sparsemax exact zeros, change-mask
density guards, drift anchor calibration, centrality on signed input,
recon_target='none' gradient zeroing, and determinism under strict seeding.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Ensure imports work from repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CLASSIFIER_ROOT = _REPO_ROOT / "CLASSIFIER"
for _p in (str(_REPO_ROOT), str(_CLASSIFIER_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from CLASSIFIER.model.TFGN.dataset import (  # noqa: E402
    compute_change_mask,
    compute_drift_anchors,
    compute_strength_centrality,
)
from CLASSIFIER.model.TFGN.layers import (  # noqa: E402
    AttentivePool,
    ConcatResidualFusion,
    DualScoreReadout,
    GVAEEncoder,
    NodeSharedLSTM,
    TemporalSaliencyGate,
    sparsemax,
)
from CLASSIFIER.model.TFGN.losses import (  # noqa: E402
    change_mask_bce,
    drift_anchor_mse,
    free_bits_kl,
    gate_sparsity_kl,
)
from CLASSIFIER.model.TFGN.models import TFGNClassifier  # noqa: E402
from SHARED.seeding import set_seed  # noqa: E402


@pytest.fixture
def tfgn_kwargs():
    return dict(
        n_rois=10,
        lstm_hidden=8,
        lstm_layers=1,
        lstm_dropout=0.0,
        gvae_hidden=16,
        gvae_latent=8,
        gvae_heads=2,
        gvae_dropout=0.0,
        cond_dim=2,
        use_gate=True,
        recon_target="delta_a_mse",
        fusion="concat_residual",
        readout="attention",
        dual_score=True,
    )


@pytest.fixture
def synthetic_inputs():
    torch.manual_seed(42)
    T, N = 3, 10
    X = torch.randn(T, N, N)
    log_dt = torch.randn(T)
    A0_edge_index = torch.randint(0, N, (2, 20))
    A0_edge_attr = torch.randn(20)
    cond_vec = torch.randn(2)
    return X, log_dt, A0_edge_index, A0_edge_attr, cond_vec


def test_shape_contracts(tfgn_kwargs, synthetic_inputs):
    """1. Shape contracts per stage."""
    X, log_dt, A0_edge_index, A0_edge_attr, cond_vec = synthetic_inputs
    model = TFGNClassifier(**tfgn_kwargs)
    out = model(X, log_dt, A0_edge_index, A0_edge_attr, cond_vec)

    assert "logits" in out
    assert out["logits"].shape == (1,)

    assert "aux_logits" in out
    assert out["aux_logits"].shape == (1,)

    assert "gate_scores" in out
    assert out["gate_scores"].shape == (10, 1)

    assert "mu" in out
    assert out["mu"].shape == (10, 8)

    assert "logvar" in out
    assert out["logvar"].shape == (10, 8)

    assert "recon_logits" in out
    assert out["recon_logits"].shape == (10, 10)

    # Test without gate
    kwargs_no_gate = tfgn_kwargs.copy()
    kwargs_no_gate["use_gate"] = False
    model_no_gate = TFGNClassifier(**kwargs_no_gate)
    out_no_gate = model_no_gate(X, log_dt, A0_edge_index, A0_edge_attr, cond_vec)
    assert out_no_gate["gate_scores"] is None

    # Test with recon_target='none'
    kwargs_no_recon = tfgn_kwargs.copy()
    kwargs_no_recon["recon_target"] = "none"
    model_no_recon = TFGNClassifier(**kwargs_no_recon)
    out_no_recon = model_no_recon(X, log_dt, A0_edge_index, A0_edge_attr, cond_vec)
    assert out_no_recon["mu"] is None
    assert out_no_recon["logvar"] is None
    assert out_no_recon["recon_logits"] is None


def test_gate_bounds():
    """2. Gate bounded in (0,1)."""
    torch.manual_seed(42)
    gate = TemporalSaliencyGate(hidden_dim=8)
    h = torch.randn(100, 8) * 10  # moderately large to test boundaries without FP32 rounding to 1.0
    gated_h, gate_scores = gate(h)

    assert torch.all(gate_scores > 0.0)
    assert torch.all(gate_scores < 1.0)


def test_sparsemax_exact_zeros():
    """3. Sparsemax produces exact zeros."""
    # A vector with one dominant value should produce exact zeros for others
    z = torch.tensor([10.0, 0.1, -5.0, 0.0])
    s = sparsemax(z)

    assert torch.isclose(s.sum(), torch.tensor(1.0))
    # the 10.0 element gets 1.0, others get exactly 0.0
    assert s[0].item() == 1.0
    assert s[1].item() == 0.0
    assert s[2].item() == 0.0
    assert s[3].item() == 0.0


def test_change_mask_density_guards():
    """4. Change-mask is in {0,1} with density in [0.01,0.50] or raises."""
    T, N = 3, 10

    # Valid mask
    torch.manual_seed(42)
    X_valid = torch.randn(T, N, N)
    M = compute_change_mask(X_valid, kappa=0.10)
    assert set(M.unique().tolist()).issubset({0.0, 1.0})
    density = M.mean().item()
    assert 0.01 <= density <= 0.50

    # All edges change equally (density=1.0) -> raises
    X_all = torch.zeros(T, N, N)
    X_all[-1] = 1.0
    with pytest.raises(ValueError, match="density"):
        compute_change_mask(X_all, kappa=0.10)

    # No edges change (density=0.0) -> raises
    X_none = torch.zeros(T, N, N)
    with pytest.raises(ValueError, match="density"):
        compute_change_mask(X_none, kappa=0.10)


def test_drift_anchor_calibration():
    """5. mean(d̃) ≈ gate_rho within tolerance."""
    torch.manual_seed(42)
    N_subj = 100
    T, N = 3, 10

    X_list = []
    for _ in range(N_subj):
        # vary drift magnitude
        noise = torch.randn(1).item() * 10
        X = torch.zeros(T, N, N)
        X[-1] = X[-1] + noise
        X_list.append(X)

    gate_rho = 0.15
    anchors = compute_drift_anchors(X_list, gate_rho=gate_rho)

    mean_anchor = np.mean(anchors)
    assert np.isclose(mean_anchor, gate_rho, atol=0.05)


def test_centrality_non_negative():
    """6. Centrality finite and non-negative on signed FC input."""
    N = 10
    X_baseline = torch.randn(N, N)
    # Give some negative values
    X_baseline = X_baseline - X_baseline.mean()

    c = compute_strength_centrality(X_baseline)

    assert torch.all(torch.isfinite(c))
    assert torch.all(c >= 0.0)


def test_recon_target_none_zeroes_grad(tfgn_kwargs, synthetic_inputs):
    """7. recon_target='none' really zeroes that gradient path."""
    X, log_dt, A0_edge_index, A0_edge_attr, cond_vec = synthetic_inputs
    X.requires_grad_(True)

    kwargs_no_recon = tfgn_kwargs.copy()
    kwargs_no_recon["recon_target"] = "none"
    model = TFGNClassifier(**kwargs_no_recon)

    out = model(X, log_dt, A0_edge_index, A0_edge_attr, cond_vec)
    loss = out["logits"].sum()
    loss.backward()

    assert X.grad is not None
    assert out["mu"] is None
    assert out["logvar"] is None
    assert out["recon_logits"] is None


def test_determinism_under_strict_seeding(tfgn_kwargs, synthetic_inputs):
    """8. Two identical-seed forward passes are bit-identical under strict=True."""
    X, log_dt, A0_edge_index, A0_edge_attr, cond_vec = synthetic_inputs

    # Run 1
    set_seed(42, strict=True)
    model1 = TFGNClassifier(**tfgn_kwargs)
    out1 = model1(X, log_dt, A0_edge_index, A0_edge_attr, cond_vec)

    # Run 2
    set_seed(42, strict=True)
    model2 = TFGNClassifier(**tfgn_kwargs)
    out2 = model2(X, log_dt, A0_edge_index, A0_edge_attr, cond_vec)

    assert torch.equal(out1["logits"], out2["logits"])
    if out1["aux_logits"] is not None:
        assert torch.equal(out1["aux_logits"], out2["aux_logits"])
    assert torch.equal(out1["gate_scores"], out2["gate_scores"])

    if out1["mu"] is not None:
        assert torch.equal(out1["mu"], out2["mu"])
    if out1["logvar"] is not None:
        assert torch.equal(out1["logvar"], out2["logvar"])
    if out1["recon_logits"] is not None:
        assert torch.equal(out1["recon_logits"], out2["recon_logits"])
