"""tests/test_tfgn.py — TFGN model package correctness tests.

Covers shape contracts, gate bounds, sparsemax exact zeros, change-mask
density guards, drift anchor calibration, centrality on signed input,
recon_target='none' gradient zeroing, determinism under strict seeding,
z_only gradient backpropagation to LSTM, log_dt wiring, and patient_embeddings hook.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

# Ensure imports work from repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CLASSIFIER_ROOT = _REPO_ROOT / "CLASSIFIER"
for _p in (str(_REPO_ROOT), str(_CLASSIFIER_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from CLASSIFIER.adapters.tfgn import TFGNAdapter  # noqa: E402
from CLASSIFIER.common.crossval import Bundle  # noqa: E402
from CLASSIFIER.model.TFGN.dataset import (  # noqa: E402
    compute_change_mask,
    compute_drift_anchor,
    compute_strength_centrality,
    prepare_tfgn_item,
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
    centrality_anchor_mse,
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
        use_time_delta=True,
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

    assert "s_topo" in out
    assert out["s_topo"].shape == (10,)

    assert "gate_scores" in out
    assert out["gate_scores"].shape == (10, 1)

    assert "mu" in out
    assert out["mu"].shape == (10, 8)

    assert "mu_raw" in out
    assert out["mu_raw"].shape == (10, 8)

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
    assert out_no_recon["mu_raw"] is None
    assert out_no_recon["logvar"] is None
    assert out_no_recon["recon_logits"] is None


def test_gate_bounds():
    """2. Gate bounded in (0,1)."""
    gate = TemporalSaliencyGate(hidden_dim=8)
    h = torch.randn(100, 8) * 10
    gated_h, gate_scores = gate(h)

    assert torch.all(gate_scores > 0.0)
    assert torch.all(gate_scores < 1.0)
    # Residual scaling (1+s)h has magnitude >= |h| because (1+s) >= 1
    assert torch.all(torch.abs(gated_h) >= torch.abs(h))


def test_sparsemax_exact_zeros():
    """3. Sparsemax produces exact zeros."""
    z = torch.tensor([10.0, 0.1, -5.0, 0.0])
    s = sparsemax(z)

    assert torch.isclose(s.sum(), torch.tensor(1.0))
    assert (s == 0.0).any()
    assert (s == 0.0).sum() >= 2


def test_change_mask_density_guards():
    """4. Change-mask is in {0,1} with density in [0.01,0.50] or raises."""
    T, N = 3, 10

    # Valid mask
    torch.manual_seed(42)
    X = torch.randn(T, N, N)
    M = compute_change_mask(X, kappa=0.10)
    assert set(torch.unique(M).tolist()).issubset({0.0, 1.0})
    density = M.mean().item()
    assert 0.01 <= density <= 0.50

    # T=1 returns all zeros (not raising)
    X_single = torch.randn(1, N, N)
    M_single = compute_change_mask(X_single, kappa=0.10)
    assert torch.all(M_single == 0.0)

    # Degenerate: identical matrices across time -> density = 1.0 >= 0.50 -> raises
    X_degen = torch.ones(T, N, N)
    with pytest.raises(ValueError, match="density"):
        compute_change_mask(X_degen, kappa=0.10)


def test_drift_anchor_calibration():
    """5. mean(d̃) ≈ gate_rho within tolerance per node."""
    torch.manual_seed(42)
    T, N = 3, 200
    gate_rho = 0.15
    X = torch.randn(T, N, N)
    d_tilde = compute_drift_anchor(X, gate_rho=gate_rho, tau_d=0.05)

    assert d_tilde.shape == (N,)
    assert torch.all(d_tilde >= 0.0) and torch.all(d_tilde <= 1.0)
    assert np.isclose(d_tilde.mean().item(), gate_rho, atol=0.03)

    # T=1 returns zeros
    X_single = torch.randn(1, N, N)
    assert torch.all(compute_drift_anchor(X_single) == 0.0)


def test_centrality_non_negative():
    """6. Centrality finite and non-negative on signed FC input."""
    torch.manual_seed(42)
    N = 20
    X_baseline = torch.randn(N, N)
    X_baseline = X_baseline - X_baseline.mean()

    c = compute_strength_centrality(X_baseline)
    assert torch.all(torch.isfinite(c))
    assert torch.all(c >= 0.0)


def test_recon_target_none_zeroes_grad(tfgn_kwargs, synthetic_inputs):
    """7. recon_target='none' zeroes that gradient path."""
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
    assert out["mu_raw"] is None
    assert out["recon_logits"] is None


def test_z_only_backpropagates_to_lstm(tfgn_kwargs, synthetic_inputs):
    """Verify that under fusion='z_only', classification loss trains the LSTM parameters."""
    X, log_dt, A0_edge_index, A0_edge_attr, cond_vec = synthetic_inputs
    kwargs_z_only = tfgn_kwargs.copy()
    kwargs_z_only["fusion"] = "z_only"
    kwargs_z_only["recon_target"] = "delta_a_topk"
    model = TFGNClassifier(**kwargs_z_only)

    out = model(X, log_dt, A0_edge_index, A0_edge_attr, cond_vec)
    loss = out["logits"].sum()
    loss.backward()

    lstm_grad_norm = sum(p.grad.norm().item() for p in model.lstm.parameters() if p.grad is not None)
    assert lstm_grad_norm > 0.0, "Classification gradient failed to reach LSTM under fusion='z_only'"


def test_determinism_under_strict_seeding(tfgn_kwargs, synthetic_inputs):
    """8. Two identical-seed forward passes are bit-identical under strict=True."""
    X, log_dt, A0_edge_index, A0_edge_attr, cond_vec = synthetic_inputs

    def run_once():
        set_seed(42, strict=True)
        model = TFGNClassifier(**tfgn_kwargs)
        model.train()
        return model(X, log_dt, A0_edge_index, A0_edge_attr, cond_vec)

    out1 = run_once()
    out2 = run_once()

    assert torch.equal(out1["logits"], out2["logits"])
    assert torch.equal(out1["gate_scores"], out2["gate_scores"])
    assert torch.equal(out1["mu_raw"], out2["mu_raw"])


def _make_synthetic_item(subject_id="sub-01", label=1, n_visits=3, n_rois=10):
    graphs = []
    for _ in range(n_visits):
        x = torch.randn(n_rois, n_rois)
        adj = torch.ones(n_rois, n_rois)
        ei, ew = dense_to_sparse(adj)
        graphs.append(Data(x=x, edge_index=ei, edge_attr=ew))
    return {
        "subject_id": subject_id,
        "label": label,
        "visit_months": [0, 12, 24][:n_visits],
        "delta_t": [0.0, 12.0 / 108.0, 12.0 / 108.0][:n_visits],
        "graphs": graphs,
        "sex": 0,
        "age": 0.7,
        "n_scans": n_visits,
        "cohort": "delcode",
    }


def test_patient_embeddings_hook():
    """Verify adapter patient_embeddings hook returns (N_subjects, D) array."""
    adapter = TFGNAdapter(
        gaae_ckpt_path="",
        gaae_hp={},
        train_config={"n_rois": 10, "lstm_hidden": 8, "gvae_latent": 8},
        data_root="",
        cohorts_csv="",
        device="cpu",
        rng=None,
    )
    items = [_make_synthetic_item(f"sub-{i}", i % 2, n_rois=10) for i in range(4)]
    bundle = Bundle([it["label"] for it in items], [it["subject_id"] for it in items], items)
    model = adapter._build_model()
    state = {
        "model_state": model.state_dict(),
        "log_dt_scaler_mean": [0.0],
        "log_dt_scaler_scale": [1.0],
        "cent_mean": 0.0,
        "cent_std": 1.0,
    }
    embs = adapter.patient_embeddings(state, bundle, device="cpu")
    assert isinstance(embs, np.ndarray)
    assert embs.shape == (4, 8)
