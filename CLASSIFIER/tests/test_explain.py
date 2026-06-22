"""Tests for the EXPLAIN layer — model-agnostic helpers + the explain-adapter registry.

These avoid touching a real GAAE checkpoint or the DELCODE matrices: they exercise the
atlas loading, the 2-D embedding, region-importance summaries, the flat-importance
unpacking, and the ``get_explain_adapter`` registry. The torch attribution paths are
covered end-to-end by the experiment runner, not here.
"""
from __future__ import annotations

import numpy as np
import pytest

import matplotlib

matplotlib.use("Agg")  # headless

from CLASSIFIER.common import explain as ce
from CLASSIFIER.adapters.explain import (
    GAAEExplainAdapter,
    GECExplainAdapter,
    GELSTMExplainAdapter,
    get_explain_adapter,
    resolve_source_run,
)


# ── atlas ─────────────────────────────────────────────────────────────────────
def test_load_schaefer_atlas_is_200_indexed():
    atlas = ce.load_schaefer_atlas()
    assert len(atlas) == 200
    assert [r["index"] for r in atlas] == list(range(200))
    coords = ce.atlas_coords(atlas)
    assert coords.shape == (200, 3)
    nets = ce.atlas_networks(atlas)
    assert len(nets) == 200
    assert set(nets) <= set(ce.NETWORK_ORDER)


# ── embed_2d ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("method", ["pca", "tsne", "umap"])
def test_embed_2d_shapes_and_determinism(method):
    X = np.random.RandomState(0).randn(24, 8)
    a = ce.embed_2d(X, method, seed=7)
    b = ce.embed_2d(X, method, seed=7)
    assert a.shape == (24, 2)
    np.testing.assert_allclose(a, b)  # seeded => deterministic


def test_embed_2d_rejects_tiny_input():
    with pytest.raises(ValueError):
        ce.embed_2d(np.zeros((2, 4)), "pca")


# ── region-importance summary ────────────────────────────────────────────────
def test_network_importance_summary_keys():
    atlas = ce.load_schaefer_atlas()
    vals = np.random.RandomState(1).rand(200)
    summary = ce.network_importance_summary(vals, atlas)
    assert set(summary) <= set(ce.NETWORK_ORDER)
    assert all(0.0 <= v <= 1.0 for v in summary.values())


# ── GEC flat-importance unpacking ─────────────────────────────────────────────
def test_unpack_flat_importance_layout():
    from CLASSIFIER.model.GEC.explain import unpack_flat_importance

    k, max_visits = 4, 3
    # z block (12) + dt block (3) + mask block (3) = 18
    imp = np.arange(18, dtype=float)
    out = unpack_flat_importance(
        imp, k=k, max_visits=max_visits, use_time_delta=True, append_visit_mask=True
    )
    assert out["per_visit"].shape == (max_visits,)
    assert out["per_dim"].shape == (k,)
    assert out["dt"].shape == (max_visits,)
    assert float(out["per_visit"].max()) == pytest.approx(1.0)


# ── GAAE reconstruction fidelity ──────────────────────────────────────────────
def test_reconstruction_quality_perfect():
    from CLASSIFIER.model.GAAE.explain import reconstruction_quality

    rng = np.random.RandomState(0)
    x = rng.randn(200, 200)
    q = reconstruction_quality(x, x.copy())
    assert q["mse"] == pytest.approx(0.0)
    assert q["rmse"] == pytest.approx(0.0)
    assert q["mae"] == pytest.approx(0.0)
    assert q["pearson_r"] == pytest.approx(1.0)
    assert q["r2"] == pytest.approx(1.0)
    assert q["nrmse"] == pytest.approx(0.0)
    assert q["quality"] == "excellent"
    assert q["residual"].shape == x.shape
    assert np.allclose(q["residual"], 0.0)


def test_reconstruction_quality_nrmse_is_sqrt_one_minus_r2():
    from CLASSIFIER.model.GAAE.explain import reconstruction_quality

    rng = np.random.RandomState(1)
    x = rng.randn(64, 64)
    x_rec = x + rng.randn(64, 64) * 0.3
    q = reconstruction_quality(x, x_rec)
    # NRMSE == sqrt(1 - R^2) by construction (both centre on the input).
    assert q["nrmse"] == pytest.approx((1.0 - q["r2"]) ** 0.5, rel=1e-6)


@pytest.mark.parametrize("noise, band", [(0.02, "excellent"), (1.5, "poor")])
def test_reconstruction_quality_bands(noise, band):
    from CLASSIFIER.model.GAAE.explain import reconstruction_quality

    rng = np.random.RandomState(2)
    base = rng.randn(100, 100)
    x = (base + base.T) / 2
    q = reconstruction_quality(x, x + rng.randn(100, 100) * noise)
    assert q["quality"] == band


def test_reconstruction_quality_shape_mismatch_raises():
    from CLASSIFIER.model.GAAE.explain import reconstruction_quality

    with pytest.raises(ValueError, match="shape mismatch"):
        reconstruction_quality(np.zeros((3, 3)), np.zeros((3, 4)))


# ── registry ──────────────────────────────────────────────────────────────────
def test_get_explain_adapter_resolves_known_keys():
    assert get_explain_adapter("gaae") is GAAEExplainAdapter
    assert get_explain_adapter("GEC") is GECExplainAdapter          # case-insensitive
    assert get_explain_adapter("gelstm") is GELSTMExplainAdapter
    assert get_explain_adapter("gegru") is GELSTMExplainAdapter      # GRU alias


def test_get_explain_adapter_unknown_raises():
    with pytest.raises(ValueError):
        get_explain_adapter("does-not-exist")


def test_capabilities_are_disjoint_by_design():
    # GAAE has no class probability; classifiers do.
    assert "probability" not in GAAEExplainAdapter.capabilities
    assert "probability" in GECExplainAdapter.capabilities
    assert "probability" in GELSTMExplainAdapter.capabilities
    # temporal-only capabilities belong to the recurrent adapter.
    assert "hidden_state" in GELSTMExplainAdapter.capabilities
    assert "hidden_state" not in GECExplainAdapter.capabilities


def test_resolve_source_run_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_source_run("no-such-experiment", classifier_root=tmp_path)


def test_resolve_source_run_latest_txt(tmp_path):
    exp = tmp_path / "outputs" / "exp-x"
    (exp / "runs" / "run-1").mkdir(parents=True)
    (exp / "latest.txt").write_text("run-1")
    out = resolve_source_run("exp-x", classifier_root=tmp_path)
    assert out == exp / "runs" / "run-1"
