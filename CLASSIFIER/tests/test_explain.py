"""Tests for the EXPLAIN layer — model-agnostic helpers + the explain-adapter registry.

These avoid touching a real GAAE checkpoint or the DELCODE matrices: they exercise the
atlas loading, the 2-D embedding, region-importance summaries, the flat-importance
unpacking, and the ``get_explain_adapter`` registry. The torch attribution paths are
covered end-to-end by the experiment runner, not here.
"""
from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # headless

from CLASSIFIER.adapters.explain import (
    ExplainAdapter,
    GAAEExplainAdapter,
    GECExplainAdapter,
    GELSTMExplainAdapter,
    VGAEExplainAdapter,
    get_explain_adapter,
    resolve_source_run,
)
from CLASSIFIER.common import explain as ce


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


def test_gaae_capabilities_include_decoder_dependent_extras():
    assert "disease_axis_steering" in GAAEExplainAdapter.capabilities
    assert "sorted_reconstruction" in GAAEExplainAdapter.capabilities
    # VGAE's adjacency decoder is structurally different — not ported (yet).
    assert "disease_axis_steering" not in VGAEExplainAdapter.capabilities
    assert "sorted_reconstruction" not in VGAEExplainAdapter.capabilities


# ── common/explain.py: latent-space separability ─────────────────────────────
def test_latent_dim_separability_ranks_discriminative_dim_first():
    rng = np.random.RandomState(0)
    n = 60
    y = np.array([0] * 30 + [1] * 30)
    X = rng.randn(n, 5)
    X[y == 1, 0] += 5.0  # dim 0 is by far the most discriminative
    out = ce.latent_dim_separability(X, y)
    assert out["fdr"].shape == (5,)
    assert out["ranked_dims"][0] == 0
    assert out["silhouette"] > 0


# ── common/explain.py: 3-D UMAP ────────────────────────────────────────────────
def test_embed_3d_shape_and_determinism():
    pytest.importorskip("umap")
    X = np.random.RandomState(0).randn(24, 8)
    a = ce.embed_3d(X, seed=7)
    b = ce.embed_3d(X, seed=7)
    assert a.shape == (24, 3)
    np.testing.assert_allclose(a, b)


def test_embed_3d_rejects_tiny_input():
    with pytest.raises(ValueError):
        ce.embed_3d(np.zeros((2, 4)))


# ── common/explain.py: disease-axis projection ────────────────────────────────
def test_disease_axis_projection_shapes_and_unit_norm():
    rng = np.random.RandomState(0)
    n = 50
    y = np.array([0] * 25 + [1] * 25)
    X = rng.randn(n, 6)
    X[y == 1] += 2.0
    proj = ce.disease_axis_projection(X, y, seed=1)
    assert proj["w_hat"].shape == (6,)
    assert proj["scores"].shape == (n,)
    assert proj["residual_pc"].shape == (n, 2)
    assert np.linalg.norm(proj["w_hat"]) == pytest.approx(1.0, rel=1e-5)


# ── common/explain.py: plotting smoke tests (headless matplotlib.use("Agg")) ──
def test_plot_latent_dim_distributions_smoke():
    rng = np.random.RandomState(0)
    n = 40
    y = np.array([0] * 20 + [1] * 20)
    X = rng.randn(n, 5)
    sep = ce.latent_dim_separability(X, y)
    fig = ce.plot_latent_dim_distributions(X, y, sep["fdr"], top_n=4)
    assert fig is not None


def test_plot_disease_axis_smoke():
    rng = np.random.RandomState(0)
    n = 30
    y = np.array([0] * 15 + [1] * 15)
    X = rng.randn(n, 4)
    proj = ce.disease_axis_projection(X, y)
    fig = ce.plot_disease_axis(proj, y)
    assert fig is not None


def test_plot_latent_space_3d_smoke():
    pytest.importorskip("umap")
    pytest.importorskip("plotly")
    X = np.random.RandomState(0).randn(20, 8)
    y = np.array([0] * 10 + [1] * 10)
    emb3d = ce.embed_3d(X, seed=3)
    fig = ce.plot_latent_space_3d(emb3d, y)
    assert fig is not None


def test_plot_disease_axis_3d_smoke():
    pytest.importorskip("plotly")
    rng = np.random.RandomState(0)
    n = 30
    y = np.array([0] * 15 + [1] * 15)
    X = rng.randn(n, 4)
    proj = ce.disease_axis_projection(X, y)
    fig = ce.plot_disease_axis_3d(proj, y)
    assert fig is not None


# ── model/GAAE/explain.py: decoder-dependent helpers (identity stub model) ────
class _IdentityAutoencoder:
    """Encode/decode as identity — isolates steer_along_axis's arithmetic from a
    real GAAE checkpoint."""

    def eval(self):
        return self

    def encode(self, x, edge_index, edge_attr):
        return x

    def decode_features(self, z, edge_index, edge_attr):
        return z


def test_steer_along_axis_zero_scale_reproduces_baseline():
    import torch
    from torch_geometric.data import Data

    from CLASSIFIER.model.GAAE.explain import steer_along_axis

    model = _IdentityAutoencoder()
    data = Data(x=torch.randn(5, 4), edge_index=torch.zeros((2, 0), dtype=torch.long))
    w_hat = np.array([1.0, 0.0, 0.0, 0.0])
    out = steer_along_axis(model, data, w_hat, sigma=1.0,
                           scales=np.array([-1.0, 0.0, 1.0]), device="cpu")
    assert out["baseline_fc"].shape == (5, 4)
    assert len(out["steered_fcs"]) == 3
    np.testing.assert_allclose(out["steered_fcs"][1], out["baseline_fc"], atol=1e-6)
    # +1 sigma along dim 0 shifts every node's first feature by exactly +1.
    np.testing.assert_allclose(
        out["steered_fcs"][2][:, 0] - out["baseline_fc"][:, 0], 1.0, atol=1e-6
    )


def test_reconstruct_sorted_by_score_aligns_outputs():
    import torch
    from torch_geometric.data import Data

    from CLASSIFIER.model.GAAE.explain import reconstruct_sorted_by_score

    model = _IdentityAutoencoder()
    subjects = [
        {"data": Data(x=torch.full((2, 3), float(i)), edge_index=torch.zeros((2, 0), dtype=torch.long)),
         "score": float(i), "label": i % 2}
        for i in range(3)
    ]
    out = reconstruct_sorted_by_score(model, subjects, device="cpu")
    assert len(out["gt"]) == len(out["recon"]) == 3
    assert out["scores"] == [0.0, 1.0, 2.0]
    assert out["labels"] == [0, 1, 0]
    for gt, recon in zip(out["gt"], out["recon"], strict=True):
        np.testing.assert_allclose(gt, recon)


# ── adapters/explain.py: generic extras available on every adapter ───────────
class _FakeAdapter(ExplainAdapter):
    """Minimal adapter exercising only ExplainAdapter.extra's generic dispatch —
    avoids needing a real GAAE checkpoint to test logreg_probe/latent_separability/
    disease_axis, which only depend on latent_embeddings()."""

    capabilities: set = set()

    def __init__(self, X, y, sids):
        self._X, self._y, self._sids = X, y, sids

    def latent_embeddings(self, bundle):
        return self._X, self._y, self._sids


def _make_fake_adapter():
    rng = np.random.RandomState(0)
    n = 40
    y = np.array([0] * 20 + [1] * 20)
    X = rng.randn(n, 6)
    X[y == 1] += 3.0
    sids = [f"s{i}" for i in range(n)]
    return _FakeAdapter(X, y, sids)


def test_extra_latent_separability_generic():
    adapter = _make_fake_adapter()
    out = adapter.extra("latent_separability", {"bundle": object()})
    assert out["fdr"].shape == (6,)
    assert out["X"].shape == (40, 6)


def test_extra_logreg_probe_generic():
    adapter = _make_fake_adapter()
    out = adapter.extra("logreg_probe", {"bundle": object()})
    assert len(out["result"].fold_aucs) > 0


def test_extra_disease_axis_generic():
    adapter = _make_fake_adapter()
    out = adapter.extra("disease_axis", {"bundle": object()})
    assert out["w_hat"].shape == (6,)
    assert out["residual_pc"].shape == (40, 2)


def test_extra_requires_bundle_in_ctx():
    adapter = _make_fake_adapter()
    with pytest.raises(ValueError, match="ctx"):
        adapter.extra("disease_axis", {})


def test_extra_unknown_name_raises():
    adapter = _make_fake_adapter()
    with pytest.raises(ValueError, match="no extra"):
        adapter.extra("not-a-real-extra", {"bundle": object()})
