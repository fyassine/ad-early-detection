"""Tests for CLASSIFIER.adapters — registry + the pure (data-shaping) hook logic.

These avoid touching a real GAAE checkpoint or the DELCODE matrices: they exercise
the registry, the shared metric helper, and the model-agnostic Bundle reshaping
(``truncate_to_n_visits``, GEC flattening). The torch training paths are covered
end-to-end by the experiment runner, not here.
"""
from __future__ import annotations

import numpy as np
import pytest

from CLASSIFIER.adapters import LongitudinalAdapter, binary_metrics, get_adapter
from CLASSIFIER.common.crossval import Bundle

_GAAE_HP = {"latent_dim": 6, "hidden_dim": 16, "num_heads": 2, "cond_dim": 2, "dropout": 0.3}


def _make(adapter_cls, train_config):
    return adapter_cls(
        gaae_ckpt_path="/nonexistent/model.pth",  # never loaded in these tests
        gaae_hp=_GAAE_HP, train_config=train_config,
        data_root="/nonexistent", cohorts_csv="/nonexistent/cohorts.csv",
        device="cpu", rng=np.random.default_rng(0),
    )


# ── registry ────────────────────────────────────────────────────────────────
def test_get_adapter_resolves_known_keys():
    from CLASSIFIER.adapters.gec import GECAdapter
    from CLASSIFIER.adapters.gelstm import GELSTMAdapter
    from CLASSIFIER.adapters.gep import GEPAdapter

    assert get_adapter("gelstm") is GELSTMAdapter
    assert get_adapter("GELSTM") is GELSTMAdapter   # case-insensitive
    assert get_adapter("gegru") is GELSTMAdapter     # alias (rnn_type via config)
    assert get_adapter("gec") is GECAdapter
    assert get_adapter("gep") is GEPAdapter
    assert issubclass(GELSTMAdapter, LongitudinalAdapter)
    assert issubclass(GECAdapter, LongitudinalAdapter)
    assert issubclass(GEPAdapter, LongitudinalAdapter)


def test_get_adapter_unknown_raises():
    with pytest.raises(ValueError, match="Unknown adapter"):
        get_adapter("does-not-exist")
    with pytest.raises(ValueError):
        get_adapter("")


# ── shared metric helper ─────────────────────────────────────────────────────
def test_binary_metrics_perfect_separation():
    targets = [0, 0, 1, 1]
    probs = [0.1, 0.2, 0.8, 0.9]
    m = binary_metrics(targets, probs, threshold=0.5)
    assert m["auc"] == 1.0
    assert m["sensitivity"] == 1.0
    assert m["specificity"] == 1.0
    assert m["f1"] == 1.0


def test_binary_metrics_single_class_safe():
    m = binary_metrics([1, 1, 1], [0.6, 0.7, 0.8], threshold=0.5)
    assert m["auc"] == 0.0  # AUC undefined with one class -> 0.0, no crash
    assert set(m) == {"auc", "sensitivity", "specificity", "f1"}


# ── GEC flattening / truncation (no GAAE needed) ─────────────────────────────
def _gec_items():
    rng = np.random.default_rng(1)
    items = []
    for i, ns in enumerate([3, 1, 2]):
        items.append({
            "subject_id": f"s{i}", "label": i % 2, "n_scans": ns,
            "visit_months": [12 * t for t in range(ns)],
            "zs": [rng.standard_normal(6).astype(np.float32) for _ in range(ns)],
            "dts": [0.0] + [0.1] * (ns - 1),
        })
    return items


def test_gec_records_to_X_shape_and_mask():
    adapter = _make(get_adapter("gec"),
                    {"use_time_delta": True, "append_visit_mask": True, "mlp_hidden_layers": [8]})
    adapter.max_visits = 3  # normally locked during prepare_data on the CV pool
    items = _gec_items()
    X, y = adapter._records_to_X(items, np.arange(6), adapter.max_visits)
    # feat_dim = k*mv + mv (Δt) + mv (mask) = 6*3 + 3 + 3 = 24
    assert X.shape == (3, 24)
    assert list(y) == [0.0, 1.0, 0.0]
    # mask block is the last 3 cols; subject s1 has 1 visit -> mask [1,0,0]
    mask = X[:, -3:]
    assert mask[1].tolist() == [1.0, 0.0, 0.0]
    assert mask[0].tolist() == [1.0, 1.0, 1.0]


def test_gec_truncate_drops_short_and_slices():
    adapter = _make(get_adapter("gec"), {})
    bundle = Bundle([it["label"] for it in _gec_items()],
                    [it["subject_id"] for it in _gec_items()], _gec_items())
    trunc = adapter.truncate_to_n_visits(bundle, 2)
    # s1 (1 visit) dropped; the rest capped at 2 visits.
    assert {it["subject_id"] for it in trunc.items} == {"s0", "s2"}
    for it in trunc.items:
        assert it["n_scans"] == 2
        assert len(it["zs"]) == 2 and len(it["visit_months"]) == 2


def test_gec_model_config_keys():
    adapter = _make(get_adapter("gec"), {"top_k": 4, "use_fdr": True})
    adapter.max_visits = 5
    adapter.feat_dim = adapter._feature_dim()
    cfg = adapter.model_config()
    assert cfg["model_type"] == "LongitudinalMLP"
    assert cfg["use_fdr"] is True
    assert cfg["top_k"] == 4               # k == top_k under FDR
    assert cfg["max_visits"] == 5


# ── GEP pooled-embedding flattening / truncation (no encoder needed) ─────────
def test_gep_records_to_X_mean_pools_visits():
    adapter = _make(get_adapter("gep"), {"mlp_hidden_layers": [8]})
    assert adapter.latent == 6  # gaae_hp latent_dim
    items = _gec_items()        # reuse: 6-dim zs, n_scans [3, 1, 2]
    X, y = adapter._records_to_X(items)
    assert X.shape == (3, 6)    # one pooled vector per subject, no flattening
    assert list(y) == [0.0, 1.0, 0.0]
    # subject 0's pooled vector is the mean of its 3 visit embeddings.
    expected = np.stack(items[0]["zs"]).mean(0)
    assert np.allclose(X[0], expected, atol=1e-6)


def test_gep_records_to_X_respects_n_visits_cap():
    adapter = _make(get_adapter("gep"), {})
    items = _gec_items()
    X1, _ = adapter._records_to_X(items[:1], n_visits=1)   # only first visit
    assert np.allclose(X1[0], items[0]["zs"][0], atol=1e-6)


def test_gep_truncate_drops_short_and_slices():
    adapter = _make(get_adapter("gep"), {})
    items = _gec_items()
    bundle = Bundle([it["label"] for it in items], [it["subject_id"] for it in items], items)
    trunc = adapter.truncate_to_n_visits(bundle, 2)
    assert {it["subject_id"] for it in trunc.items} == {"s0", "s2"}
    for it in trunc.items:
        assert it["n_scans"] == 2 and len(it["zs"]) == 2 and len(it["visit_months"]) == 2


def test_gep_model_config_keys():
    adapter = _make(get_adapter("gep"), {"mlp_hidden_layers": [16]})
    cfg = adapter.model_config()
    assert cfg["model_type"] == "LongitudinalMLP"
    assert cfg["encoder_arch"] == "gaae"
    assert cfg["input_dim"] == 6 and cfg["latent"] == 6


def test_gep_vgae_encoder_arch_reads_config_dims():
    adapter = _make(get_adapter("gep"),
                    {"encoder_arch": "vgae", "latent_dim": 12, "hidden_dim": 24,
                     "conv_type": "gat", "adjacency_k": 10})
    assert adapter.encoder_arch == "vgae"
    assert adapter.latent == 12            # from cfg, not gaae_hp
    assert adapter.enc_conv_type == "gat"
    assert adapter.adjacency_k == 10
    assert adapter.model_config()["encoder_arch"] == "vgae"


def test_gep_load_state_roundtrip(tmp_path):
    import pickle

    import torch
    from sklearn.preprocessing import StandardScaler

    from CLASSIFIER.adapters import read_run_threshold
    from CLASSIFIER.adapters.gec import LongitudinalMLP

    adapter = _make(get_adapter("gep"), {"mlp_hidden_layers": [8], "mlp_dropout": 0.3})
    latent = 6
    mlp = LongitudinalMLP(latent, [8], 0.3)
    scaler = StandardScaler().fit(np.random.default_rng(0).standard_normal((10, latent)))
    with open(tmp_path / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    torch.save(
        {"model_state_dict": mlp.state_dict(),
         "model_config": {"latent": latent, "input_dim": latent},
         "best_threshold": 0.37},
        tmp_path / "checkpoint_test.pth",
    )
    state = adapter.load_state(tmp_path)
    assert state["latent"] == latent
    rebuilt = adapter._model_for_state(state)
    for k, v in mlp.state_dict().items():
        assert torch.equal(v, rebuilt.state_dict()[k])
    assert read_run_threshold(tmp_path) == 0.37


def test_gep_load_state_missing_scaler_raises(tmp_path):
    import torch

    adapter = _make(get_adapter("gep"), {})
    torch.save({"model_state_dict": {}, "model_config": {"latent": 6}},
               tmp_path / "checkpoint_test.pth")
    with pytest.raises(FileNotFoundError):
        adapter.load_state(tmp_path)


# ── GELSTM truncation (no GAAE needed) ───────────────────────────────────────
def test_gelstm_truncate_slices_visit_arrays():
    adapter = _make(get_adapter("gelstm"), {})
    items = [
        {"subject_id": "a", "label": 1, "n_scans": 3,
         "graphs": [0, 1, 2], "delta_t": [0.0, 0.1, 0.2], "visit_months": [0, 12, 24]},
        {"subject_id": "b", "label": 0, "n_scans": 1,
         "graphs": [0], "delta_t": [0.0], "visit_months": [0]},
    ]
    bundle = Bundle([1, 0], ["a", "b"], items)
    trunc = adapter.truncate_to_n_visits(bundle, 2)
    assert [it["subject_id"] for it in trunc.items] == ["a"]
    it = trunc.items[0]
    assert it["n_scans"] == 2
    assert it["graphs"] == [0, 1]
    assert it["delta_t"] == [0.0, 0.1]
    assert it["visit_months"] == [0, 12]


def test_gelstm_model_config_reports_rnn_and_fdr():
    adapter = _make(get_adapter("gelstm"), {"rnn_type": "gru", "use_fdr": True, "top_k": 10})
    cfg = adapter.model_config()
    assert cfg["model_type"] == "GELSTMClassifier"
    assert cfg["rnn_type"] == "gru"
    assert cfg["use_fdr"] is True
    assert cfg["top_k"] == 10


def test_gelstm_model_config_reports_classifier_norm():
    base = _make(get_adapter("gelstm"), {})
    assert base.model_config()["classifier_norm"] == "none"   # back-compat default
    ln = _make(get_adapter("gelstm"), {"classifier_norm": "layernorm"})
    assert ln.model_config()["classifier_norm"] == "layernorm"


def test_build_classifier_head_layernorm():
    import torch.nn as nn

    from CLASSIFIER.model.GELSTM.models import build_classifier_head

    head = build_classifier_head(32, 16, 0.1, "layernorm")
    assert any(isinstance(m, nn.LayerNorm) for m in head.modules())
    plain = build_classifier_head(32, 16, 0.1, "none")
    assert not any(isinstance(m, nn.LayerNorm) for m in plain.modules())
    direct = build_classifier_head(32, 0, 0.1, "layernorm")  # no hidden -> direct Linear
    assert isinstance(direct, nn.Linear)


# ── load_state reload plumbing (no GAAE / data needed) ───────────────────────
def test_gec_load_state_roundtrip(tmp_path):
    import pickle

    import torch
    from sklearn.preprocessing import StandardScaler

    from CLASSIFIER.adapters import read_run_threshold
    from CLASSIFIER.adapters.gec import LongitudinalMLP

    adapter = _make(get_adapter("gec"),
                    {"mlp_hidden_layers": [8], "mlp_dropout": 0.4,
                     "use_time_delta": True, "append_visit_mask": True})
    # k = gaae_latent = 6, max_visits = 3 -> feat_dim = 6*3 + 3 + 3 = 24
    feat_dim = 24
    mlp = LongitudinalMLP(feat_dim, [8], 0.4)
    scaler = StandardScaler().fit(np.random.default_rng(0).standard_normal((10, feat_dim)))

    np.save(tmp_path / "dim_filter.npy", np.arange(6))
    with open(tmp_path / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    torch.save(
        {"model_state_dict": mlp.state_dict(),
         "model_config": {"input_dim": feat_dim, "max_visits": 3},
         "best_threshold": 0.42},
        tmp_path / "checkpoint_test.pth",
    )

    state = adapter.load_state(tmp_path)
    assert state["feat_dim"] == feat_dim
    assert state["max_visits"] == 3
    assert state["dim_filter"].tolist() == list(range(6))
    # The reloaded weights rebuild an identical MLP through the eval path.
    rebuilt = adapter._model_for_state(state)
    for k, v in mlp.state_dict().items():
        assert torch.equal(v, rebuilt.state_dict()[k])
    assert read_run_threshold(tmp_path) == 0.42


def test_gec_load_state_missing_artifact_raises(tmp_path):
    import torch

    adapter = _make(get_adapter("gec"), {})
    torch.save({"model_state_dict": {}, "model_config": {"max_visits": 3}},
               tmp_path / "checkpoint_test.pth")
    # scaler.pkl / dim_filter.npy absent -> loud failure, no silent fallback.
    with pytest.raises(FileNotFoundError):
        adapter.load_state(tmp_path)


def test_gelstm_load_state_reads_artifacts(tmp_path):
    import torch

    from CLASSIFIER.adapters import read_run_threshold

    adapter = _make(get_adapter("gelstm"), {})
    torch.save({"model_state_dict": {"w": torch.zeros(2)}, "best_threshold": 0.55},
               tmp_path / "checkpoint_x.pth")

    state = adapter.load_state(tmp_path)
    assert "w" in state["model_state"]
    assert state["dim_filter"] is None          # no dim_filter.npy -> non-FDR run

    np.save(tmp_path / "dim_filter.npy", np.array([1, 3, 5]))
    assert adapter.load_state(tmp_path)["dim_filter"] == [1, 3, 5]
    assert read_run_threshold(tmp_path) == 0.55
