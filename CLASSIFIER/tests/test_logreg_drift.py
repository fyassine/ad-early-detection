import numpy as np
import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

from CLASSIFIER.adapters.logreg_drift import LogRegDriftAdapter
from CLASSIFIER.common.crossval import Bundle


def _make_synthetic_item(subject_id, label, n_visits=3, n_rois=15):
    graphs = []
    for _ in range(n_visits):
        x = torch.randn(n_rois, n_rois)
        adj = torch.ones(n_rois, n_rois)
        ei, ew = dense_to_sparse(adj)
        graphs.append(Data(x=x, edge_index=ei, edge_attr=ew))
    return {
        "subject_id": subject_id,
        "label": label,
        "visit_months": list(range(0, n_visits * 12, 12)),
        "delta_t": [0.0] + [12.0 / 108.0] * (n_visits - 1),
        "graphs": graphs,
        "sex": 0,
        "age": 0.7,
        "n_scans": n_visits,
    }


def _make_adapter():
    return LogRegDriftAdapter(
        gaae_ckpt_path="",
        gaae_hp={},
        train_config={"min_visits": 2},
        data_root="",
        cohorts_csv="",
        device="cpu",
        rng=None,
    )

def test_extract_features_shape():
    adapter = _make_adapter()
    # 40 items and 15 ROIs (105 features) ensures PCA components = 32
    items = [_make_synthetic_item(f"sub_{i}", i % 2, n_rois=15) for i in range(40)]

    X, pca = adapter._extract_features(items)

    # 32 PCA components + 4 metadata features (n_visits, total_months, age, sex) = 36
    assert X.shape == (40, 36)
    assert pca is not None
    assert pca.n_components == 32


def test_extract_features_pca_reuse():
    adapter = _make_adapter()
    train_items = [_make_synthetic_item(f"sub_{i}", i % 2, n_rois=15) for i in range(40)]
    val_items = [_make_synthetic_item(f"sub_val_{i}", i % 2, n_rois=15) for i in range(10)]

    X_tr, pca = adapter._extract_features(train_items)
    X_va, pca_out = adapter._extract_features(val_items, pca=pca)

    assert X_tr.shape == (40, 36)
    assert X_va.shape == (10, 36)
    assert pca is pca_out


def test_extract_features_empty_raises():
    adapter = _make_adapter()
    with pytest.raises(ValueError, match="Empty items list provided"):
        adapter._extract_features([])


def test_train_fold_returns_correct_keys():
    adapter = _make_adapter()
    adapter.n_folds = 2

    # Use enough items so internal CV doesn't fail on split
    train_items = [_make_synthetic_item(f"sub_{i}", i % 2, n_rois=15) for i in range(20)]
    val_items = [_make_synthetic_item(f"sub_val_{i}", i % 2, n_rois=15) for i in range(4)]

    bundle_tr = Bundle(
        [it["label"] for it in train_items],
        [it["subject_id"] for it in train_items],
        train_items
    )
    bundle_va = Bundle(
        [it["label"] for it in val_items],
        [it["subject_id"] for it in val_items],
        val_items
    )

    rng = np.random.default_rng(42)
    res = adapter.train_fold(bundle_tr, bundle_va, cfg={}, rng=rng, device=None)

    expected_keys = {"state_dict", "val_metrics", "best_threshold", "oof_probs", "oof_targets", "oof_sids"}
    assert set(res.keys()) == expected_keys

    state_dict = res["state_dict"]
    assert {"pca", "scaler", "clf", "threshold"}.issubset(state_dict.keys())
    assert len(res["oof_probs"]) == 4
    assert len(res["oof_targets"]) == 4
    assert len(res["oof_sids"]) == 4
