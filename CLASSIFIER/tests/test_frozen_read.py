"""Tests for CLASSIFIER.common.frozen_read — round-trips a tiny saved run."""

from __future__ import annotations

import json

import numpy as np
import pytest

from CLASSIFIER.adapters.logreg_drift import LogRegDriftAdapter
from CLASSIFIER.common.crossval import Bundle
from CLASSIFIER.common.frozen_read import build_adapter_from_run, score_frozen_split
from CLASSIFIER.common.run_artifacts import save_run


def _demo_items(prefix, n, seed):
    rng = np.random.default_rng(seed)
    return [
        {
            "subject_id": f"{prefix}{i}",
            "label": int(i % 2),
            "age": float(rng.uniform(0, 1)),
            "sex": int(i % 2),
            "n_scans": 2,
        }
        for i in range(n)
    ]


def _bundle(items):
    return Bundle([it["label"] for it in items], [it["subject_id"] for it in items], items)


def _save_tiny_run(tmp_path):
    train_items = _demo_items("tr", 20, seed=0)
    val_items = _demo_items("va", 6, seed=1)
    bundle_tr, bundle_va = _bundle(train_items), _bundle(val_items)

    adapter = LogRegDriftAdapter(
        gaae_ckpt_path="",
        gaae_hp={},
        train_config={"min_visits": 2, "feature_set": "demo"},
        data_root="",
        cohorts_csv="",
        device="cpu",
        rng=None,
    )
    adapter.n_folds = 2
    fold_out = adapter.train_fold(bundle_tr, bundle_va, cfg={}, rng=np.random.default_rng(2), device="cpu")

    run_dir_arg = tmp_path / "runs" / "tiny-run"
    _run_name, run_dir_path = save_run(
        output_dir=str(tmp_path),
        run_dir=str(run_dir_arg),
        run_name="tiny-run",
        model_state=adapter.model_state_for_save(fold_out["state_dict"]),
        model_config=adapter.model_config(),
        training_config={"min_visits": 2, "feature_set": "demo"},
        data_info={},
        dataset_info={},
        rng=np.random.default_rng(3),
        best_val_auc=fold_out["val_metrics"]["auc"],
        active_threshold=fold_out["best_threshold"],
        threshold_method="oof_f1",
        best_fold=1,
        cv_results={
            "fold": [1],
            "val_auc": [fold_out["val_metrics"]["auc"]],
            "best_threshold": [fold_out["best_threshold"]],
        },
        gaae_checkpoint="",
        gaae_run_name="none",
        source_files=adapter.source_files(),
        n_folds=1,
        model_tag=adapter.model_tag,
    )
    return run_dir_path, bundle_va


def test_build_adapter_from_run_reproduces_training_config(tmp_path):
    run_dir, _ = _save_tiny_run(tmp_path)
    adapter, summary = build_adapter_from_run(
        run_dir, adapter_key="logregdrift", data_root="", cohorts_csv="",
        gaae_ckpt_path="", gaae_hp={}, device="cpu",
    )
    assert isinstance(adapter, LogRegDriftAdapter)
    assert adapter.feature_set == "demo"
    assert summary["training_config"]["feature_set"] == "demo"


def test_load_state_round_trips_composite_state(tmp_path):
    run_dir, bundle_va = _save_tiny_run(tmp_path)
    adapter, _ = build_adapter_from_run(
        run_dir, adapter_key="logregdrift", data_root="", cohorts_csv="",
        gaae_ckpt_path="", gaae_hp={}, device="cpu",
    )
    state = adapter.load_state(run_dir)
    assert {"pca", "scaler", "clf", "threshold"}.issubset(state.keys())

    res = adapter.eval_split(state, bundle_va, state["threshold"], device="cpu")
    assert len(res["probs"]) == len(bundle_va)


def test_score_frozen_split_records_test_metrics(tmp_path, monkeypatch):
    run_dir, bundle_va = _save_tiny_run(tmp_path)
    monkeypatch.setattr(LogRegDriftAdapter, "prepare_data", lambda self, df: bundle_va)

    metrics = score_frozen_split(
        run_dir, df=None, adapter_key="logregdrift", data_root="", cohorts_csv="",
        gaae_ckpt_path="", gaae_hp={}, device="cpu", record_as="test",
    )
    assert "auc" in metrics

    summary = json.loads((run_dir / "run_summary.json").read_text())
    assert summary["metrics"]["test_auc"] == pytest.approx(metrics["auc"])


def test_score_frozen_split_external_records_ext_metrics(tmp_path, monkeypatch):
    run_dir, bundle_va = _save_tiny_run(tmp_path)
    monkeypatch.setattr(LogRegDriftAdapter, "prepare_data", lambda self, df: bundle_va)

    score_frozen_split(
        run_dir, df=None, adapter_key="logregdrift", data_root="", cohorts_csv="",
        gaae_ckpt_path="", gaae_hp={}, device="cpu", record_as="external", cohort="oasis3",
    )
    summary = json.loads((run_dir / "run_summary.json").read_text())
    assert "ext_oasis3_auc" in summary["metrics"]


def test_score_frozen_split_external_requires_cohort(tmp_path, monkeypatch):
    run_dir, bundle_va = _save_tiny_run(tmp_path)
    monkeypatch.setattr(LogRegDriftAdapter, "prepare_data", lambda self, df: bundle_va)
    with pytest.raises(ValueError, match="cohort"):
        score_frozen_split(
            run_dir, df=None, adapter_key="logregdrift", data_root="", cohorts_csv="",
            gaae_ckpt_path="", gaae_hp={}, device="cpu", record_as="external",
        )


def test_score_frozen_split_invalid_record_as_raises(tmp_path, monkeypatch):
    run_dir, bundle_va = _save_tiny_run(tmp_path)
    monkeypatch.setattr(LogRegDriftAdapter, "prepare_data", lambda self, df: bundle_va)
    with pytest.raises(ValueError, match="record_as"):
        score_frozen_split(
            run_dir, df=None, adapter_key="logregdrift", data_root="", cohorts_csv="",
            gaae_ckpt_path="", gaae_hp={}, device="cpu", record_as="bogus",
        )
