"""Tests for CLASSIFIER.common.run_artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from CLASSIFIER.common.seeding import make_rng
from CLASSIFIER.common.run_artifacts import save_run, record_test_metrics


def _save_kwargs(output_dir):
    return dict(
        output_dir=str(output_dir),
        run_dir=None,
        run_name=None,
        model_state={"w": torch.zeros(3)},
        model_config={"model_type": "Stub", "in_features": 3},
        training_config={"epochs": 1, "n_folds": 5},
        data_info={"region": "whole-brain", "atlas": "sch200", "dataset_dir": "d"},
        dataset_info={"region": "whole-brain", "n_folds": 5},
        rng=make_rng(0),
        best_val_auc=0.83,
        active_threshold=0.42,
        threshold_method="oof_f1",
        best_fold=2,
        cv_results={"val_auc": [0.8, 0.83]},
        gaae_checkpoint="/fake/gaae.pth",
        gaae_run_name="gaae-run",
        source_files=[Path(__file__)],  # an existing file to snapshot
        n_folds=5,
        model_tag="sample",
    )


def test_save_run_writes_all_artifacts(tmp_path):
    run_name, run_dir = save_run(**_save_kwargs(tmp_path))

    assert run_dir.is_dir()
    assert "whole-brain" in run_name
    assert (run_dir / f"model_{run_name}.pth").exists()
    assert (run_dir / f"checkpoint_{run_name}.pth").exists()
    assert (run_dir / "run_summary.json").exists()
    assert (run_dir / "source").is_dir()

    summary = json.loads((run_dir / "run_summary.json").read_text())
    assert summary["best_fold"] == 2
    assert summary["active_threshold"] == 0.42
    assert summary["threshold_method"] == "oof_f1"
    assert summary["model_config"]["model_type"] == "Stub"

    # Full-state checkpoint is reloadable and self-describing.
    ckpt = torch.load(run_dir / f"checkpoint_{run_name}.pth", weights_only=False)
    assert "model_state_dict" in ckpt and "model_config" in ckpt
    assert ckpt["best_threshold"] == 0.42


def test_save_run_honours_preset_run_dir(tmp_path):
    preset = tmp_path / "preset_run"
    kwargs = _save_kwargs(tmp_path)
    kwargs.update(run_dir=str(preset), run_name="my-run")
    run_name, run_dir = save_run(**kwargs)
    assert run_name == "my-run"
    assert run_dir == preset
    assert (preset / "model_my-run.pth").exists()


def test_record_test_metrics_patches_summary(tmp_path):
    run_name, run_dir = save_run(**_save_kwargs(tmp_path))
    metrics = {
        "auc": 0.77,
        "sensitivity": 0.7,
        "specificity": 0.8,
        "f1": 0.72,
        "probs": np.array([0.1, 0.9]),
        "targets": np.array([0, 1]),
    }
    record_test_metrics(run_dir, metrics, threshold=0.42, threshold_method="oof_f1")

    summary = json.loads((run_dir / "run_summary.json").read_text())
    assert summary["test_auc"] == 0.77
    assert summary["metrics"]["test_f1"] == 0.72
    assert summary["test_probabilities"] == [0.1, 0.9]
    assert summary["test_labels"] == [0, 1]
