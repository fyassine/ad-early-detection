"""Unit tests for the experiment-runner registry helpers."""
from __future__ import annotations

import json
import textwrap

import pytest

from CLASSIFIER.common import experiment_utils as eu


def _write_registry(tmp_path, body: str):
    path = tmp_path / "experiments.yaml"
    path.write_text(textwrap.dedent(body))
    return path


VALID = """
experiments:
  - id: gelstm-test
    mode: longitudinal
    model: GELSTM
    dataset: DELCODE_WHOLE_BRAIN
    seed: 42
    notebook: notebooks/LONGITUDINAL/LONGITUDINAL_GELSTM_DELCODE.ipynb
    threshold_mode: best-f1
    hyperparams:
      epochs: 7
      lstm_hidden: 256
"""


def test_load_experiment_returns_entry(tmp_path):
    reg = _write_registry(tmp_path, VALID)
    exp = eu.load_experiment(reg, "gelstm-test")
    assert exp["model"] == "GELSTM"


def test_load_experiment_unknown_id_raises(tmp_path):
    reg = _write_registry(tmp_path, VALID)
    with pytest.raises(ValueError, match="No experiment with id"):
        eu.load_experiment(reg, "does-not-exist")


def test_missing_required_field_fails_loudly(tmp_path):
    reg = _write_registry(tmp_path, """
    experiments:
      - id: broken
        mode: static
        model: GAAE
        dataset: X
        notebook: foo.ipynb
    """)  # missing seed
    with pytest.raises(ValueError, match="missing required field"):
        eu.load_registry(reg)


def test_duplicate_ids_fail(tmp_path):
    reg = _write_registry(tmp_path, """
    experiments:
      - {id: dup, mode: static, model: GAAE, dataset: X, seed: 1, notebook: a.ipynb}
      - {id: dup, mode: static, model: GAAE, dataset: X, seed: 1, notebook: b.ipynb}
    """)
    with pytest.raises(ValueError, match="Duplicate experiment id"):
        eu.load_registry(reg)


def test_fixed_threshold_requires_value(tmp_path):
    reg = _write_registry(tmp_path, """
    experiments:
      - id: f
        mode: static
        model: LogReg
        dataset: X
        seed: 1
        notebook: a.ipynb
        threshold_mode: fixed
    """)
    with pytest.raises(ValueError, match="requires 'fixed_threshold'"):
        eu.load_registry(reg)


def test_invalid_threshold_mode_fails(tmp_path):
    reg = _write_registry(tmp_path, """
    experiments:
      - id: f
        mode: static
        model: LogReg
        dataset: X
        seed: 1
        notebook: a.ipynb
        threshold_mode: bogus
    """)
    with pytest.raises(ValueError, match="threshold_mode"):
        eu.load_registry(reg)


def test_build_config_merge_order(tmp_path):
    """dataclass defaults < JSON config < hyperparams."""
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "c.json").write_text(json.dumps({"epochs": 50, "lstm_hidden": 128}))
    exp = {
        "id": "x", "model": "GELSTM",
        "config_path": "configs/c.json",
        "hyperparams": {"epochs": 7},  # overrides JSON
    }
    cfg = eu.build_config(exp, tmp_path)
    assert cfg["epochs"] == 7          # hyperparams wins
    assert cfg["lstm_hidden"] == 128   # from JSON
    assert cfg["lr"] == 1e-3           # untouched dataclass default


def test_build_parameter_dict_keys(tmp_path):
    exp = eu.load_experiment(_write_registry(tmp_path, VALID), "gelstm-test")
    params = eu.build_parameter_dict(exp, tmp_path)
    for key in ("EXPERIMENT_ID", "SEED", "THRESHOLD_MODE", "WANDB_ENABLED",
                "OUTPUT_DIR", "RESOLVED_CONFIG", "RUN_DIR", "RUN_NAME"):
        assert key in params
    assert params["THRESHOLD_MODE"] == "best-f1"
    assert params["RESOLVED_CONFIG"]["epochs"] == 7
    assert params["OUTPUT_DIR"] == "outputs/gelstm-test"


def test_collect_results_writes_ledger(tmp_path):
    run_dir = tmp_path / "exp-a" / "runs" / "2026-01-01_00-00-00"
    run_dir.mkdir(parents=True)
    (run_dir / "run_summary.json").write_text(json.dumps({
        "experiment_id": "exp-a",
        "timestamp": "2026-01-01_00-00-00",
        "git": {"short_commit": "abc123def", "dirty": False},
        "metrics": {"test_auc": 0.81, "test_f1": 0.7},
    }))
    rows = eu.collect_results(tmp_path)
    assert len(rows) == 1
    assert rows[0]["metric.test_auc"] == 0.81
    assert (tmp_path / "RESULTS.csv").is_file()
    assert (tmp_path / "RESULTS.jsonl").is_file()
