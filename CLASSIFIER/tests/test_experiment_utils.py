"""Unit tests for the experiment-runner registry helpers."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

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
    reg = _write_registry(
        tmp_path,
        """
    experiments:
      - id: broken
        mode: static
        model: GAAE
        dataset: X
        notebook: foo.ipynb
    """,
    )  # missing seed
    with pytest.raises(ValueError, match="missing required field"):
        eu.load_registry(reg)


def test_duplicate_ids_fail(tmp_path):
    reg = _write_registry(
        tmp_path,
        """
    experiments:
      - {id: dup, mode: static, model: GAAE, dataset: X, seed: 1, notebook: a.ipynb}
      - {id: dup, mode: static, model: GAAE, dataset: X, seed: 1, notebook: b.ipynb}
    """,
    )
    with pytest.raises(ValueError, match="Duplicate experiment id"):
        eu.load_registry(reg)


def test_fixed_threshold_requires_value(tmp_path):
    reg = _write_registry(
        tmp_path,
        """
    experiments:
      - id: f
        mode: static
        model: LogReg
        dataset: X
        seed: 1
        notebook: a.ipynb
        threshold_mode: fixed
    """,
    )
    with pytest.raises(ValueError, match="requires 'fixed_threshold'"):
        eu.load_registry(reg)


def test_invalid_threshold_mode_fails(tmp_path):
    reg = _write_registry(
        tmp_path,
        """
    experiments:
      - id: f
        mode: static
        model: LogReg
        dataset: X
        seed: 1
        notebook: a.ipynb
        threshold_mode: bogus
    """,
    )
    with pytest.raises(ValueError, match="threshold_mode"):
        eu.load_registry(reg)


def test_build_config_merge_order(tmp_path):
    """dataclass defaults < JSON config < hyperparams < registry seed."""
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "c.json").write_text(json.dumps({"epochs": 50, "lstm_hidden": 128}))
    exp = {
        "id": "x",
        "model": "GELSTM",
        "seed": 43,
        "config_path": "configs/c.json",
        "hyperparams": {"epochs": 7},  # overrides JSON
    }
    cfg = eu.build_config(exp, tmp_path)
    assert cfg["epochs"] == 7  # hyperparams wins
    assert cfg["lstm_hidden"] == 128  # from JSON
    assert cfg["lr"] == 1e-3  # untouched dataclass default
    assert cfg["seed"] == 43  # registry seed, from exp["seed"]


def test_build_config_registry_seed_shadows_config_and_hyperparams(tmp_path):
    """A JSON config_path or a hyperparams block claiming a different seed must
    not win — the registry's exp["seed"] is the single source of truth for
    what ends up in run_summary.json's training_config.seed. Regression test
    for the bug where every run on disk recorded seed=42 (the dataclass
    default) regardless of the seed it actually trained with."""
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "c.json").write_text(json.dumps({"seed": 42}))
    exp = {
        "id": "x",
        "model": "GELSTM",
        "seed": 45,
        "config_path": "configs/c.json",
        "hyperparams": {"seed": 42},
    }
    cfg = eu.build_config(exp, tmp_path)
    assert cfg["seed"] == 45


def test_build_parameter_dict_keys(tmp_path):
    exp = eu.load_experiment(_write_registry(tmp_path, VALID), "gelstm-test")
    params = eu.build_parameter_dict(exp, tmp_path)
    for key in (
        "EXPERIMENT_ID",
        "SEED",
        "THRESHOLD_MODE",
        "WANDB_ENABLED",
        "OUTPUT_DIR",
        "RESOLVED_CONFIG",
        "RUN_DIR",
        "RUN_NAME",
    ):
        assert key in params
    assert params["THRESHOLD_MODE"] == "best-f1"
    assert params["RESOLVED_CONFIG"]["epochs"] == 7
    assert params["OUTPUT_DIR"] == "outputs/gelstm-test"


def test_collect_results_writes_ledger(tmp_path):
    run_dir = tmp_path / "exp-a" / "runs" / "2026-01-01_00-00-00"
    run_dir.mkdir(parents=True)
    (run_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "experiment_id": "exp-a",
                "timestamp": "2026-01-01_00-00-00",
                "git": {"short_commit": "abc123def", "dirty": False},
                "metrics": {"test_auc": 0.81, "test_f1": 0.7},
            }
        )
    )
    rows = eu.collect_results(tmp_path)
    assert len(rows) == 1
    assert rows[0]["metric.test_auc"] == 0.81
    assert (tmp_path / "RESULTS.csv").is_file()
    assert (tmp_path / "RESULTS.jsonl").is_file()


def test_collect_results_includes_cv_summary(tmp_path):
    """cv_results per-fold lists become cv.* mean/std columns in the ledger."""
    run_dir = tmp_path / "gelstm-test" / "runs" / "leafy-oasis-7"
    run_dir.mkdir(parents=True)
    summary = {
        "experiment_id": "gelstm-test",
        "git": {"short_commit": "abc1234", "dirty": False},
        "metrics": {"test_auc": 0.53, "test_f1": 0.65},
        "cv_results": {
            "val_auc": [0.98, 0.97, 0.96, 0.99, 0.98],
            "val_f1": [0.95, 0.90, 0.90, 0.95, 0.95],
            "best_threshold": [0.6, 0.28, 0.33, 0.76, 0.37],
        },
        "best_fold": 4,
        "best_val_auc": 0.99,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary))

    rows = eu.collect_results(tmp_path)
    row = next(r for r in rows if r["experiment_id"] == "gelstm-test")
    assert row["cv.n_folds"] == 5
    assert row["cv.best_fold"] == 4
    assert row["cv.best_val_auc"] == 0.99
    assert row["cv.val_auc_mean"] == pytest.approx(0.976, abs=1e-3)
    assert row["cv.val_auc_std"] > 0
    assert "cv.val_f1_mean" in row
    # best_threshold is not a val_* metric -> not summarised as mean/std.
    assert "cv.best_threshold_mean" not in row
    # test metrics still present.
    assert row["metric.test_auc"] == 0.53


def test_collect_results_no_cv_block(tmp_path):
    """Runs without cv_results get no cv.* columns (sanity/comparison notebooks)."""
    run_dir = tmp_path / "sanity" / "runs" / "calm-lake-1"
    run_dir.mkdir(parents=True)
    (run_dir / "run_summary.json").write_text(
        json.dumps({"experiment_id": "sanity", "metrics": {"test_auc": 0.7}})
    )
    rows = eu.collect_results(tmp_path)
    row = next(r for r in rows if r["experiment_id"] == "sanity")
    assert not any(k.startswith("cv.") for k in row)


def test_collect_results_and_read_statuses_duration(tmp_path):
    """Duration is collected from status.json or run_summary.json."""
    run_dir = tmp_path / "exp1" / "runs" / "fast-wind-1"
    run_dir.mkdir(parents=True)
    (run_dir / "run_summary.json").write_text(
        json.dumps({"experiment_id": "exp1", "metrics": {"test_auc": 0.9}})
    )
    (run_dir / "status.json").write_text(
        json.dumps({"state": "done", "started_at": "2026-08-21T10:00:00", "duration_seconds": 15.5})
    )
    rows = eu.collect_results(tmp_path)
    assert rows[0]["duration_seconds"] == 15.5

    statuses = eu.read_statuses(tmp_path)
    assert len(statuses) == 1
    assert statuses[0]["duration_seconds"] == 15.5


def test_read_statuses_limit(tmp_path):
    for i in range(5):
        run_dir = tmp_path / f"exp{i}" / "runs" / f"run-{i}"
        run_dir.mkdir(parents=True)
        (run_dir / "status.json").write_text(
            json.dumps({"state": "done", "started_at": f"2026-08-21T1{i}:00:00", "run_name": f"run-{i}"})
        )

    all_statuses = eu.read_statuses(tmp_path)
    assert len(all_statuses) == 5

    limited_statuses = eu.read_statuses(tmp_path, limit=2)
    assert len(limited_statuses) == 2
    # Check that it returns most recent first
    assert limited_statuses[0]["started_at"] == "2026-08-21T14:00:00"
    assert limited_statuses[1]["started_at"] == "2026-08-21T13:00:00"


def test_read_statuses_experiment_id_filter(tmp_path):
    for exp_id in ("exp-alpha", "exp-beta"):
        for i in range(2):
            run_dir = tmp_path / exp_id / "runs" / f"run-{i}"
            run_dir.mkdir(parents=True)
            (run_dir / "status.json").write_text(
                json.dumps({"experiment_id": exp_id, "state": "done", "started_at": f"2026-08-21T1{i}:00:00"})
            )

    alpha_statuses = eu.read_statuses(tmp_path, experiment_id="exp-alpha")
    assert len(alpha_statuses) == 2
    assert all(s["experiment_id"] == "exp-alpha" for s in alpha_statuses)

    beta_statuses = eu.read_statuses(tmp_path, experiment_id="exp-beta")
    assert len(beta_statuses) == 2
    assert all(s["experiment_id"] == "exp-beta" for s in beta_statuses)


def test_find_latest_run_symlink_and_fallback(tmp_path):
    exp_dir = tmp_path / "exp-gamma"
    run1 = exp_dir / "runs" / "run-1"
    run2 = exp_dir / "runs" / "run-2"
    run1.mkdir(parents=True)
    run2.mkdir(parents=True)
    (run1 / "status.json").write_text(json.dumps({"started_at": "2026-08-21T10:00:00"}))
    (run2 / "status.json").write_text(json.dumps({"started_at": "2026-08-21T11:00:00"}))

    # Without symlink, picks most recent
    found = eu.find_latest_run(tmp_path, "exp-gamma")
    assert found.resolve() == run2.resolve()

    # With symlink, follows symlink
    latest_link = exp_dir / "latest"
    latest_link.symlink_to(Path("runs") / "run-1")
    found_sym = eu.find_latest_run(tmp_path, "exp-gamma")
    assert found_sym.resolve() == run1.resolve()


def test_read_statuses_auto_reconciles_dead_runs(tmp_path):
    run_dir = tmp_path / "exp-reconcile" / "runs" / "dead-probe-run"
    run_dir.mkdir(parents=True)
    status_file = run_dir / "status.json"
    status_file.write_text(
        json.dumps({
            "experiment_id": "exp-reconcile",
            "run_name": "dead-probe-run",
            "state": "running",
            "pid": 99999999,
            "started_at": "2026-08-21T10:00:00",
            "git_commit": "deadbeef1",
        })
    )

    statuses = eu.read_statuses(tmp_path)
    assert len(statuses) == 1
    assert statuses[0]["state"] == "killed"
    assert "abruptly" in statuses[0]["error"]

    # Verify run_summary.json created
    assert (run_dir / "run_summary.json").is_file()
    summary = json.loads((run_dir / "run_summary.json").read_text())
    assert summary["state"] == "killed"
    assert summary["git"]["short_commit"] == "deadbeef1"

    # Also collect_results now collects it
    rows = eu.collect_results(tmp_path)
    assert len(rows) == 1
    assert rows[0]["experiment_id"] == "exp-reconcile"
    assert rows[0]["state"] == "killed"



