"""Unit tests for the W&B tracking wrapper (no network / no wandb needed)."""
from __future__ import annotations

from CLASSIFIER.common import tracking

EXP = {"id": "exp-x", "mode": "static", "model": "GAAE", "dataset": "X", "seed": 42}
PARAMS = {"SEED": 42, "epochs": 3}


def test_disabled_mode_returns_noop(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    run = tracking.init_run(EXP, PARAMS)
    assert isinstance(run, tracking._NoOpRun)
    # No-op surface must be safe to call.
    tracking.log_metrics(run, {"loss": 1.0})
    tracking.finish_run(run)


def test_experiment_wandb_false_disables(monkeypatch):
    monkeypatch.delenv("WANDB_MODE", raising=False)
    run = tracking.init_run({**EXP, "wandb": False}, PARAMS)
    assert isinstance(run, tracking._NoOpRun)


def test_noop_run_has_log_and_finish():
    run = tracking._NoOpRun()
    assert run.log({"x": 1}) is None
    assert run.finish() is None


def test_log_metrics_swallows_errors():
    class Boom:
        def log(self, *a, **k):
            raise RuntimeError("boom")

    # Should warn, not raise.
    tracking.log_metrics(Boom(), {"a": 1})


def test_run_name_splices_local_display_name_and_timestamp():
    params = {**PARAMS, "RUN_NAME": "classic-wind-17-c68891b6f-2026-06-22_01-11-38"}
    kwargs = tracking._build_init_kwargs(EXP, params, None)
    assert kwargs["name"] == "classic-wind-17-exp-x-2026-06-22_01-11-38"


def test_run_name_falls_back_without_local_run_name():
    kwargs = tracking._build_init_kwargs(EXP, PARAMS, None)
    assert kwargs["name"] == "exp-x"


def test_run_name_keeps_fold_suffix():
    params = {**PARAMS, "RUN_NAME": "classic-wind-17-c68891b6f-2026-06-22_01-11-38"}
    kwargs = tracking._build_init_kwargs(EXP, params, 2)
    assert kwargs["name"] == "classic-wind-17-exp-x-2026-06-22_01-11-38-fold2"
