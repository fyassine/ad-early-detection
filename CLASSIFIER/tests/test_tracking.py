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
