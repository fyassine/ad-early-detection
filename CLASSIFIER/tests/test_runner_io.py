"""Tests for CLASSIFIER.common.runner_io (terminal helpers for the runners)."""
from __future__ import annotations

import io

from common.runner_io import color, format_elapsed, supports_color


def test_format_elapsed_minutes_seconds():
    assert format_elapsed(0) == "00:00"
    assert format_elapsed(5) == "00:05"
    assert format_elapsed(65) == "01:05"
    assert format_elapsed(600) == "10:00"


def test_format_elapsed_hours():
    assert format_elapsed(3661) == "1:01:01"
    assert format_elapsed(7325) == "2:02:05"


def test_color_noop_on_non_tty():
    # A plain StringIO is not a TTY -> no ANSI codes, text returned verbatim.
    buf = io.StringIO()
    assert not supports_color(buf)
    assert color("ok", "green", stream=buf) == "ok"


def test_color_wraps_when_supported(monkeypatch):
    class _TTY(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.delenv("NO_COLOR", raising=False)
    tty = _TTY()
    out = color("ok", "green", stream=tty)
    assert out.startswith("\033[32m") and out.endswith("\033[0m") and "ok" in out


def test_no_color_env_disables(monkeypatch):
    class _TTY(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setenv("NO_COLOR", "1")
    tty = _TTY()
    assert color("ok", "green", stream=tty) == "ok"


def test_format_metric_summary():
    from common.runner_io import format_metric_summary
    out = format_metric_summary({"test_auc": 0.532, "threshold_method": "oof_f1"})
    assert "test_auc 0.532" in out
    assert "threshold_method oof_f1" in out


def test_format_cv_summary():
    from common.runner_io import format_cv_summary
    cv = {
        "n_folds": 5,
        "val_auc_mean": 0.9816, "val_auc_std": 0.0107,
        "val_f1_mean": 0.9351, "val_f1_std": 0.0237,
        "best_fold": 4, "best_val_auc": 0.9938,
    }
    out = format_cv_summary(cv)
    assert out.startswith("5 folds —")
    assert "val_auc 0.982±0.011" in out
    assert "best fold 4 (val_auc 0.994)" in out


def test_format_cv_summary_empty():
    from common.runner_io import format_cv_summary
    assert format_cv_summary({}) == ""
