from __future__ import annotations

import io
import time

from SHARED.runner_io import color, format_elapsed, supports_color


def test_format_elapsed_minutes_seconds():
    assert format_elapsed(0) == "00:00"
    assert format_elapsed(5) == "00:05"
    assert format_elapsed(65) == "01:05"
    assert format_elapsed(600) == "10:00"


def test_format_elapsed_none_or_invalid():
    assert format_elapsed(None) == "-"
    assert format_elapsed("invalid") == "-"  # type: ignore


def test_format_elapsed_hours():
    assert format_elapsed(3661) == "1:01:01"
    assert format_elapsed(7325) == "2:02:05"


def test_format_elapsed_always_show_hours():
    assert format_elapsed(0, always_show_hours=True) == "00:00:00"
    assert format_elapsed(5, always_show_hours=True) == "00:00:05"
    assert format_elapsed(65, always_show_hours=True) == "00:01:05"
    assert format_elapsed(3665, always_show_hours=True) == "01:01:05"


def test_infer_notebook_duration(tmp_path):
    import json
    from SHARED.runner_io import infer_notebook_duration

    nb = tmp_path / "test.ipynb"
    nb.write_text(
        json.dumps(
            {
                "metadata": {"papermill": {"duration": 45.2}},
                "cells": [],
            }
        )
    )
    assert infer_notebook_duration(nb) == 45.2


def test_infer_run_duration_from_status(tmp_path):
    import json
    from SHARED.runner_io import infer_run_duration

    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    (run_dir / "status.json").write_text(json.dumps({"duration_seconds": 12.3}))
    assert infer_run_duration(run_dir) == 12.3


def test_infer_run_duration_from_notebook(tmp_path):
    import json
    from SHARED.runner_io import infer_run_duration

    run_dir = tmp_path / "run2"
    run_dir.mkdir()
    (run_dir / "test_run.ipynb").write_text(
        json.dumps({"metadata": {"papermill": {"duration": 88.5}}, "cells": []})
    )
    assert infer_run_duration(run_dir) == 88.5


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
    from SHARED.runner_io import format_metric_summary

    out = format_metric_summary({"test_auc": 0.532, "threshold_method": "oof_f1"})
    assert "test_auc 0.532" in out
    assert "threshold_method oof_f1" in out


def test_format_cv_summary():
    from SHARED.runner_io import format_cv_summary

    cv = {
        "n_folds": 5,
        "val_auc_mean": 0.9816,
        "val_auc_std": 0.0107,
        "val_f1_mean": 0.9351,
        "val_f1_std": 0.0237,
        "best_fold": 4,
        "best_val_auc": 0.9938,
    }
    out = format_cv_summary(cv)
    assert out.startswith("5 folds —")
    assert "val_auc 0.982±0.011" in out
    assert "best fold 4 (val_auc 0.994)" in out


def test_format_cv_summary_empty():
    from SHARED.runner_io import format_cv_summary

    assert format_cv_summary({}) == ""


def test_render_status_table(capsys):
    from SHARED.runner_io import render_status_table

    statuses = [
        {"experiment_id": "exp1", "state": "done", "started_at": "2026-08-20T10:00:00", "duration_seconds": 20, "run_name": "run1"},
        {"experiment_id": "exp2", "state": "running", "started_at": "2026-08-20T11:00:00", "duration_seconds": None, "run_name": "run2"},
        {"experiment_id": "exp3", "state": "failed", "started_at": "2026-08-20T12:00:00", "duration_seconds": 5, "run_name": "run3"},
        {"experiment_id": "exp4", "state": "interrupted", "started_at": "2026-08-20T13:00:00", "duration_seconds": 3, "run_name": "run4"},
    ]
    render_status_table(statuses)
    out = capsys.readouterr().out
    assert "STATE" in out
    assert "exp1" in out
    assert "exp2" in out
    assert "exp3" in out
    assert "interrupted" in out
    assert "exp4" in out
    assert "00:00:20" in out
    assert "00:00:05" in out
    assert "00:00:03" in out


def test_render_status_table_limit(capsys):
    from SHARED.runner_io import render_status_table

    statuses = [
        {"experiment_id": "exp1", "state": "done", "started_at": "2026-08-20T10:00:00", "run_name": "run1"},
        {"experiment_id": "exp2", "state": "done", "started_at": "2026-08-20T11:00:00", "run_name": "run2"},
        {"experiment_id": "exp3", "state": "done", "started_at": "2026-08-20T12:00:00", "run_name": "run3"},
    ]
    render_status_table(statuses, limit=2)
    out = capsys.readouterr().out
    assert "exp1" in out
    assert "exp2" in out
    assert "exp3" not in out


def test_render_status_table_empty(capsys):
    from SHARED.runner_io import render_status_table

    render_status_table([])
    out = capsys.readouterr().out
    assert "No runs found under outputs/." in out


def test_render_status_table_limit_zero(capsys):
    from SHARED.runner_io import render_status_table

    render_status_table([{"experiment_id": "exp1"}], limit=0)
    out = capsys.readouterr().out
    assert "No runs to display" in out


def test_watch_status_table_max_iterations():
    from SHARED.runner_io import watch_status_table

    calls = 0

    def mock_status_fn():
        nonlocal calls
        calls += 1
        return [{"experiment_id": f"exp{calls}", "state": "running", "run_name": f"run{calls}"}]

    buf = io.StringIO()
    watch_status_table(mock_status_fn, interval=0.01, limit=1, stream=buf, max_iterations=2)
    assert calls == 2
    output = buf.getvalue()
    assert "Experiment Status" in output
    assert "exp1" in output
    assert "exp2" in output


def test_watch_status_table_handles_keyboard_interrupt(monkeypatch):
    import time
    from SHARED.runner_io import watch_status_table

    def mock_sleep(_):
        raise KeyboardInterrupt()

    monkeypatch.setattr(time, "sleep", mock_sleep)
    buf = io.StringIO()
    # Should not raise KeyboardInterrupt
    watch_status_table(lambda: [{"experiment_id": "exp1"}], interval=0.01, stream=buf)
    assert "exp1" in buf.getvalue()


def test_classifier_parse_args():
    from CLASSIFIER.run_experiment import parse_args

    args = parse_args(["--status"])
    assert args.status is True
    assert args.watch is False
    assert args.limit is None

    args = parse_args(["--status", "--watch"])
    assert args.status is True
    assert args.watch is True

    args = parse_args(["-w", "-n", "5", "--interval", "1.5"])
    assert args.watch is True
    assert args.limit == 5
    assert args.interval == 1.5

    args = parse_args(["--status", "--lines", "10"])
    assert args.limit == 10

    args = parse_args(["--id", "my-exp", "--follow"])
    assert args.id == "my-exp"
    assert args.follow is True

    args = parse_args(["--follow", "my-exp"])
    assert args.follow == "my-exp"


def test_follow_run_log_completed(tmp_path):
    import json
    from SHARED.runner_io import follow_run_log

    run_dir = tmp_path / "runs" / "run-done"
    run_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text(
        json.dumps({"experiment_id": "exp1", "run_name": "run-done", "state": "done", "duration_seconds": 12.0})
    )
    (run_dir / "run.log").write_text("Epoch 1/5\nEpoch 2/5\nFinished training.\n")

    buf = io.StringIO()
    follow_run_log(run_dir, lines=2, stream=buf)
    output = buf.getvalue()
    assert "Experiment: exp1" in output
    assert "Epoch 2/5" in output
    assert "Finished training." in output
    assert "✓ DONE (00:00:12)" in output


def test_follow_run_log_streaming_to_done(tmp_path):
    import json
    import os
    import threading
    from SHARED.runner_io import follow_run_log

    run_dir = tmp_path / "runs" / "run-streaming"
    run_dir.mkdir(parents=True)
    status_file = run_dir / "status.json"
    log_file = run_dir / "run.log"

    status_file.write_text(
        json.dumps({
            "experiment_id": "exp-stream",
            "run_name": "run-streaming",
            "state": "running",
            "pid": os.getpid(),
        })
    )
    log_file.write_text("Initial line\n")

    def _simulate_run():
        time.sleep(0.05)
        with open(log_file, "a") as f:
            f.write("Appended line 1\n")
            f.write("Appended line 2\n")
        time.sleep(0.05)
        status_file.write_text(
            json.dumps({
                "experiment_id": "exp-stream",
                "run_name": "run-streaming",
                "state": "done",
                "duration_seconds": 3.0,
            })
        )

    t = threading.Thread(target=_simulate_run)
    t.start()

    buf = io.StringIO()
    follow_run_log(run_dir, poll_interval=0.02, stream=buf)
    t.join()

    output = buf.getvalue()
    assert "Initial line" in output
    assert "Appended line 1" in output
    assert "Appended line 2" in output
    assert "✓ DONE (00:00:03)" in output


def test_follow_run_log_keyboard_interrupt(tmp_path, monkeypatch):
    import json
    import os
    import time
    from SHARED.runner_io import follow_run_log

    run_dir = tmp_path / "runs" / "run-int"
    run_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text(
        json.dumps({"experiment_id": "exp-int", "state": "running", "pid": os.getpid()})
    )
    (run_dir / "run.log").write_text("Line 1\n")

    def mock_sleep(_):
        raise KeyboardInterrupt()

    monkeypatch.setattr(time, "sleep", mock_sleep)
    buf = io.StringIO()
    # Should not raise KeyboardInterrupt
    follow_run_log(run_dir, poll_interval=0.01, stream=buf)
    output = buf.getvalue()
    assert "Detached from log stream" in output


def test_infer_run_duration_running(tmp_path):
    import json
    import os
    from datetime import datetime, timedelta
    from SHARED.runner_io import infer_run_duration

    run_dir = tmp_path / "runs" / "run-active"
    run_dir.mkdir(parents=True)
    started = (datetime.now() - timedelta(minutes=15, seconds=20)).isoformat()
    (run_dir / "status.json").write_text(
        json.dumps({"state": "running", "started_at": started, "pid": os.getpid()})
    )
    dur = infer_run_duration(run_dir)
    assert dur is not None
    assert 915 <= dur <= 930


def test_is_process_alive():
    import os
    from SHARED.runner_io import is_process_alive

    # Current process is definitely alive
    assert is_process_alive(os.getpid()) is True

    # PID 0 or negative or None is not alive
    assert is_process_alive(None) is False
    assert is_process_alive(0) is False
    assert is_process_alive(-1) is False

    # Very unlikely PID (e.g. 99999999)
    assert is_process_alive(99999999) is False


def test_reconcile_run_status_alive(tmp_path):
    import json
    import os
    from datetime import datetime, timedelta
    from SHARED.runner_io import reconcile_run_status

    run_dir = tmp_path / "runs" / "live-run"
    run_dir.mkdir(parents=True)
    started = (datetime.now() - timedelta(seconds=10)).isoformat()
    status_file = run_dir / "status.json"
    status_file.write_text(
        json.dumps({
            "experiment_id": "test-exp",
            "run_name": "live-run",
            "state": "running",
            "pid": os.getpid(),
            "started_at": started,
        })
    )

    reconciled = reconcile_run_status(status_file, write_disk=True)
    assert reconciled["state"] == "running"
    assert reconciled["duration_seconds"] is not None
    assert 8 <= reconciled["duration_seconds"] <= 15
    # File on disk was not marked killed
    disk_data = json.loads(status_file.read_text())
    assert disk_data["state"] == "running"


def test_reconcile_run_status_dead_creates_summary(tmp_path):
    import json
    from datetime import datetime, timedelta
    from SHARED.runner_io import reconcile_run_status

    run_dir = tmp_path / "runs" / "dead-run"
    run_dir.mkdir(parents=True)
    started = (datetime.now() - timedelta(minutes=5)).isoformat(timespec="seconds")
    status_file = run_dir / "status.json"
    status_file.write_text(
        json.dumps({
            "experiment_id": "test-exp",
            "run_name": "dead-run",
            "state": "running",
            "pid": 99999999,  # non-existent PID
            "started_at": started,
            "git_commit": "abc1234",
            "git_dirty": False,
        })
    )
    # Simulate run.log created 2 minutes after start
    log_file = run_dir / "run.log"
    log_file.write_text("Epoch 1\nEpoch 2\n")

    reconciled = reconcile_run_status(status_file, write_disk=True)
    assert reconciled["state"] == "killed"
    assert "abruptly" in reconciled["error"]
    assert reconciled["duration_seconds"] is not None

    # Verify status.json updated on disk
    disk_status = json.loads(status_file.read_text())
    assert disk_status["state"] == "killed"
    assert "error" in disk_status

    # Verify fallback run_summary.json created on disk
    summary_file = run_dir / "run_summary.json"
    assert summary_file.is_file()
    summary_data = json.loads(summary_file.read_text())
    assert summary_data["state"] == "killed"
    assert summary_data["status"] == "killed"
    assert summary_data["experiment_id"] == "test-exp"
    assert summary_data["git"]["short_commit"] == "abc1234"


def test_run_lifecycle_normal(tmp_path):
    import json
    from SHARED.runner_io import RunLifecycle

    run_dir = tmp_path / "runs" / "lifecycle-run"
    run_dir.mkdir(parents=True)

    lifecycle = RunLifecycle(
        run_dir=run_dir,
        exp_id="exp-lc",
        run_name="lifecycle-run",
        git_info={"short_commit": "123456", "dirty": False},
        notebook_path="notebook.ipynb",
    )

    with lifecycle:
        # Check initial status.json
        status_file = run_dir / "status.json"
        assert status_file.is_file()
        st = json.loads(status_file.read_text())
        assert st["state"] == "running"
        assert st["experiment_id"] == "exp-lc"

        time.sleep(0.05)
        lifecycle.mark_done()

    final_status = json.loads((run_dir / "status.json").read_text())
    assert final_status["state"] == "done"
    assert final_status["exit_code"] == 0
    assert final_status["duration_seconds"] >= 0.04


def test_run_lifecycle_failure(tmp_path):
    import json
    from SHARED.runner_io import RunLifecycle

    run_dir = tmp_path / "runs" / "lifecycle-fail"
    run_dir.mkdir(parents=True)

    lifecycle = RunLifecycle(
        run_dir=run_dir,
        exp_id="exp-fail",
        run_name="lifecycle-fail",
    )

    with lifecycle:
        lifecycle.mark_failed("Custom failure error", duration_seconds=1.5)

    final_status = json.loads((run_dir / "status.json").read_text())
    assert final_status["state"] == "failed"
    assert final_status["error"] == "Custom failure error"
    assert final_status["duration_seconds"] == 1.5

    summary = json.loads((run_dir / "run_summary.json").read_text())
    assert summary["state"] == "failed"
    assert summary["error"] == "Custom failure error"


def test_run_lifecycle_unhandled_exception(tmp_path):
    import json
    import pytest
    from SHARED.runner_io import RunLifecycle

    run_dir = tmp_path / "runs" / "lifecycle-exc"
    run_dir.mkdir(parents=True)

    lifecycle = RunLifecycle(
        run_dir=run_dir,
        exp_id="exp-exc",
        run_name="lifecycle-exc",
    )

    with pytest.raises(RuntimeError, match="Crash inside run"):
        with lifecycle:
            raise RuntimeError("Crash inside run")

    final_status = json.loads((run_dir / "status.json").read_text())
    assert final_status["state"] == "failed"
    assert "Crash inside run" in final_status["error"]

    summary = json.loads((run_dir / "run_summary.json").read_text())
    assert summary["state"] == "failed"
    assert "Crash inside run" in summary["error"]


def test_render_status_table_warning_on_killed_runs(capsys):
    from datetime import datetime, timedelta
    from SHARED.runner_io import render_status_table

    recent_finished = (datetime.now() - timedelta(minutes=10)).isoformat()
    old_finished = (datetime.now() - timedelta(hours=3)).isoformat()

    # Case 1: recent killed run in the last hour triggers banner
    statuses_recent = [
        {"experiment_id": "exp1", "state": "done", "started_at": "2026-08-20T10:00:00", "duration_seconds": 20, "run_name": "run1"},
        {"experiment_id": "exp2", "state": "killed", "started_at": "2026-08-20T11:00:00", "finished_at": recent_finished, "duration_seconds": 35, "run_name": "run2"},
    ]
    render_status_table(statuses_recent)
    out = capsys.readouterr().out
    assert "STATE" in out
    assert "killed" in out
    assert "1 run(s) were stopped or killed in the last hour and may need re-launching" in out

    # Case 2: old killed run (> 1h ago) does NOT trigger banner
    statuses_old = [
        {"experiment_id": "exp1", "state": "done", "started_at": "2026-08-20T10:00:00", "duration_seconds": 20, "run_name": "run1"},
        {"experiment_id": "exp2", "state": "killed", "started_at": "2026-08-20T11:00:00", "finished_at": old_finished, "duration_seconds": 35, "run_name": "run2"},
    ]
    render_status_table(statuses_old)
    out_old = capsys.readouterr().out
    assert "STATE" in out_old
    assert "killed" in out_old
    assert "need re-launching" not in out_old





