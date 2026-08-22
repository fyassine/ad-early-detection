"""Cross-host run liveness: fritz and frieda share one outputs/ tree.

The failure these guard against is not hypothetical: before hosts.py, a
``--status`` on fritz checked a frieda run's pid against fritz's own process
table, found nothing, and rewrote that healthy run's status.json to "killed" on
shared disk.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from SHARED import hosts
from SHARED.runner_io import (
    active_runs,
    assert_not_already_running,
    reconcile_run_status,
)

REMOTE = "frieda" if hosts.local_host() != "frieda" else "fritz"


def _make_run(root, exp_id, *, host, beat_age_seconds=None, state="running", pid=999_999):
    """A run directory with a status.json and optionally a heartbeat of some age."""
    run_dir = root / exp_id / "runs" / f"{exp_id}-run"
    run_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text(
        json.dumps(
            {
                "experiment_id": exp_id,
                "run_name": f"{exp_id}-run",
                "state": state,
                "pid": pid,
                "host": host,
                "started_at": (datetime.now() - timedelta(minutes=5)).isoformat(),
            }
        )
    )
    if beat_age_seconds is not None:
        hosts.write_heartbeat(run_dir, datetime.now() - timedelta(seconds=beat_age_seconds))
    return run_dir


def test_normalise_strips_case_and_domain():
    assert hosts._normalise("FRiEDA.local") == "frieda"
    assert hosts._normalise("FRiTZ") == "fritz"
    assert hosts._normalise(None) == ""


def test_local_host_is_a_known_training_host():
    assert hosts.local_host() in hosts.TRAINING_HOSTS
    assert hosts.is_local(hosts.local_host().upper())
    assert not hosts.is_local(REMOTE)


def test_heartbeat_roundtrip(tmp_path):
    stamp = datetime.now().replace(microsecond=0)
    hosts.write_heartbeat(tmp_path, stamp)
    assert hosts.read_heartbeat(tmp_path) == stamp


def test_heartbeat_liveness_states(tmp_path):
    assert hosts.heartbeat_liveness(tmp_path) == "unknown"  # never written
    hosts.write_heartbeat(tmp_path, datetime.now())
    assert hosts.heartbeat_liveness(tmp_path) == "alive"
    hosts.write_heartbeat(tmp_path, datetime.now() - timedelta(seconds=10_000))
    assert hosts.heartbeat_liveness(tmp_path) == "dead"


def test_remote_run_with_fresh_heartbeat_survives_reconcile(tmp_path):
    """The regression: reconciling on THIS box must not kill the other box's run."""
    run_dir = _make_run(tmp_path, "remote-live", host=REMOTE, beat_age_seconds=5)
    out = reconcile_run_status(run_dir / "status.json", write_disk=True)
    assert out["state"] == "running"
    assert json.loads((run_dir / "status.json").read_text())["state"] == "running"


def test_remote_run_with_stale_heartbeat_is_reaped(tmp_path):
    run_dir = _make_run(tmp_path, "remote-stale", host=REMOTE, beat_age_seconds=10_000)
    out = reconcile_run_status(run_dir / "status.json", write_disk=True)
    assert out["state"] == "killed"


def test_local_run_still_uses_the_process_table(tmp_path):
    """A local run with a dead pid is reaped without consulting any heartbeat."""
    run_dir = _make_run(tmp_path, "local-dead", host=hosts.local_host(), pid=999_999)
    assert reconcile_run_status(run_dir / "status.json", write_disk=True)["state"] == "killed"


def test_active_runs_reports_remote_and_skips_dead(tmp_path):
    _make_run(tmp_path, "live", host=REMOTE, beat_age_seconds=5)
    _make_run(tmp_path, "stale", host=REMOTE, beat_age_seconds=10_000)
    _make_run(tmp_path, "finished", host=REMOTE, beat_age_seconds=5, state="done")
    found = {s["experiment_id"] for s in active_runs(tmp_path)}
    assert found == {"live"}


def test_guard_refuses_id_running_on_the_other_box(tmp_path):
    _make_run(tmp_path, "busy", host=REMOTE, beat_age_seconds=5)
    with pytest.raises(RuntimeError, match="already running"):
        assert_not_already_running(tmp_path, "busy")


def test_guard_refuses_when_liveness_is_unknown(tmp_path):
    """No heartbeat yet is not evidence of death -- refuse rather than race."""
    _make_run(tmp_path, "nobeat", host=REMOTE, beat_age_seconds=None)
    with pytest.raises(RuntimeError, match="no heartbeat yet"):
        assert_not_already_running(tmp_path, "nobeat")


def test_guard_allows_a_free_id(tmp_path):
    _make_run(tmp_path, "other", host=REMOTE, beat_age_seconds=5)
    assert_not_already_running(tmp_path, "not-running-anywhere")


def test_assign_hosts_small_batch_goes_to_the_freer_box():
    ranked = ["frieda", "fritz"]
    assert hosts.assign_hosts(["a"], ranked=ranked) == [("frieda", "a")]
    assert hosts.assign_hosts(["a", "b"], ranked=ranked) == [("frieda", "a"), ("frieda", "b")]


def test_assign_hosts_splits_across_both_above_the_threshold():
    ranked = ["frieda", "fritz"]
    assert hosts.assign_hosts(["a", "b", "c"], ranked=ranked) == [
        ("frieda", "a"),
        ("fritz", "b"),
        ("frieda", "c"),
    ]


def test_assign_hosts_degrades_to_a_single_reachable_box():
    assert hosts.assign_hosts(["a", "b", "c"], ranked=["fritz"]) == [
        ("fritz", "a"),
        ("fritz", "b"),
        ("fritz", "c"),
    ]


def test_assign_hosts_empty_and_unreachable():
    assert hosts.assign_hosts([], ranked=["fritz"]) == []
    with pytest.raises(RuntimeError, match="No training host"):
        hosts.assign_hosts(["a"], ranked=[])


def test_run_on_rejects_a_host_outside_the_inventory():
    """No caller can turn a stray field or argument into an ssh target."""
    with pytest.raises(ValueError, match="unknown training host"):
        hosts.run_on("evil.example.com", "echo hi")


def test_gpu_free_mib_raises_for_a_host_outside_the_inventory():
    """An unknown host is a caller bug, not a transient outage: fail loudly.

    None is reserved for "this known box could not be queried right now", which
    is what rank_by_free_gpu uses to drop an unreachable box.
    """
    with pytest.raises(ValueError, match="unknown training host"):
        hosts.gpu_free_mib("not-a-box")


def test_gpu_snapshot_shape_for_a_real_box():
    """Skips rather than fails when the box is momentarily unreachable."""
    snap = hosts.gpu_snapshot(hosts.local_host())
    if snap is None:
        pytest.skip("local GPU could not be queried (no nvidia-smi?)")
    assert snap["host"] == hosts.local_host()
    assert snap["free_mib"] == snap["total_mib"] - snap["used_mib"]
    assert 0 <= snap["util_pct"] <= 100


def test_gpu_snapshot_rejects_a_host_outside_the_inventory():
    with pytest.raises(ValueError, match="unknown training host"):
        hosts.gpu_snapshot("not-a-box")
