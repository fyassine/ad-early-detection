"""Tests for CLASSIFIER.common.provenance."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from common.provenance import (
    capture_env,
    capture_git_provenance,
    make_run_dir,
    patch_run_summary,
    region_from_data_root,
    save_full_checkpoint,
    snapshot_source,
    snapshot_source_dirs,
    write_run_summary,
)


@pytest.mark.parametrize(
    "data_root,expected",
    [
        (
            "/data/DELCODE/__fc_wholebrain_sch200_flat__/matrices",
            ("fc", "wholebrain", "sch200", "flat"),
        ),
        (
            "/data/DELCODE/__fc_dmn-hippo-limbic-dan_sch200-tian2_flat__/matrices",
            ("fc", "dmn-hippo-limbic-dan", "sch200-tian2", "flat"),
        ),
        (
            "/data/DELCODE/__fc_hippo_tian2_flat__/metadata",
            ("fc", "hippo", "tian2", "flat"),
        ),
        (
            "/data/DELCODE/__fmri_wholebrain_sch200_session__",
            ("fmri", "wholebrain", "sch200", "session"),
        ),
    ],
)
def test_region_from_data_root_parses_identity(data_root, expected):
    info = region_from_data_root(data_root)
    assert (info["modality"], info["region"], info["atlas"], info["variant"]) == expected
    assert info["data_root"] == data_root


def test_region_from_data_root_raises_without_dataset_dir():
    with pytest.raises(ValueError):
        region_from_data_root("/data/DELCODE/some_plain_dir/matrices")


def test_make_run_dir_embeds_region(tmp_path):
    info = region_from_data_root(
        "/data/DELCODE/__fc_wholebrain_sch200_flat__/matrices"
    )
    run_name, run_dir = make_run_dir(tmp_path, "gelstm", info, timestamp="2026-05-28_00-00-00")
    assert run_name == "gelstm_wholebrain_2026-05-28_00-00-00"
    assert run_dir.is_dir()
    assert run_dir.name == run_name


def test_capture_git_provenance_keys():
    git = capture_git_provenance()
    # Always returns the same key set, never raises.
    assert set(git).issuperset({"commit", "branch", "dirty"})


def test_capture_env_has_python():
    env = capture_env()
    assert env["python"]


def test_snapshot_source_copies_files_and_writes_commit(tmp_path):
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    f = repo / "pkg" / "models.py"
    f.write_text("x = 1\n")
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    manifest = snapshot_source(run_dir, [f, repo / "pkg" / "missing.py"], repo_root=repo)

    assert "pkg/models.py" in manifest["copied"]
    assert any("missing.py" in m for m in manifest["missing"])
    assert (run_dir / "source" / "pkg" / "models.py").read_text() == "x = 1\n"
    assert (run_dir / "git_commit.txt").exists()
    assert json.loads((run_dir / "source" / "manifest.json").read_text())["copied"]


def test_run_summary_write_and_patch(tmp_path):
    summary = {
        "run_name": "gelstm_wholebrain_x",
        "cv_auc": np.float64(0.91),
        "dims": np.array([1, 2, 3]),
    }
    write_run_summary(tmp_path, summary)
    loaded = json.loads((tmp_path / "run_summary.json").read_text())
    assert loaded["cv_auc"] == pytest.approx(0.91)
    assert loaded["dims"] == [1, 2, 3]

    patch_run_summary(tmp_path, {"test_auc": 0.88})
    loaded = json.loads((tmp_path / "run_summary.json").read_text())
    assert loaded["test_auc"] == pytest.approx(0.88)
    assert loaded["run_name"] == "gelstm_wholebrain_x"


def test_patch_run_summary_requires_existing(tmp_path):
    with pytest.raises(FileNotFoundError):
        patch_run_summary(tmp_path, {"a": 1})


def test_save_full_checkpoint_roundtrip(tmp_path):
    import torch

    model = torch.nn.Linear(4, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(0)
    path = tmp_path / "checkpoint.pth"

    save_full_checkpoint(
        path,
        model_state=model.state_dict(),
        model_config={"in_features": 4, "out_features": 1},
        training_config={"lr": 1e-3},
        rng=rng,
        optimizer=optimizer,
        val_auc=0.9,
        best_threshold=0.5,
    )

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert ckpt["model_config"]["in_features"] == 4
    assert ckpt["training_config"]["lr"] == pytest.approx(1e-3)
    assert ckpt["optimizer_state_dict"] is not None
    assert ckpt["rng_state"] is not None
    assert ckpt["torch_rng_state"] is not None
    assert ckpt["val_auc"] == pytest.approx(0.9)

    # The state dict reloads into a freshly built model — flawless rerun.
    fresh = torch.nn.Linear(4, 1)
    fresh.load_state_dict(ckpt["model_state_dict"])


def test_snapshot_source_dirs(tmp_path):
    """Walks roots, copies only matching suffixes, skips excluded dirs, records missing."""
    repo = tmp_path / "repo"
    pkg = repo / "pkg"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "keep.py").write_text("x = 1\n")
    (pkg / "cfg.yaml").write_text("a: 1\n")
    (pkg / "data.npz").write_bytes(b"\x00\x01")          # excluded by suffix
    (pkg / "sub" / "deep.py").write_text("y = 2\n")
    cache = pkg / "__pycache__"
    cache.mkdir()
    (cache / "junk.py").write_text("garbage\n")           # excluded dir
    (repo / "lone.py").write_text("z = 3\n")              # a file root

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = snapshot_source_dirs(
        run_dir,
        ["pkg", "lone.py", "does_not_exist"],
        repo_root=repo,
    )

    copied = set(manifest["copied"])
    assert "pkg/keep.py" in copied
    assert "pkg/cfg.yaml" in copied
    assert "pkg/sub/deep.py" in copied
    assert "lone.py" in copied
    assert not any("data.npz" in c for c in copied)       # wrong suffix skipped
    assert not any("__pycache__" in c for c in copied)    # excluded dir skipped
    assert "does_not_exist" in manifest["missing"]

    # Files actually land under run_dir/source/ preserving repo-relative paths.
    assert (run_dir / "source" / "pkg" / "keep.py").is_file()
    assert (run_dir / "source" / "lone.py").is_file()
    assert (run_dir / "source" / "manifest.json").is_file()
    assert (run_dir / "git_commit.txt").is_file()
