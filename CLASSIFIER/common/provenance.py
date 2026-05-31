"""
Run-provenance helpers shared by every training notebook.

A "run" produced by a training notebook should be reconstructable from its own
directory alone. These helpers standardise how that directory is named and what
it contains:

- ``region_from_data_root`` parses the DELCODE data directory name
  (``__fc_<region>_<atlas>_<variant>__``) so the brain region / atlas can be
  surfaced in the run name and stored in the config.
- ``make_run_dir`` builds a ``<model>_<region>_<timestamp>`` directory.
- ``snapshot_source`` copies the model's source files + a ``git_commit.txt`` into
  the run directory ("save code").
- ``save_full_checkpoint`` writes a self-describing checkpoint (state dict + RNG
  state + config) so a run can be reloaded and re-run flawlessly.
- ``write_run_summary`` / ``patch_run_summary`` centralise the JSON artifact that
  notebooks write then back-patch with test metrics.

This module deliberately has no model-specific imports — callers pass explicit
file paths and config dicts.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]  # .../ad-early-detection
_DATA_DIR_RE = re.compile(r"^__(?P<inner>.+)__$")


# --------------------------------------------------------------------------- #
# Region / dataset identity
# --------------------------------------------------------------------------- #
def region_from_data_root(data_root: str | os.PathLike) -> Dict[str, Optional[str]]:
    """Parse a DELCODE data directory name into its identity fields.

    The DELCODE convention encodes the input data in the directory name, e.g.
    ``__fc_wholebrain_sch200_flat__`` or
    ``__fc_dmn-hippo-limbic-dan_sch200-tian2_flat__``. ``data_root`` is usually
    the ``.../matrices`` subdirectory, so we scan the path for the ``__...__``
    component.

    Returns a dict with ``modality``, ``region``, ``atlas``, ``variant``,
    ``dataset_dir`` (the ``__...__`` name) and ``data_root`` (the input string).

    Raises ``ValueError`` if no ``__...__`` component is present — a silent
    fallback here would mislabel every run.
    """
    parts = Path(data_root).parts
    dataset_dir = next(
        (p for p in parts if _DATA_DIR_RE.match(p)),
        None,
    )
    if dataset_dir is None:
        raise ValueError(
            f"Could not find a '__<name>__' dataset directory in path {data_root!r}. "
            "Expected something like '__fc_wholebrain_sch200_flat__'."
        )
    inner = _DATA_DIR_RE.match(dataset_dir).group("inner")
    tokens = inner.split("_")
    return {
        "data_root": str(data_root),
        "dataset_dir": dataset_dir,
        "modality": tokens[0] if len(tokens) > 0 else None,
        "region": tokens[1] if len(tokens) > 1 else None,
        "atlas": tokens[2] if len(tokens) > 2 else None,
        "variant": tokens[3] if len(tokens) > 3 else None,
    }


def make_run_dir(
    output_root: str | os.PathLike,
    model_tag: str,
    region_info: Dict[str, Optional[str]] | str,
    *,
    timestamp: Optional[str] = None,
) -> tuple[str, Path]:
    """Create and return ``(run_name, run_dir)`` named ``<model>_<region>_<ts>``.

    ``region_info`` may be the dict from :func:`region_from_data_root` or a plain
    region string. The region is embedded in the run name so it is visible at a
    glance in directory listings.
    """
    region = region_info["region"] if isinstance(region_info, dict) else region_info
    region = region or "unknown-region"
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{model_tag}_{region}_{timestamp}"
    run_dir = Path(output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_name, run_dir


# --------------------------------------------------------------------------- #
# Code / environment provenance
# --------------------------------------------------------------------------- #
def capture_git_provenance(repo_root: str | os.PathLike = _REPO_ROOT) -> Dict[str, Any]:
    """Best-effort git commit / branch / dirty-flag capture.

    Never raises: a failed ``git`` call must not abort a long training run at
    save time. On failure the ``error`` key explains why.
    """
    def _git(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(
                ["git", "-C", str(repo_root), *args],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if out.returncode != 0:
                return None
            return out.stdout.strip()
        except Exception:
            return None

    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "branch": None, "dirty": None,
                "error": "git unavailable or not a repository"}
    status = _git("status", "--porcelain")
    return {
        "commit": commit,
        "short_commit": commit[:9],
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),  # uncommitted changes present
    }


def capture_env() -> Dict[str, Optional[str]]:
    """Capture interpreter + key library versions (best-effort)."""
    def _ver(mod: str) -> Optional[str]:
        try:
            return __import__(mod).__version__
        except Exception:
            return None

    return {
        "python": sys.version.split()[0],
        "torch": _ver("torch"),
        "torch_geometric": _ver("torch_geometric"),
        "numpy": _ver("numpy"),
        "sklearn": _ver("sklearn"),
    }


def snapshot_source(
    run_dir: str | os.PathLike,
    source_files: Sequence[str | os.PathLike],
    *,
    repo_root: str | os.PathLike = _REPO_ROOT,
) -> Dict[str, Any]:
    """Copy ``source_files`` into ``run_dir/source/`` and write ``git_commit.txt``.

    Files are mirrored under ``source/`` preserving their path relative to
    ``repo_root`` so the origin of each file is unambiguous. Missing files are
    recorded under ``missing`` rather than raising — a notebook should still save
    its run even if a path is stale.
    """
    run_dir = Path(run_dir)
    source_root = run_dir / "source"
    source_root.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    missing: list[str] = []
    for f in source_files:
        src = Path(f)
        if not src.is_file():
            missing.append(str(src))
            continue
        try:
            rel = src.resolve().relative_to(Path(repo_root).resolve())
        except ValueError:
            rel = Path(src.name)
        dest = source_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied.append(str(rel))

    git = capture_git_provenance(repo_root)
    (run_dir / "git_commit.txt").write_text(
        "\n".join(
            [
                f"commit: {git.get('commit')}",
                f"branch: {git.get('branch')}",
                f"dirty:  {git.get('dirty')}",
                f"captured_at: {datetime.now().isoformat(timespec='seconds')}",
            ]
        )
        + "\n"
    )

    manifest = {"copied": copied, "missing": missing, "git": git}
    (source_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


# --------------------------------------------------------------------------- #
# Run-summary JSON (write + back-patch)
# --------------------------------------------------------------------------- #
def _json_default(obj: Any) -> Any:
    """Serializer for the types notebooks routinely put in summaries."""
    import numpy as np  # local import keeps the module light

    if isinstance(obj, (datetime, Path)):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, set):
        return sorted(obj)
    # torch.device etc.
    return str(obj)


_SUMMARY_NAME = "run_summary.json"


def write_run_summary(run_dir: str | os.PathLike, summary: Dict[str, Any]) -> Path:
    """Write ``run_summary.json`` into ``run_dir`` (pretty-printed, type-safe)."""
    path = Path(run_dir) / _SUMMARY_NAME
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    return path


def patch_run_summary(run_dir: str | os.PathLike, updates: Dict[str, Any]) -> Path:
    """Merge ``updates`` into an existing ``run_summary.json`` (e.g. test metrics)."""
    path = Path(run_dir) / _SUMMARY_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist — call write_run_summary() before patching."
        )
    with open(path) as f:
        summary = json.load(f)
    summary.update(updates)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    return path


# --------------------------------------------------------------------------- #
# Full-state checkpoint
# --------------------------------------------------------------------------- #
def save_full_checkpoint(
    path: str | os.PathLike,
    *,
    model_state: Dict[str, Any],
    model_config: Dict[str, Any],
    training_config: Dict[str, Any],
    rng: Any = None,
    optimizer: Any = None,
    scheduler: Any = None,
    **extra: Any,
) -> Path:
    """Write a self-describing checkpoint for flawless reload + rerun.

    Stores the state dict (not a pickled module — robust to code moves), the
    architecture/training config needed to rebuild the model, and both RNG
    states. Optimizer/scheduler state are optional (sklearn models have none).
    Extra keyword args (``val_auc``, ``best_threshold``, ``epoch``, ...) are
    merged into the checkpoint.
    """
    import torch  # local import: keeps region/git helpers torch-free

    checkpoint: Dict[str, Any] = {
        "model_state_dict": model_state,
        "model_config": model_config,
        "training_config": training_config,
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "rng_state": rng.bit_generator.state if rng is not None else None,
        "torch_rng_state": torch.get_rng_state(),
        "env": capture_env(),
        "git": capture_git_provenance(),
        **extra,
    }
    path = Path(path)
    torch.save(checkpoint, str(path))
    return path
