"""
common/run_artifacts.py — high-level run-saving wrappers over ``common.provenance``.

The longitudinal notebooks all save a run the same way: create the run dir, dump a
back-compat ``model_<run>.pth`` state dict, write a full-state checkpoint, snapshot
the source files, and write ``run_summary.json`` — then later patch in the test
metrics. This module bundles that fixed sequence so a notebook calls one function
instead of ~50 lines of identical boilerplate.

``provenance`` stays the home of the low-level primitives; this module only
orchestrates them. (Named distinctly from the unrelated ``runner_io.py``, which is
terminal-output niceties.) The only model-specific inputs are ``model_config`` and
``source_files``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

from .provenance import (
    make_run_dir,
    snapshot_source,
    save_full_checkpoint,
    write_run_summary,
    patch_run_summary,
    capture_env,
    capture_git_provenance,
)


def save_run(
    *,
    output_dir: str,
    run_dir: str | None,
    run_name: str | None,
    model_state: Dict[str, Any],
    model_config: Dict[str, Any],
    training_config: Dict[str, Any],
    data_info: Dict[str, Any],
    dataset_info: Dict[str, Any],
    rng: Any,
    best_val_auc: float,
    active_threshold: float,
    threshold_method: str,
    best_fold: int,
    cv_results: Dict[str, Any],
    gaae_checkpoint: str,
    gaae_run_name: str,
    source_files: Sequence[str | Path],
    n_folds: int,
    model_tag: str = "sample",
) -> Tuple[str, Path]:
    """Persist a completed run and return ``(run_name, run_dir)``.

    If ``run_dir`` is provided (experiment-runner path) it is used as-is; otherwise
    a timestamped ``make_run_dir`` is created under ``output_dir`` with the region
    embedded in the name. Writes:

      * ``model_<run_name>.pth``      — back-compat plain state dict.
      * ``checkpoint_<run_name>.pth`` — full-state checkpoint (state + RNG + config).
      * ``source/`` + ``git_commit.txt`` — code snapshot of ``source_files``.
      * ``run_summary.json``          — the run ledger (test metrics patched later
                                        via :func:`record_test_metrics`).
    """
    import torch  # local import keeps the module importable without torch loaded eagerly

    if run_dir:
        run_dir_path = Path(run_dir)
        run_dir_path.mkdir(parents=True, exist_ok=True)
        run_name = run_name or run_dir_path.name
    else:
        run_name, run_dir_path = make_run_dir(output_dir, model_tag, data_info)

    # Back-compat artifact: plain state_dict (loaded directly by comparison code).
    torch.save(model_state, run_dir_path / f"model_{run_name}.pth")

    save_full_checkpoint(
        run_dir_path / f"checkpoint_{run_name}.pth",
        model_state=model_state,
        model_config=model_config,
        training_config=training_config,
        rng=rng,
        val_auc=float(best_val_auc),
        best_threshold=float(active_threshold),
        threshold_method=threshold_method,
        best_fold=int(best_fold),
        gaae_checkpoint=gaae_checkpoint,
        run_name=run_name,
    )

    snapshot_source(run_dir_path, list(source_files))

    run_summary = {
        "run_name": run_name,
        "data_info": data_info,
        "dataset_info": dataset_info,
        "model_config": model_config,
        "training_config": training_config,
        "gaae_checkpoint": gaae_checkpoint,
        "gaae_run_name": gaae_run_name,
        "n_folds": n_folds,
        "best_fold": int(best_fold),
        "best_val_auc": float(best_val_auc),
        "active_threshold": float(active_threshold),
        "threshold_method": threshold_method,
        "cv_results": cv_results,
        "env": capture_env(),
        "git": capture_git_provenance(),
    }
    write_run_summary(run_dir_path, run_summary)
    return run_name, run_dir_path


def record_test_metrics(
    run_dir: str | Path,
    metrics: Dict[str, Any],
    *,
    threshold: float,
    threshold_method: str,
) -> Path:
    """Patch ``run_summary.json`` with the standard test-metric schema.

    ``metrics`` is the dict returned by a model's ``eval_split`` hook
    (``auc``, ``sensitivity``, ``specificity``, ``f1``, ``probs``, ``targets``).
    """
    return patch_run_summary(
        run_dir,
        {
            "metrics": {
                "test_auc": float(metrics["auc"]),
                "test_f1": float(metrics["f1"]),
                "test_sensitivity": float(metrics["sensitivity"]),
                "test_specificity": float(metrics["specificity"]),
                "threshold": float(threshold),
                "threshold_method": threshold_method,
            },
            "test_auc": float(metrics["auc"]),
            "test_sensitivity": float(metrics["sensitivity"]),
            "test_specificity": float(metrics["specificity"]),
            "test_f1": float(metrics["f1"]),
            "test_probabilities": [float(p) for p in metrics["probs"]],
            "test_labels": [int(t) for t in metrics["targets"]],
        },
    )
