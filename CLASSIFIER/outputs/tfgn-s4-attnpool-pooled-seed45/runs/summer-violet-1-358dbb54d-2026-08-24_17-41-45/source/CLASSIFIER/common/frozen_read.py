"""
common/frozen_read.py — the one frozen read of a held-out split for a saved run.

``DOCS/temporal-first-ablation.md``'s 2026-08-24 addendum (§4): the in-domain
test set and OASIS-3 are each read exactly once, after the ladder is frozen —
never during a ladder run. Every ladder arm now sets ``defer_test_eval: true``
(see ``LONGITUDINAL_COMMON_DELCODE.ipynb``'s Configuration cell), so its
``run_summary.json`` carries no ``test_*`` / ``ext_*`` keys. This module is
the one place that read happens, called from the comparison notebook on the
frozen winner(s) only: it reconstructs a run's adapter from its own saved
``run_summary.json`` + checkpoint (no retraining — ``adapter.load_state``),
scores a given dataframe at the run's own OOF-derived threshold
(``adapters.read_run_threshold`` — never re-optimized here, see
``.claude/rules/evaluation.md``), and records the result with the exact same
``common.run_artifacts.record_test_metrics`` / ``record_external_metrics``
every non-deferred run already used, so the schema and ``RESULTS.csv``
columns are identical whether the read happened inline or here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from adapters import get_adapter, read_run_threshold
from common.run_artifacts import record_external_metrics, record_test_metrics


def load_run_summary(run_dir: str | Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    path = run_dir / "run_summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"No run_summary.json in {run_dir}.")
    return json.loads(path.read_text())


def build_adapter_from_run(
    run_dir: str | Path,
    *,
    adapter_key: str,
    data_root: str,
    cohorts_csv: Optional[str],
    gaae_ckpt_path: str,
    gaae_hp: Dict[str, Any],
    device: Any,
    rng: Any = None,
):
    """Reconstruct a run's (untrained-shell) adapter from its saved config.

    Uses the run's own ``training_config`` — the exact merged config the
    original notebook trained with — so every ladder knob (``node_lstm_init``,
    ``recon_target``, ``fusion``, ``feature_set``, ``min_visits``, ...) is
    reproduced exactly. The caller supplies the path-resolution pieces a run
    does not persist itself (``data_root`` / ``cohorts_csv`` /
    ``gaae_ckpt_path`` / ``gaae_hp``), mirroring
    ``LONGITUDINAL_COMMON_DELCODE.ipynb``'s own adapter-construction cell.
    Returns ``(adapter, run_summary)``.
    """
    summary = load_run_summary(run_dir)
    train_config = dict(summary["training_config"])
    adapter_cls = get_adapter(adapter_key)
    adapter = adapter_cls(
        gaae_ckpt_path=gaae_ckpt_path,
        gaae_hp=gaae_hp,
        train_config=train_config,
        data_root=data_root,
        cohorts_csv=cohorts_csv,
        device=device,
        rng=rng,
    )
    return adapter, summary


def score_frozen_split(
    run_dir: str | Path,
    df: Any,
    *,
    adapter_key: str,
    data_root: str,
    cohorts_csv: Optional[str],
    gaae_ckpt_path: str,
    gaae_hp: Dict[str, Any],
    device: Any,
    record_as: str,
    cohort: Optional[str] = None,
) -> Dict[str, Any]:
    """Score ``df`` once against a saved run's frozen checkpoint and record it.

    ``record_as`` is ``"test"`` (records via ``record_test_metrics``) or
    ``"external"`` (``record_external_metrics``, requires ``cohort``, e.g.
    ``"oasis3"``). Returns the adapter's ``eval_split`` result dict.
    """
    if record_as not in ("test", "external"):
        raise ValueError(f"record_as must be 'test' or 'external', got {record_as!r}.")
    if record_as == "external" and not cohort:
        raise ValueError("record_as='external' requires cohort=.")

    adapter, summary = build_adapter_from_run(
        run_dir,
        adapter_key=adapter_key,
        data_root=data_root,
        cohorts_csv=cohorts_csv,
        gaae_ckpt_path=gaae_ckpt_path,
        gaae_hp=gaae_hp,
        device=device,
    )
    state = adapter.load_state(run_dir)
    threshold = read_run_threshold(run_dir)
    bundle = adapter.prepare_data(df)
    metrics = adapter.eval_split(state, bundle, threshold, device=device)

    threshold_method = summary.get("threshold_method", "oof_f1")
    if record_as == "test":
        record_test_metrics(run_dir, metrics, threshold=threshold, threshold_method=threshold_method)
    else:
        record_external_metrics(
            run_dir, metrics, threshold=threshold, threshold_method=threshold_method, cohort=cohort,
        )
    return metrics


__all__ = ["load_run_summary", "build_adapter_from_run", "score_frozen_split"]
