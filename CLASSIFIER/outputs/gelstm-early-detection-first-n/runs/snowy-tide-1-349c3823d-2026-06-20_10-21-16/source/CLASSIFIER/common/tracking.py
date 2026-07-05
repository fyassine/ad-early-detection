"""
Single Weights & Biases entry point shared by every CLASSIFIER model.

Why this module exists
----------------------
Before this, GAAE called ``wandb.init`` with hardcoded args inside its training
loop while GEC used a ``wandb_run=None`` hook and GELSTM logged nothing. That is
three different conventions and no consistent run naming. ``tracking.init_run``
centralises *one* convention so every run lands in the same project with the
same group / job_type / tags (see CLASSIFIER/README and the experiment-runner
plan for the scheme).

Design contract
---------------
- ``init_run`` ALWAYS returns an object exposing ``.log(dict, step=None)``,
  ``.finish()``, ``.config`` and ``.name`` — either a real ``wandb.Run`` or the
  no-op :class:`_NoOpRun`. Callers never branch on ``None``.
- It NEVER raises and NEVER blocks: a missing key, missing ``wandb`` install, or
  a dead network downgrades to offline (or no-op) with a loud warning rather than
  aborting a long training run. This mirrors the fail-soft stance of
  ``provenance.capture_git_provenance``.
- W&B is ON by default. It is disabled only when ``WANDB_MODE=disabled`` or the
  experiment entry sets ``wandb: false``.

Mode resolution (env ``WANDB_MODE`` wins, then experiment, then default):
    disabled            -> no-op stub (no run created)
    offline             -> local run dir, syncable later via ``wandb sync``
    online (default)    -> live logging; auto-falls-back to offline if no
                           credentials or ``wandb.init`` fails.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

from .provenance import capture_git_provenance

_DEFAULT_PROJECT = "ad-early-detection"


class _NoOpRun:
    """Stand-in run with the wandb.Run surface the notebooks use.

    Returned whenever logging is disabled or unavailable so notebook/training
    code can call ``run.log(...)`` / ``run.finish()`` unconditionally.
    """

    def __init__(self, name: str = "wandb-disabled", config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = dict(config or {})
        self.id = None
        self.url = None
        self.disabled = True

    def log(self, *_args: Any, **_kwargs: Any) -> None:  # noqa: D401 - no-op
        return None

    def finish(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def __repr__(self) -> str:
        return f"_NoOpRun(name={self.name!r})"


def _credentials_available() -> bool:
    """True if wandb can authenticate non-interactively (env key or netrc).

    Without this guard ``wandb.init`` in online mode would PROMPT on stdin and
    hang a headless papermill run forever.
    """
    if os.environ.get("WANDB_API_KEY"):
        return True
    netrc = Path.home() / ".netrc"
    if netrc.is_file() and "api.wandb.ai" in netrc.read_text(errors="ignore"):
        return True
    return False


def _resolve_mode(exp: Dict[str, Any]) -> str:
    """Resolve the effective W&B mode: disabled | offline | online."""
    if exp.get("wandb") is False:
        return "disabled"
    env_mode = os.environ.get("WANDB_MODE", "").strip().lower()
    if env_mode in {"disabled", "offline", "online"}:
        return env_mode
    return "online"  # default: log everything


def _region_tag(exp: Dict[str, Any], params: Dict[str, Any]) -> Optional[str]:
    """Best-effort brain-region tag from params/dataset (never raises)."""
    region = params.get("REGION") or params.get("region")
    if region:
        return str(region)
    data_root = params.get("DATA_ROOT") or params.get("data_root")
    if data_root:
        try:
            from .provenance import region_from_data_root

            return region_from_data_root(data_root).get("region")
        except Exception:
            return None
    return None


def _build_init_kwargs(exp: Dict[str, Any], params: Dict[str, Any], fold: Optional[int]) -> Dict[str, Any]:
    """Assemble the wandb.init kwargs implementing the naming convention."""
    git = capture_git_provenance()

    # W&B run name = the experiment id (the human-readable name given in the
    # registry), so runs are easy to tell apart in the UI. The commit lives in
    # the run config and the timestamp in the local run_dir; W&B shows its own
    # created-at column. Folds get a ``-fold{k}`` suffix so they stay distinct
    # within the experiment's group.
    base_name = exp.get("id") or params.get("RUN_NAME") or "run"
    name = f"{base_name}-fold{fold}" if fold is not None else base_name

    region = _region_tag(exp, params)
    tags = [
        str(exp.get("mode", "")),
        str(exp.get("model", "")),
        str(exp.get("dataset", "")),
        f"seed={exp.get('seed', params.get('SEED'))}",
    ]
    if region:
        tags.append(str(region))
    tags = [t for t in tags if t and not t.endswith("=None")]

    config = {**params, "experiment_id": exp.get("id"), "git_commit": git.get("commit"), "fold": fold}

    return {
        "entity": os.environ.get("WANDB_ENTITY") or None,
        "project": os.environ.get("WANDB_PROJECT") or _DEFAULT_PROJECT,
        "group": exp.get("id"),
        "job_type": exp.get("model"),
        "name": name,
        "tags": tags,
        "config": config,
        "notes": exp.get("notes"),
    }


def init_run(exp: Dict[str, Any], params: Dict[str, Any], *, fold: Optional[int] = None):
    """Initialise (or stub) a W&B run for ``exp``.

    Parameters
    ----------
    exp : dict
        Experiment entry from ``experiments.yaml`` (needs at least ``id``,
        ``model``; ``mode``/``dataset``/``seed``/``notes`` are used for tags).
    params : dict
        The merged hyperparameter dict; logged verbatim as the run config.
    fold : int, optional
        Cross-validation fold index; appended to the run name when given so
        folds of one experiment cluster under the same ``group``.

    Returns
    -------
    A ``wandb.Run`` or :class:`_NoOpRun` — always safe to ``.log()``/``.finish()``.
    """
    mode = _resolve_mode(exp)
    if mode == "disabled":
        return _NoOpRun(config=params)

    try:
        import wandb
    except Exception as exc:  # wandb not installed in this env
        warnings.warn(f"[tracking] wandb unavailable ({exc!r}); logging disabled for this run.")
        return _NoOpRun(config=params)

    if mode == "online" and not _credentials_available():
        warnings.warn(
            "[tracking] WANDB_MODE=online but no WANDB_API_KEY/.netrc credentials found; "
            "falling back to offline mode (sync later with `wandb sync`)."
        )
        mode = "offline"

    init_kwargs = _build_init_kwargs(exp, params, fold)
    # wandb writes its startup banner to stderr the instant init() runs. Without
    # this newline it glues onto whatever was last on the line — typically a tqdm
    # progress bar or a print() without a trailing newline. Emit one blank line on
    # the same stream so the banner always starts fresh.
    print(file=sys.stderr, flush=True)
    try:
        return wandb.init(mode=mode, reinit=True, **init_kwargs)
    except Exception as exc:
        if mode != "offline":
            warnings.warn(f"[tracking] wandb.init failed ({exc!r}); retrying in offline mode.")
            try:
                return wandb.init(mode="offline", reinit=True, **init_kwargs)
            except Exception as exc2:
                warnings.warn(f"[tracking] offline wandb.init also failed ({exc2!r}); logging disabled.")
        else:
            warnings.warn(f"[tracking] offline wandb.init failed ({exc!r}); logging disabled.")
        return _NoOpRun(config=params)


def log_metrics(run: Any, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """Log ``metrics`` to ``run`` (real or stub), swallowing logging errors."""
    if run is None:
        return
    try:
        run.log(metrics, step=step) if step is not None else run.log(metrics)
    except Exception as exc:
        warnings.warn(f"[tracking] run.log failed ({exc!r}); continuing.")


def finish_run(run: Any) -> None:
    """Finish ``run`` (real or stub), swallowing errors."""
    if run is None:
        return
    try:
        run.finish()
    except Exception as exc:
        warnings.warn(f"[tracking] run.finish failed ({exc!r}); continuing.")
