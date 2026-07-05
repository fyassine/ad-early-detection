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
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

from .provenance import capture_git_provenance

_DEFAULT_PROJECT = "ad-early-detection"

# Matches the local run_dir/notebook name the runner generates (run_naming.py +
# run_experiment.py::run_one): "<adjective>-<noun>-<n>-<gitsha-or-nogit>-<timestamp>".
# Used to lift the random display name + timestamp back out so the W&B run name
# can splice in the experiment id in place of the git hash.
_LOCAL_RUN_NAME_RE = re.compile(
    r"^(?P<display>.+)-(?P<git>[0-9a-f]+|nogit)-(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$"
)


def _split_local_run_name(run_name: Optional[str]) -> Optional[tuple[str, str]]:
    """Split a local run name into (display_name, timestamp), or None if it doesn't match."""
    if not run_name:
        return None
    m = _LOCAL_RUN_NAME_RE.match(run_name)
    if not m:
        return None
    return m.group("display"), m.group("timestamp")


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

    # W&B run name = "<local-display-name>-<experiment-id>-<timestamp>", e.g.
    # "classic-wind-17-gec-trajectory-whole-brain-2026-06-23_21-13-30". The
    # display name + timestamp are lifted from the local run_dir/notebook name
    # (params["RUN_NAME"], set by run_experiment.py) so a run is identifiable
    # from either W&B or the local outputs/ tree at a glance. Falls back to the
    # bare experiment id if RUN_NAME isn't available (e.g. interactive notebook
    # use outside the runner). Folds get a ``-fold{k}`` suffix.
    base_name = exp.get("id") or params.get("RUN_NAME") or "run"
    local_parts = _split_local_run_name(params.get("RUN_NAME"))
    if local_parts:
        display_name, timestamp = local_parts
        name = f"{display_name}-{base_name}-{timestamp}"
    else:
        name = base_name
    if fold is not None:
        name = f"{name}-fold{fold}"

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
        Experiment entry from ``the experiments/ directory`` (needs at least ``id``,
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
        warnings.warn(f"[tracking] wandb unavailable ({exc!r}); logging disabled for this run.", stacklevel=2)
        return _NoOpRun(config=params)

    if mode == "online" and not _credentials_available():
        warnings.warn(
            "[tracking] WANDB_MODE=online but no WANDB_API_KEY/.netrc credentials found; "
            "falling back to offline mode (sync later with `wandb sync`).", stacklevel=2
        )
        mode = "offline"

    init_kwargs = _build_init_kwargs(exp, params, fold)
    # wandb writes its startup banner to stderr the instant init() runs. Without
    # this newline it glues onto whatever was last on the line — typically a tqdm
    # progress bar or a print() without a trailing newline. Emit one blank line on
    # the same stream so the banner always starts fresh.
    print(file=sys.stderr, flush=True)
    try:
        return wandb.init(mode=mode, reinit=True, **init_kwargs)  # type: ignore[arg-type]
    except Exception as exc:
        if mode != "offline":
            warnings.warn(f"[tracking] wandb.init failed ({exc!r}); retrying in offline mode.", stacklevel=2)
            try:
                return wandb.init(mode="offline", reinit=True, **init_kwargs)
            except Exception as exc2:
                warnings.warn(f"[tracking] offline wandb.init also failed ({exc2!r}); logging disabled.", stacklevel=2)
        else:
            warnings.warn(f"[tracking] offline wandb.init failed ({exc!r}); logging disabled.", stacklevel=2)
        return _NoOpRun(config=params)


def log_metrics(run: Any, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """Log ``metrics`` to ``run`` (real or stub), swallowing logging errors."""
    if run is None:
        return
    try:
        run.log(metrics, step=step) if step is not None else run.log(metrics)
    except Exception as exc:
        warnings.warn(f"[tracking] run.log failed ({exc!r}); continuing.", stacklevel=2)


def finish_run(run: Any) -> None:
    """Finish ``run`` (real or stub), swallowing errors."""
    if run is None:
        return
    try:
        run.finish()
    except Exception as exc:
        warnings.warn(f"[tracking] run.finish failed ({exc!r}); continuing.", stacklevel=2)
