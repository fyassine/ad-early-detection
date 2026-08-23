"""
SHARED/plotting.py — cross-package plot-footnote helpers.

Promotes the ``_add_note`` / ``_note_center_x`` pair inlined in
``CLASSIFIER/notebooks/SANITY/SANITY_SPLIT_HYGIENE_DELCODE.ipynb`` into a
public, reusable module so both CLASSIFIER and PROGNOSER notebooks can stamp
plots with a footnote naming the model run(s) that produced the plotted data.

This module deliberately has no model-specific imports — callers pass explicit
run-name strings and figure/axes objects.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping


def note_center_x(fig: Any, axes: Any) -> float:
    """x-center (figure coords) spanning ``axes``, for a note centered under a panel row."""
    import numpy as np

    ax_list = np.atleast_1d(axes).ravel().tolist()
    fig.canvas.draw()
    x0 = min(ax.get_position().x0 for ax in ax_list)
    x1 = max(ax.get_position().x1 for ax in ax_list)
    return (x0 + x1) / 2


def add_note(fig: Any, axes: Any, text: str, y: float = -0.03, **text_kwargs: Any) -> None:
    """Render an italic gray footnote centered under ``axes``, at height ``y`` (figure coords)."""
    x = note_center_x(fig, axes)
    style = dict(ha="center", fontsize=8, style="italic", color="gray")
    style.update(text_kwargs)
    fig.text(x, y, text, **style)


def format_model_runs_note(runs: Mapping[str, str], *, max_models: int | None = None) -> str:
    """Compose the standard "model runs used" footnote line.

    ``runs`` maps a model label (e.g. ``"GAAE"``, ``"GEC"``) to its run
    identifier, in the order they should be read. If ``max_models`` is set and
    ``runs`` has more entries, only the first ``max_models`` are listed and a
    trailing ``"+N more"`` summarizes the rest.
    """
    if not runs:
        raise ValueError("format_model_runs_note requires at least one model run")

    items = list(runs.items())
    if max_models is not None and len(items) > max_models:
        shown = items[:max_models]
        remainder = len(items) - max_models
        pairs = "; ".join(f"{tag}: {name}" for tag, name in shown)
        return f"Model runs used — {pairs}; +{remainder} more"

    pairs = "; ".join(f"{tag}: {name}" for tag, name in items)
    return f"Model runs used — {pairs}"


def append_note_line(existing_text: str, new_line: str) -> str:
    """Append ``new_line`` to ``existing_text`` as an additional line (never replace)."""
    return f"{existing_text}\n{new_line}" if existing_text else new_line


def run_name_from_checkpoint_path(ckpt_path: str | os.PathLike) -> str:
    """Resolve a run name from an encoder checkpoint path.

    Encoder checkpoints are saved as ``<run_dir>/model_<run_dir.name>.pth`` —
    the run name is the parent directory name.
    """
    return Path(ckpt_path).parent.name
