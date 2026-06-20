#!/usr/bin/env python3
"""
Assign a stable ``id`` to every notebook cell that lacks one.

papermill (via ``nbformat.validate``) emits ``MissingIDFieldWarning`` for cells
authored before the nbformat 4.5 cell-id requirement; a future nbformat will make
this a hard error. nbformat 5.10 has no top-level ``normalize`` helper, so we
assign ids ourselves (``uuid4().hex[:8]``, matching ``new_code_cell``'s scheme).

Idempotent: cells that already have an id are left untouched. Run once:

    python CLASSIFIER/dev/normalize_notebook_ids.py
"""
from __future__ import annotations

import glob
import json
import sys
import uuid
import warnings
from pathlib import Path

import nbformat

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GLOBS = [
    "CLASSIFIER/notebooks/**/*.ipynb",
    "PROGNOSER/notebooks/*.ipynb",
]


def ensure_cell_ids(nb) -> int:
    """Give every id-less cell a fresh id. Returns the number of cells fixed.

    Defensive against a future nbformat that stops auto-filling ids on read; can
    also be imported by the notebook-wiring scripts before they write.
    """
    fixed = 0
    for cell in nb.cells:
        if not cell.get("id"):
            cell["id"] = uuid.uuid4().hex[:8]
            fixed += 1
    return fixed


def _count_missing_on_disk(path: str) -> int:
    """Count cells with no ``id`` in the raw JSON (nbformat.read auto-fills them,
    so we must inspect the file itself to know what is actually persisted)."""
    data = json.loads(Path(path).read_text())
    return sum(1 for c in data.get("cells", []) if not c.get("id"))


def main() -> int:
    total_fixed = 0
    for pattern in _GLOBS:
        for path in sorted(glob.glob(str(_REPO_ROOT / pattern), recursive=True)):
            missing = _count_missing_on_disk(path)
            if not missing:
                print(f"[normalize] {Path(path).name}: already clean.")
                continue
            # nbformat.read fills ids in memory (with a warning); ensure_cell_ids
            # covers any it misses; nbformat.write persists them to disk.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                nb = nbformat.read(path, as_version=4)
            ensure_cell_ids(nb)
            nbformat.validate(nb)
            nbformat.write(nb, path)
            total_fixed += missing
            print(f"[normalize] {Path(path).name}: assigned {missing} cell id(s).")
    print(f"[normalize] done — {total_fixed} cell id(s) assigned in total.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
