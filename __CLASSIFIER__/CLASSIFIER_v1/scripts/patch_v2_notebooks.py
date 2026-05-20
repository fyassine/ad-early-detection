"""
One-shot patcher: turns copied production notebooks in
CLASSIFIER/notebooks/ into v2 notebooks.

Applies the following idempotent edits to each target .ipynb:
    * sys.path injection: '.../CLASSIFIER' → '.../CLASSIFIER'
    * Inserts a leading "Task framing" markdown cell.
    * Inserts a "Split-hygiene audit" code cell right after the path-setup cell.
    * Per-notebook extras: subject-level rollup cells appended at the end of
      scan-level notebooks (GAAE_LOGREG_CLASSIFIER).

The patcher is idempotent: re-running it does not duplicate insertions.

Run with:  python CLASSIFIER/scripts/patch_v2_notebooks.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

V2_NB = Path(__file__).resolve().parents[1] / "notebooks"

SANITY_CELL_TAG = "__v2_sanity_audit__"
FRAMING_CELL_TAG = "__v2_task_framing__"
SUBJECT_ROLLUP_TAG = "__v2_subject_rollup__"
TASK_COLUMN_TAG = "__v2_task_column__"


def _make_code_cell(source_lines, tag=None):
    cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": [tag]} if tag else {},
        "outputs": [],
        "source": source_lines,
    }
    return cell


def _make_md_cell(source_lines, tag=None):
    return {
        "cell_type": "markdown",
        "metadata": {"tags": [tag]} if tag else {},
        "source": source_lines,
    }


def _has_tag(nb, tag):
    return any(tag in (c.get("metadata", {}).get("tags") or []) for c in nb["cells"])


def _patch_sys_path(nb):
    """Replace any 'CLASSIFIER' path in sys.path setup cells with 'CLASSIFIER'."""
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        new_src = []
        changed = False
        for line in cell["source"]:
            new_line = line.replace(
                "ad-early-detection/CLASSIFIER'",
                "ad-early-detection/CLASSIFIER'",
            ).replace(
                "ad-early-detection/CLASSIFIER\"",
                "ad-early-detection/CLASSIFIER\"",
            )
            if new_line != line:
                changed = True
            new_src.append(new_line)
        if changed:
            cell["source"] = new_src


def _insert_framing(nb, framing_text):
    if _has_tag(nb, FRAMING_CELL_TAG):
        return
    cell = _make_md_cell(framing_text, tag=FRAMING_CELL_TAG)
    nb["cells"].insert(0, cell)


def _insert_sanity_call(nb, split_dir_subpath):
    if _has_tag(nb, SANITY_CELL_TAG):
        return
    src = [
        "# v2 split-hygiene audit — hard-fails if any subject crosses splits.\n",
        "import sys\n",
        "from pathlib import Path\n",
        "_V2_ROOT = Path('/mnt/e/fyassine/ad-early-detection/CLASSIFIER')\n",
        "if str(_V2_ROOT) not in sys.path:\n",
        "    sys.path.insert(0, str(_V2_ROOT))\n",
        "from common.sanity import run_full_audit\n",
        "if 'METADATA_DIR' in globals():\n",
        f"    _splits_dir = Path(METADATA_DIR) / '{split_dir_subpath}'\n",
        "    _ = run_full_audit({\n",
        "        'train': str(_splits_dir / 'train.csv'),\n",
        "        'val':   str(_splits_dir / 'val.csv'),\n",
        "        'test':  str(_splits_dir / 'test.csv'),\n",
        "    })\n",
        "else:\n",
        "    print('[SANITY] METADATA_DIR not defined in this notebook — skipping split audit')\n",
    ]
    cell = _make_code_cell(src, tag=SANITY_CELL_TAG)
    # Prefer to insert after the cell that defines METADATA_DIR — that's where
    # the audit can actually run. Fall back to right after the sys.path cell.
    insert_idx = None
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") != "code":
            continue
        if any("METADATA_DIR" in line and "=" in line for line in c.get("source", [])):
            insert_idx = i + 1
            break
    if insert_idx is None:
        for i, c in enumerate(nb["cells"]):
            if c.get("cell_type") == "code" and any(
                "sys.path" in line and "CLASSIFIER" in line for line in c.get("source", [])
            ):
                insert_idx = i + 1
                break
    if insert_idx is None:
        insert_idx = 1
    nb["cells"].insert(insert_idx, cell)


def _append_subject_rollup(nb):
    if _has_tag(nb, SUBJECT_ROLLUP_TAG):
        return
    md = _make_md_cell(
        [
            "## Subject-level rollup (v2)\n",
            "\n",
            "The CV/test AUC reported above is **scan-level** — each subject contributes\n",
            "one row per scan. A subject with N scans is N× more influential than a\n",
            "single-visit subject. The cell below collapses scan-level predictions to\n",
            "one row per subject and re-computes AUC.\n",
        ],
        tag=SUBJECT_ROLLUP_TAG,
    )
    code = _make_code_cell(
        [
            "# v2 subject-level rollup — quote these numbers alongside the scan-level AUC.\n",
            "from model.utils.metrics import aggregate_scan_to_subject\n",
            "from sklearn.metrics import roc_auc_score\n",
            "\n",
            "def report_subject_auc(probs, sids, labels, scan_order=None, name=''):\n",
            "    for reduce in ('mean', 'max', 'last' if scan_order is not None else 'median'):\n",
            "        try:\n",
            "            _, p, y = aggregate_scan_to_subject(\n",
            "                probs, sids, labels, reduce=reduce, scan_order=scan_order,\n",
            "            )\n",
            "        except ValueError as e:\n",
            "            print(f'  [{name}] reduce={reduce}: {e}')\n",
            "            continue\n",
            "        auc = roc_auc_score(y, p) if len(set(y)) > 1 else float('nan')\n",
            "        print(f'  [{name}] reduce={reduce:<6} n_subjects={len(y):<4} subject-AUC={auc:.4f}')\n",
            "\n",
            "# Expects the notebook to have produced: probs, scan_subject_ids, labels.\n",
            "# Adapt variable names per notebook if needed.\n",
            "try:\n",
            "    report_subject_auc(probs, scan_subject_ids, labels, name='test')\n",
            "except NameError:\n",
            "    print('Set probs / scan_subject_ids / labels (scan-level arrays) before running this cell.')\n",
        ],
        tag=SUBJECT_ROLLUP_TAG,
    )
    nb["cells"].extend([md, code])


# ── Per-notebook framing ──────────────────────────────────────────────────────

FRAMING = {
    "GELSTM_DELCODE_WHOLE_BRAIN.ipynb": [
        "# [v2] Trajectory classification — GELSTM (whole-brain)\n",
        "\n",
        "**Prediction task: trajectory classification.** The model sees the *entire*\n",
        "visit history of each subject and outputs one subject-level converter\n",
        "probability. This is **NOT early detection** — for that, see\n",
        "`EARLY_DETECTION_GELSTM_FIRST_N.ipynb`, which restricts the model to the\n",
        "first N=2 or N=3 visits with `require_full_window=True`.\n",
        "\n",
        "Sanity checks (split overlap, Δt-only baseline, shuffled-order, fixed-N,\n",
        "duplicate-matrix audit) live in the four `SANITY_*` notebooks. The cell\n",
        "below runs the cheapest of them (split overlap) at the head of training.\n",
    ],
    "GELSTM_FDR_FILTERED_DELCODE_WHOLE_BRAIN.ipynb": [
        "# [v2] Trajectory classification — GELSTM with per-fold FDR filtering\n",
        "\n",
        "Same task as `GELSTM_DELCODE_WHOLE_BRAIN.ipynb` (**trajectory**, not early\n",
        "detection) — adds per-fold FDR-based selection of GAAE latent dimensions.\n",
    ],
    "LONGITUDINAL_GEC_MLP_DELCODE_WHOLE_BRAIN.ipynb": [
        "# [v2] Trajectory classification — Long-GEC-MLP\n",
        "\n",
        "**Task: trajectory classification.** Visits are flattened into a single\n",
        "fixed-width feature vector (top-K FDR latent dims × MAX_VISITS + Δt + mask)\n",
        "and passed to an MLP. No temporal ordering is preserved — this is a\n",
        "*static* whole-trajectory classifier, NOT early detection.\n",
    ],
    "GAAE_LOGREG_CLASSIFIER.ipynb": [
        "# [v2] Static per-scan classification — GAAE + LogReg\n",
        "\n",
        "**Task: static single-scan classification.** Each scan is one sample;\n",
        "labels are the subject's eventual converter status. This is *not* longitudinal\n",
        "and *not* early-detection — it tests whether a single scan's GAAE embedding\n",
        "is predictive on its own.\n",
        "\n",
        "v2 adds **subject-level rollup** of the scan-level AUC at the end of the\n",
        "notebook. Quote both numbers in any final table.\n",
    ],
    "MODEL_COMPARISON_DELCODE_WHOLE_BRAIN.ipynb": [
        "# [v2] Model Comparison\n",
        "\n",
        "Each row in this comparison must be tagged with one of three **Task** values:\n",
        "* **trajectory** — model sees the entire visit history (GELSTM, GELSTM-FDR, Long-GEC-MLP).\n",
        "* **early_detection** — model sees only the first N visits (`EARLY_DETECTION_GELSTM_FIRST_N`).\n",
        "* **static_per_scan** — model treats each scan independently (GAAE-LogReg, GEC).\n",
        "\n",
        "All AUC numbers in the headline table are **subject-level**. Scan-level numbers\n",
        "are moved to an appendix table. The metadata-only baseline (from\n",
        "`SANITY_TIME_METADATA_BASELINE.ipynb`) is included as an explicit floor.\n",
    ],
}

# Per-notebook: which splits dir to point at, and whether to append a
# subject-level rollup at the end.
TARGETS = {
    "GELSTM_DELCODE_WHOLE_BRAIN.ipynb":              ("splits_gaae", False),
    "GELSTM_FDR_FILTERED_DELCODE_WHOLE_BRAIN.ipynb": ("splits_gaae", False),
    "LONGITUDINAL_GEC_MLP_DELCODE_WHOLE_BRAIN.ipynb": ("splits_gaae", False),
    "GAAE_LOGREG_CLASSIFIER.ipynb":                  ("splits_gaae", True),
    "MODEL_COMPARISON_DELCODE_WHOLE_BRAIN.ipynb":    ("splits_gaae", False),
}


def _strip_existing_tags(nb):
    """Remove any cells previously inserted by this patcher (so re-runs are idempotent)."""
    keep = []
    drop_tags = {SANITY_CELL_TAG, FRAMING_CELL_TAG, SUBJECT_ROLLUP_TAG, TASK_COLUMN_TAG}
    for c in nb["cells"]:
        tags = set(c.get("metadata", {}).get("tags") or [])
        if tags & drop_tags:
            continue
        keep.append(c)
    nb["cells"] = keep


def _append_task_column_summary(nb):
    """Append a Task-tagged comparison + sanity-row summary to MODEL_COMPARISON."""
    if _has_tag(nb, TASK_COLUMN_TAG):
        return
    md = _make_md_cell(
        [
            "## v2 — Task-tagged comparison + sanity floor\n",
            "\n",
            "Headline table for the paper. Every row is tagged with its **Task** so a\n",
            "reviewer can tell a trajectory classifier from an early-detection one at a\n",
            "glance. AUC is **subject-level** for every model. The last two rows are the\n",
            "sanity floors from `SANITY_TIME_METADATA_BASELINE.ipynb` and\n",
            "`SANITY_LSTM_CHECKS.ipynb` — quote any LSTM row *relative to these*.\n",
        ],
        tag=TASK_COLUMN_TAG,
    )
    code = _make_code_cell(
        [
            "# v2 — task-tagged comparison table. Populate the cv/test AUC columns\n",
            "# from the per-model dicts produced earlier in this notebook.\n",
            "import pandas as pd\n",
            "\n",
            "v2_rows = [\n",
            "    # name,                    task,             cv_auc, cv_std, test_auc\n",
            "    ('GELSTM (no FDR)',          'trajectory',     None,   None,   None),\n",
            "    ('GELSTM FDR',               'trajectory',     None,   None,   None),\n",
            "    ('Longitudinal GEC-MLP',     'trajectory',     None,   None,   None),\n",
            "    ('GAAE LogReg',              'static_per_scan',None,   None,   None),\n",
            "    ('GEC Baseline',             'static_per_scan',None,   None,   None),\n",
            "    ('GELSTM early-detection N=2','early_detection',None,   None,   None),\n",
            "    ('GELSTM early-detection N=3','early_detection',None,   None,   None),\n",
            "    # Sanity floors:\n",
            "    ('Metadata-only LogReg',     'metadata_only',  None,   None,   None),\n",
            "    ('LSTM shuffled-order',      'sanity_ablation',None,   None,   None),\n",
            "    ('LSTM no-Δt',               'sanity_ablation',None,   None,   None),\n",
            "]\n",
            "v2_table = pd.DataFrame(v2_rows, columns=['model','task','cv_auc','cv_std','test_auc'])\n",
            "v2_table\n",
        ],
        tag=TASK_COLUMN_TAG,
    )
    nb["cells"].extend([md, code])


def patch_one(path: Path):
    nb = json.loads(path.read_text())
    _strip_existing_tags(nb)
    _patch_sys_path(nb)
    framing = FRAMING.get(path.name)
    if framing:
        _insert_framing(nb, framing)
    split_subpath, append_rollup = TARGETS[path.name]
    _insert_sanity_call(nb, split_subpath)
    if append_rollup:
        _append_subject_rollup(nb)
    if path.name == "MODEL_COMPARISON_DELCODE_WHOLE_BRAIN.ipynb":
        _append_task_column_summary(nb)
    path.write_text(json.dumps(nb, indent=1))
    print(f"  patched: {path.name}")


def main():
    for name in TARGETS:
        p = V2_NB / name
        if not p.exists():
            print(f"  missing: {p}")
            continue
        patch_one(p)


if __name__ == "__main__":
    main()
