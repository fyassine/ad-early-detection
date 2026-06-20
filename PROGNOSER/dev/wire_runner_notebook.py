#!/usr/bin/env python3
"""
Idempotent, fail-loud wiring of the PROGNOSER notebooks for the experiment runner.

Sibling of CLASSIFIER/dev/wire_runner_notebooks.py. Run from anywhere:

    python PROGNOSER/dev/wire_runner_notebook.py

What it does (each step skipped if already applied):
  PROGNOSER_RUNNER.ipynb
    - tag the EXPERIMENT-config cell `parameters` and add the runner-injected
      vars (EXPERIMENT_ID / RUN_DIR / RUN_NAME / WANDB_ENABLED / OUTPUT_DIR);
      move the import + path setup into a following cell so papermill's injected
      parameters land before paths are computed.
    - init a W&B run (CLASSIFIER.common.tracking, survival project) in the setup
      cell; pass a per-epoch logging callback into the lstm_surv fit; log the
      final metrics + close the run, and route git/env through provenance, in the
      save cell, honouring an injected RUN_DIR.
  KAPLAN_MEIER_BASELINE.ipynb, CROSS_NETWORK_COMPARISON.ipynb
    - add a `parameters` cell so they can be executed headless.
    - point CROSS_NETWORK_COMPARISON.collect_runs at PROGNOSER/outputs/ too
      (keeping the legacy checkpoints_prognoser_* glob for back-compat).

Every code cell is `compile()`-checked after editing; the notebook is rejected
(SystemExit) if a target anchor is missing or a cell no longer compiles.
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

import nbformat

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"
RUNNER = NB_DIR / "PROGNOSER_RUNNER.ipynb"
KM = NB_DIR / "KAPLAN_MEIER_BASELINE.ipynb"
LEADERBOARD = NB_DIR / "CROSS_NETWORK_COMPARISON.ipynb"


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def find_cell(nb, needle: str) -> int:
    for i, c in enumerate(nb.cells):
        if c.cell_type == "code" and needle in c.source:
            return i
    raise SystemExit(f"[wire] anchor not found: {needle!r}")


def replace(src: str, old: str, new: str) -> str:
    if old not in src:
        raise SystemExit(f"[wire] substring to replace not found:\n{old}")
    return src.replace(old, new)


def _ensure_cell_ids(nb) -> None:
    """Give every id-less cell a fresh id so writes never reintroduce the
    nbformat MissingIDFieldWarning (see CLASSIFIER/dev/normalize_notebook_ids.py)."""
    for cell in nb.cells:
        if not cell.get("id"):
            cell["id"] = uuid.uuid4().hex[:8]


def validate(nb, path: Path) -> None:
    for i, c in enumerate(nb.cells):
        if c.cell_type != "code":
            continue
        try:
            compile(c.source, f"{path.name}[cell {i}]", "exec")
        except SyntaxError as exc:
            raise SystemExit(f"[wire] {path.name} cell {i} does not compile: {exc}\n---\n{c.source}\n---")
    _ensure_cell_ids(nb)
    nbformat.validate(nb)


# --------------------------------------------------------------------------- #
# Shared parameters cell for the auxiliary notebooks
# --------------------------------------------------------------------------- #
PARAMS_CELL = (
    "# Papermill parameters (injected by run_experiment.py; safe interactive defaults).\n"
    "EXPERIMENT_ID = None\n"
    "RUN_DIR = None\n"
    "RUN_NAME = None\n"
    "WANDB_ENABLED = True\n"
    "OUTPUT_DIR = None\n"
)

EXPERIMENT_DICT = """\
# ── EXPERIMENT CONFIG (papermill parameters) ──────────────────────────────────
# Edit for interactive Jupyter use. run_experiment.py overrides EXPERIMENT and the
# RUN_* / WANDB_ENABLED / OUTPUT_DIR vars below via papermill.
EXPERIMENT = {
    "network_combo": "dmn_hippo",
    "data_version": "__fc_dmn-hippo_sch200-tian2_flat__",
    "file_suffix": "_dmn_hippo_correlation_matrix_z_transformed.npz",
    "method": "cox_clinical_longitudinal",
        # km | cox_clinical | cox_embedding | cox_combined |
        # cox_clinical_longitudinal | cox_time_varying |
        # rsf | deepsurv | lstm_surv
    "feature_set": "clinical_longitudinal",
        # clinical | embedding | clinical+embedding |
        # clinical_longitudinal | clinical_longitudinal+embedding | sequence
    "embedding_strategy": "last",
        # baseline | last | mean | slope | all_aggs | sequence
    "longitudinal_features": ["mmstot", "cdrglobal"],
    "longitudinal_aggs": ["baseline", "last", "slope", "delta"],
    "eval_times": [12, 24, 36, 48, 60, 72],
    "penalizer": 0.05,
    "pca_components": 16,
    "rsf_n_estimators": 200,
    "rsf_min_samples_leaf": 5,
    "lstm_n_time_bins": 12,
    "lstm_max_horizon_months": 72,
    "lstm_hidden_dim": 64,
    "lstm_epochs": 100,
    "random_state": 42,
}
# Runner-injected parameters (None => interactive Jupyter behaviour):
EXPERIMENT_ID = None
RUN_DIR = None
RUN_NAME = None
WANDB_ENABLED = True
OUTPUT_DIR = None
# ─────────────────────────────────────────────────────────────────────────────"""

SETUP_CELL = """\
import sys, os, json
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path('/mnt/e/fyassine/ad-early-detection')
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DATA.src.splitting.load_splits import splits_dir
from CLASSIFIER.common import tracking, provenance

COHORTS_CSV = REPO_ROOT / 'DATA' / 'DELCODE' / '__fc_wholebrain_sch200_flat__' / 'metadata' / 'cohorts.csv'
SPLITS_DIR = splits_dir('downstream')
EMBEDDINGS_CACHE = REPO_ROOT / 'PROGNOSER' / 'notebooks' / '_embeddings_cache_'
CHECKPOINT_ROOT = REPO_ROOT / 'PROGNOSER' / 'notebooks' / f"checkpoints_prognoser_{EXPERIMENT['network_combo']}"
CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

print(f"Experiment: {EXPERIMENT['network_combo']} | Method: {EXPERIMENT['method']}")
print(f"Feature set: {EXPERIMENT['feature_set']} | Embedding strategy: {EXPERIMENT['embedding_strategy']}")

# W&B run: on by default, no-op stub when disabled / no creds, never blocks.
# Survival logs to its own project, separate from the classifier.
os.environ.setdefault('WANDB_PROJECT', 'ad-early-detection-prognosis')
_wandb_exp = {
    'id': EXPERIMENT_ID or RUN_NAME or f"{EXPERIMENT['method']}-{EXPERIMENT['network_combo']}",
    'model': EXPERIMENT['method'],
    'mode': 'survival',
    'dataset': EXPERIMENT['network_combo'],
    'seed': EXPERIMENT['random_state'],
    'wandb': WANDB_ENABLED,
}
wandb_run = tracking.init_run(_wandb_exp, {**EXPERIMENT, 'RUN_NAME': RUN_NAME})"""


# --------------------------------------------------------------------------- #
# PROGNOSER_RUNNER.ipynb
# --------------------------------------------------------------------------- #
def wire_runner() -> None:
    nb = nbformat.read(RUNNER, as_version=4)

    if "tracking.init_run" in "\n".join(c.source for c in nb.cells):
        print(f"[wire] {RUNNER.name}: already wired — skipping.")
        return

    # Replace the config cell with just the dict + injected vars, tag it.
    cfg_idx = find_cell(nb, '"network_combo": "dmn_hippo"')
    cfg = nb.cells[cfg_idx]
    cfg.source = EXPERIMENT_DICT
    tags = list(cfg.metadata.get("tags", []))
    if "parameters" not in tags:
        tags.append("parameters")
    cfg.metadata["tags"] = tags
    # Insert the setup cell (imports + paths + W&B init) right after it.
    nb.cells.insert(cfg_idx + 1, nbformat.v4.new_code_cell(SETUP_CELL))

    # lstm per-epoch logging via the pure epoch_callback hook.
    fit_idx = find_cell(nb, "model.fit_sequences(seq_tr, len_tr, T_tr, E_tr,")
    nb.cells[fit_idx].source = replace(
        nb.cells[fit_idx].source,
        "    model.fit_sequences(seq_tr, len_tr, T_tr, E_tr,\n"
        "                        val_data=(seq_va, len_va, T_va, E_va))",
        "    model.fit_sequences(\n"
        "        seq_tr, len_tr, T_tr, E_tr,\n"
        "        val_data=(seq_va, len_va, T_va, E_va),\n"
        "        epoch_callback=lambda e, tr, vl: tracking.log_metrics(\n"
        "            wandb_run, {'epoch': e, 'train_loss': tr, 'val_loss': vl}),\n"
        "    )",
    )

    # Save cell: honour RUN_DIR, route git/env via provenance, log + finish W&B.
    save_idx = find_cell(nb, "run_summary = {")
    save = nb.cells[save_idx]
    if "if RUN_DIR:" not in save.source:
        save.source = replace(
            save.source,
            "run_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')\n"
            "run_name = f\"{method}_{EXPERIMENT['feature_set'].replace('+','-')}_{EXPERIMENT['embedding_strategy']}_{run_timestamp}\"\n"
            "run_dir = CHECKPOINT_ROOT / run_name\n"
            "run_dir.mkdir(parents=True, exist_ok=True)",
            "if RUN_DIR:\n"
            "    run_dir = Path(RUN_DIR)\n"
            "    run_name = RUN_NAME or run_dir.name\n"
            "    run_timestamp = run_dir.name\n"
            "else:\n"
            "    run_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')\n"
            "    run_name = f\"{method}_{EXPERIMENT['feature_set'].replace('+','-')}_{EXPERIMENT['embedding_strategy']}_{run_timestamp}\"\n"
            "    run_dir = CHECKPOINT_ROOT / run_name\n"
            "run_dir.mkdir(parents=True, exist_ok=True)",
        )
        save.source = replace(
            save.source,
            "    'experiment': EXPERIMENT, 'method': method,",
            "    'experiment_id': EXPERIMENT_ID or run_name,\n"
            "    'experiment': EXPERIMENT, 'method': method,",
        )
        save.source = replace(
            save.source,
            "    'metrics': metrics,\n"
            "    'eval_times': EXPERIMENT['eval_times'],\n"
            "}",
            "    'metrics': metrics,\n"
            "    'eval_times': EXPERIMENT['eval_times'],\n"
            "    'git': provenance.capture_git_provenance(),\n"
            "    'env': provenance.capture_env(),\n"
            "}",
        )
        save.source = replace(
            save.source,
            "print(f'Saved: {run_dir}')",
            "# Log final metrics to W&B (flattened) and close the run.\n"
            "_wb_metrics = {}\n"
            "for _split, _m in metrics.items():\n"
            "    _wb_metrics[f'{_split}/c_index'] = _m.get('c_index')\n"
            "    _wb_metrics[f'{_split}/ibs'] = _m.get('ibs')\n"
            "    for _t, _v in (_m.get('auc') or {}).items():\n"
            "        _wb_metrics[f'{_split}/auc_{_t}'] = _v\n"
            "tracking.log_metrics(wandb_run, _wb_metrics)\n"
            "tracking.finish_run(wandb_run)\n"
            "\n"
            "print(f'Saved: {run_dir}')",
        )

    validate(nb, RUNNER)
    nbformat.write(nb, RUNNER)
    print(f"[wire] {RUNNER.name}: wired OK.")


# --------------------------------------------------------------------------- #
# Auxiliary notebooks: add a parameters cell
# --------------------------------------------------------------------------- #
def add_params_cell(path: Path) -> None:
    nb = nbformat.read(path, as_version=4)
    has_params = any("parameters" in c.metadata.get("tags", []) for c in nb.cells)
    if has_params:
        print(f"[wire] {path.name}: already has a parameters cell — skipping.")
        return
    cell = nbformat.v4.new_code_cell(PARAMS_CELL)
    cell.metadata["tags"] = ["parameters"]
    # Insert after a leading markdown title if present, else at the top.
    insert_at = 1 if nb.cells and nb.cells[0].cell_type == "markdown" else 0
    nb.cells.insert(insert_at, cell)
    validate(nb, path)
    nbformat.write(nb, path)
    print(f"[wire] {path.name}: parameters cell added.")


# --------------------------------------------------------------------------- #
# Leaderboard collector: also read PROGNOSER/outputs/
# --------------------------------------------------------------------------- #
COLLECTOR_CELL = """\
def _row_from_summary(s, combo, run_dir_name):
    metrics = s.get('metrics', {})
    test = metrics.get('test', {}); val = metrics.get('val', {}); train = metrics.get('train', {})
    test_auc = test.get('auc', {})
    return {
        'network_combo': combo,
        'method': s.get('method', 'unknown'),
        'feature_set': s.get('feature_set', ''),
        'embedding_strategy': s.get('embedding_strategy', 'baseline'),
        'n_features': s.get('n_features'),
        'n_train': s.get('n_train'), 'n_test': s.get('n_test'),
        'n_events_test': s.get('n_events_test'),
        'c_train': train.get('c_index'),
        'c_val':   val.get('c_index'),
        'c_test':  test.get('c_index'),
        'ibs_test': test.get('ibs'),
        'auc_test_24': test_auc.get('24'),
        'auc_test_36': test_auc.get('36'),
        'auc_test_60': test_auc.get('60'),
        'run_name': s.get('run_name', run_dir_name),
        'timestamp': s.get('timestamp', ''),
    }


def collect_runs(prognoser_dir: Path) -> pd.DataFrame:
    rows = []
    # Legacy layout: notebooks/checkpoints_prognoser_<combo>/<run_name>/
    for combo_dir in sorted(prognoser_dir.glob('checkpoints_prognoser_*')):
        if not combo_dir.is_dir():
            continue
        combo = combo_dir.name.replace('checkpoints_prognoser_', '')
        for run_dir in sorted(combo_dir.iterdir()):
            summary = run_dir / 'run_summary.json'
            if not summary.exists():
                continue
            with open(summary) as f:
                s = json.load(f)
            rows.append(_row_from_summary(s, combo, run_dir.name))
    # New layout written by run_experiment.py: ../outputs/<id>/runs/<ts>/
    outputs = prognoser_dir.parent / 'outputs'
    for summary in sorted(outputs.glob('*/runs/*/run_summary.json')):
        try:
            with open(summary) as f:
                s = json.load(f)
        except Exception:
            continue
        combo = (s.get('experiment') or {}).get('network_combo', 'unknown')
        rows.append(_row_from_summary(s, combo, summary.parent.name))
    return pd.DataFrame(rows)

leaderboard = collect_runs(PROGNOSER_DIR)
if leaderboard.empty:
    print('No runs found. Run PROGNOSER_RUNNER.ipynb first.')
else:
    leaderboard = leaderboard.sort_values(['c_test', 'auc_test_36'], ascending=False).reset_index(drop=True)
    print(f'Total runs: {len(leaderboard)}')
    print(f'Combos:    {sorted(leaderboard.network_combo.unique())}')
    print(f'Methods:   {sorted(leaderboard.method.unique())}')
    print(f'Strategies:{sorted(leaderboard.embedding_strategy.unique())}')
leaderboard"""


def wire_leaderboard_collector() -> None:
    nb = nbformat.read(LEADERBOARD, as_version=4)
    idx = find_cell(nb, "def collect_runs(prognoser_dir: Path)")
    cell = nb.cells[idx]
    if "prognoser_dir.parent / 'outputs'" in cell.source:
        print(f"[wire] {LEADERBOARD.name}: collector already reads outputs/ — skipping.")
        return
    cell.source = COLLECTOR_CELL
    validate(nb, LEADERBOARD)
    nbformat.write(nb, LEADERBOARD)
    print(f"[wire] {LEADERBOARD.name}: collector now reads outputs/ + legacy layout.")


def main() -> int:
    for p in (RUNNER, KM, LEADERBOARD):
        if not p.is_file():
            raise SystemExit(f"[wire] notebook missing: {p}")
    wire_runner()
    add_params_cell(KM)
    add_params_cell(LEADERBOARD)
    wire_leaderboard_collector()
    print("[wire] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
