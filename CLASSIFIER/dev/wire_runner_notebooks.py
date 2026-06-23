"""
Idempotent patcher: wire each registry notebook's call sites to consume the
injected papermill parameters (from the `parameters` cell added by
patch_runner_params.py).

Applies, where each construct exists:
  1. checkpoint guard      — `if GAAE_CHECKPOINT_PATH is not None: <resolve>` else prompt
  2. threshold guard       — branch on THRESHOLD_MODE, else original input()
  3. RESOLVED_CONFIG merge  — over the JSON-loaded TRAIN_CONFIG
  4. RUN_DIR preference     — over make_run_dir(...)
  5. W&B (per-epoch curves) — tracking.init_run before the CV loop, log per epoch,
                              log CV-summary + test metrics, finish
  6. uniform `metrics` block in the patch_run_summary(...) call

Every edit is guarded by an assert: a drifted notebook fails loudly rather than
being silently skipped. A notebook already containing `tracking.init_run` is
treated as wired and skipped (idempotent).

Run with:  python CLASSIFIER/dev/wire_runner_notebooks.py
"""
from __future__ import annotations

import json
from pathlib import Path

NB_ROOT = Path(__file__).resolve().parents[1] / "notebooks"

WIRED_MARKER = "tracking.init_run"


# --------------------------------------------------------------------------- #
# Low-level cell helpers
# --------------------------------------------------------------------------- #
def src(nb, i):
    return "".join(nb["cells"][i]["source"])


def setsrc(nb, i, s):
    nb["cells"][i]["source"] = s.splitlines(keepends=True)


def replace(nb, i, old, new):
    s = src(nb, i)
    assert old in s, f"cell {i}: anchor not found:\n{old!r}"
    setsrc(nb, i, s.replace(old, new, 1))


def append(nb, i, text):
    s = src(nb, i)
    if not s.endswith("\n"):
        s += "\n"
    setsrc(nb, i, s + text)


def already_wired(nb, marker=WIRED_MARKER):
    return any(marker in src(nb, i) for i in range(len(nb["cells"])))


# --------------------------------------------------------------------------- #
# Shared snippets
# --------------------------------------------------------------------------- #
CONFIG_MERGE = (
    "    print(f'Training config not found at {TRAIN_CONFIG_PATH} — using inline defaults.')\n"
)
CONFIG_MERGE_NEW = CONFIG_MERGE + (
    "\n# Runner override: merge injected RESOLVED_CONFIG (YAML hyperparams) over JSON config.\n"
    "if RESOLVED_CONFIG:\n"
    "    TRAIN_CONFIG = {**TRAIN_CONFIG, **RESOLVED_CONFIG}\n"
    "    print('Applied RESOLVED_CONFIG overrides from runner.')\n"
)

CKPT_PROMPT = (
    "selected_idx = int(input('Select checkpoint index: '))\n"
    "GAAE_RUN_NAME, GAAE_CKPT_PATH, GAAE_RUN_DIR = checkpoint_candidates[selected_idx]\n"
)
CKPT_GUARD = (
    "if GAAE_CHECKPOINT_PATH is not None:\n"
    "    _t = str(Path(GAAE_CHECKPOINT_PATH).resolve())\n"
    "    _m = [c for c in checkpoint_candidates if str(Path(c[1]).resolve()) == _t]\n"
    "    if not _m:\n"
    "        raise FileNotFoundError(f'GAAE_CHECKPOINT_PATH={GAAE_CHECKPOINT_PATH!r} not among '\n"
    "                                f'candidates: {[c[1] for c in checkpoint_candidates]}')\n"
    "    GAAE_RUN_NAME, GAAE_CKPT_PATH, GAAE_RUN_DIR = _m[0]\n"
    "else:\n"
    "    selected_idx = int(input('Select checkpoint index: '))\n"
    "    GAAE_RUN_NAME, GAAE_CKPT_PATH, GAAE_RUN_DIR = checkpoint_candidates[selected_idx]\n"
)


def wb_init(exp_id, mode, model):
    return (
        "from common import tracking\n"
        f"_wb_exp = {{'id': EXPERIMENT_ID or '{exp_id}', 'mode': MODE or '{mode}', "
        f"'model': MODEL or '{model}', 'dataset': DATASET or REGION, 'seed': SEED, "
        "'wandb': WANDB_ENABLED}\n"
        "wandb_run = tracking.init_run(_wb_exp, {**(RESOLVED_CONFIG or {}), 'REGION': REGION})\n\n"
    )


PEREPOCH = (
    "        tracking.log_metrics(wandb_run, {{'fold': fold+1, 'epoch': epoch+1, "
    "'train_loss': tr_loss, 'val_auc': va_auc}})\n"
)


def wb_cv_summary():
    return (
        "\ntry:\n"
        "    tracking.log_metrics(wandb_run, {'cv_best_val_auc': float(best_val_auc), "
        "'active_threshold': float(ACTIVE_THRESHOLD)})\n"
        "except NameError:\n"
        "    pass\n"
    )


def wb_test_finish(auc, f1, sens, spec):
    return (
        "\ntry:\n"
        f"    tracking.log_metrics(wandb_run, {{'test_auc': float({auc}), 'test_f1': float({f1}), "
        f"'test_sensitivity': float({sens}), 'test_specificity': float({spec})}})\n"
        "    tracking.finish_run(wandb_run)\n"
        "except NameError:\n"
        "    pass\n"
    )


def metrics_block(auc, f1, sens, spec):
    return (
        "    'metrics': {\n"
        f"        'test_auc': float({auc}), 'test_f1': float({f1}),\n"
        f"        'test_sensitivity': float({sens}), 'test_specificity': float({spec}),\n"
        "        'threshold': float(ACTIVE_THRESHOLD), 'threshold_method': THRESHOLD_METHOD,\n"
        "    },\n"
    )


def threshold_guard(youden_var, f1_var, prompt_block):
    """Wrap an existing interactive threshold block with a THRESHOLD_MODE guard."""
    guard = (
        "if THRESHOLD_MODE == 'best-f1':\n"
        f"    ACTIVE_THRESHOLD = {f1_var}; THRESHOLD_METHOD = 'oof_f1'\n"
        "elif THRESHOLD_MODE == 'youden':\n"
        f"    ACTIVE_THRESHOLD = {youden_var}; THRESHOLD_METHOD = 'oof_youden'\n"
        "elif THRESHOLD_MODE == 'fixed':\n"
        "    if FIXED_THRESHOLD is None:\n"
        "        raise ValueError(\"THRESHOLD_MODE='fixed' requires FIXED_THRESHOLD\")\n"
        "    ACTIVE_THRESHOLD = float(FIXED_THRESHOLD); THRESHOLD_METHOD = 'fixed'\n"
        "else:\n"
    )
    indented = "".join("    " + ln if ln.strip() else ln
                       for ln in prompt_block.splitlines(keepends=True))
    return guard + indented


# --------------------------------------------------------------------------- #
# Per-notebook wiring
# --------------------------------------------------------------------------- #
def wire_gelstm_flagship(nb):
    replace(nb, 8, CONFIG_MERGE, CONFIG_MERGE_NEW)
    replace(nb, 11, CKPT_PROMPT, CKPT_GUARD)
    # threshold (cell 20)
    prompt20 = (
        "choice = input('Select threshold [1=Youden (default), 2=Best-F1]: ').strip()\n"
        "if choice == '2':\n"
        "    ACTIVE_THRESHOLD = best_f1_threshold; THRESHOLD_METHOD = 'oof_f1'\n"
        "else:\n"
        "    ACTIVE_THRESHOLD = best_threshold_overall; THRESHOLD_METHOD = 'oof_youden'\n"
    )
    replace(nb, 20, prompt20, threshold_guard("best_threshold_overall", "best_f1_threshold", prompt20))
    # W&B init before CV loop + per-epoch (cell 17)
    replace(nb, 17, "sgkf = StratifiedGroupKFold(n_splits=N_FOLDS)",
            wb_init("gelstm-trajectory-whole-brain", "longitudinal", "GELSTM")
            + "sgkf = StratifiedGroupKFold(n_splits=N_FOLDS)")
    replace(nb, 17, "        scheduler.step(va_auc)\n        fold_train_losses.append(tr_loss)",
            "        scheduler.step(va_auc)\n" + PEREPOCH.format()
            + "        fold_train_losses.append(tr_loss)")
    # run-dir preference (cell 22) + CV summary log
    replace(nb, 22, "run_name, run_dir = make_run_dir(OUTPUT_DIR, 'gelstm', DATA_INFO)",
            "if RUN_DIR:\n    run_dir = Path(RUN_DIR); run_dir.mkdir(parents=True, exist_ok=True)\n"
            "    run_name = RUN_NAME or run_dir.name\nelse:\n"
            "    run_name, run_dir = make_run_dir(OUTPUT_DIR, 'gelstm', DATA_INFO)")
    append(nb, 22, wb_cv_summary())
    # metrics block + test log/finish (cell 24)
    replace(nb, 24, "patch_run_summary(run_dir, {\n    'test_auc':          float(te_metrics['auc']),",
            "patch_run_summary(run_dir, {\n"
            + metrics_block("te_metrics['auc']", "te_metrics['f1']",
                            "te_metrics['sensitivity']", "te_metrics['specificity']")
            + "    'test_auc':          float(te_metrics['auc']),")
    append(nb, 24, wb_test_finish("te_metrics['auc']", "te_metrics['f1']",
                                  "te_metrics['sensitivity']", "te_metrics['specificity']"))


def _wire_cv_inline(nb, *, exp_id, model_tag_line, ckpt_cell, thr_cell, thr_prompt,
                    youden_var, f1_var, loop_cell, loop_anchor, rundir_cell,
                    rundir_old, patch_cell, patch_header, te):
    replace(nb, 8, CONFIG_MERGE, CONFIG_MERGE_NEW)
    replace(nb, ckpt_cell, CKPT_PROMPT, CKPT_GUARD)
    replace(nb, thr_cell, thr_prompt, threshold_guard(youden_var, f1_var, thr_prompt))
    replace(nb, loop_cell, "sgkf = StratifiedGroupKFold(n_splits=N_FOLDS)",
            wb_init(exp_id, "longitudinal", "GELSTM" if "gelstm" in exp_id else "GEC")
            + "sgkf = StratifiedGroupKFold(n_splits=N_FOLDS)")
    replace(nb, loop_cell, loop_anchor, loop_anchor.split("\n")[0] + "\n" + PEREPOCH.format()
            + "\n".join(loop_anchor.split("\n")[1:]))
    replace(nb, rundir_cell, rundir_old,
            "if RUN_DIR:\n    run_dir = Path(RUN_DIR); run_dir.mkdir(parents=True, exist_ok=True)\n"
            "    run_name = RUN_NAME or run_dir.name\nelse:\n    " + rundir_old)
    append(nb, rundir_cell, wb_cv_summary())
    replace(nb, patch_cell, patch_header,
            patch_header.split("{\n")[0] + "{\n" + metrics_block(*te)
            + patch_header.split("{\n")[1])
    append(nb, patch_cell, wb_test_finish(*te))


def wire_gelstm_fdr(nb):
    thr_prompt = (
        "choice=input('Select [1=Youden, 2=Best-F1]: ').strip()\n"
        "ACTIVE_THRESHOLD=best_f1_threshold if choice=='2' else best_threshold_overall\n"
        "THRESHOLD_METHOD='oof_f1' if choice=='2' else 'oof_youden'\n"
    )
    _wire_cv_inline(
        nb, exp_id="gelstm-trajectory-fdr", model_tag_line=None,
        ckpt_cell=11, thr_cell=27, thr_prompt=thr_prompt,
        youden_var="best_threshold_overall", f1_var="best_f1_threshold",
        loop_cell=24,
        loop_anchor="        scheduler.step(va_auc)\n        fold_train_losses.append(tr_loss)",
        rundir_cell=29,
        rundir_old="run_name, run_dir = make_run_dir(OUTPUT_DIR, f'gelstm_fdr_{TOP_K}', DATA_INFO)",
        patch_cell=31,
        patch_header="patch_run_summary(run_dir, {\n    'test_auc':          float(te_auc),",
        te=("te_auc", "te_f1", "te_sens", "te_spec"),
    )


def wire_gec(nb):
    thr_prompt = (
        "choice = input('Select [1=Youden, 2=Best-F1]: ').strip()\n"
        "ACTIVE_THRESHOLD = best_f1_thr if choice == '2' else best_threshold_overall\n"
        "THRESHOLD_METHOD = 'oof_f1' if choice == '2' else 'oof_youden'\n"
    )
    _wire_cv_inline(
        nb, exp_id="gec-trajectory", model_tag_line=None,
        ckpt_cell=11, thr_cell=27, thr_prompt=thr_prompt,
        youden_var="best_threshold_overall", f1_var="best_f1_thr",
        loop_cell=24,
        loop_anchor="        sched.step(va_auc)\n\n        fold_tr_losses.append(tr_loss)",
        rundir_cell=29,
        rundir_old="run_name, run_dir = make_run_dir(OUTPUT_DIR, f'long_gec_mlp_{TOP_K}dims', DATA_INFO)",
        patch_cell=31,
        patch_header="patch_run_summary(run_dir, {\n    'test_auc':          float(te_auc),",
        te=("te_auc", "te_f1", "te_sens", "te_spec"),
    )


def wire_logreg(nb):
    # checkpoint via the central helper — pass checkpoint_path through.
    replace(nb, 11,
            "GAAE_RUN_NAME, _ckpt_path, GAAE_RUN_DIR = select_gaae_checkpoint(CHECKPOINT_SEARCH_DIRS)",
            "GAAE_RUN_NAME, _ckpt_path, GAAE_RUN_DIR = select_gaae_checkpoint(\n"
            "    CHECKPOINT_SEARCH_DIRS, checkpoint_path=GAAE_CHECKPOINT_PATH)")
    replace(nb, 8, CONFIG_MERGE, CONFIG_MERGE_NEW)
    thr_prompt = (
        "choice = input('Select threshold [1=Youden (default), 2=Best-F1]: ').strip()\n"
        "if choice == '2':\n"
        "    ACTIVE_THRESHOLD = best_f1_threshold\n"
        "    THRESHOLD_METHOD = 'oof_f1'\n"
        "else:\n"
        "    ACTIVE_THRESHOLD = best_threshold_overall\n"
        "    THRESHOLD_METHOD = 'oof_youden'\n"
    )
    replace(nb, 21, thr_prompt, threshold_guard("best_threshold_overall", "best_f1_threshold", thr_prompt))
    # sklearn: no epoch loop. Init W&B + log CV summary in the run-dir cell (23).
    replace(nb, 23, "run_name, run_dir = make_run_dir(OUTPUT_DIR, 'gaae_logreg', DATA_INFO)",
            wb_init("logreg-static", "static", "LogReg")
            + "if RUN_DIR:\n    run_dir = Path(RUN_DIR); run_dir.mkdir(parents=True, exist_ok=True)\n"
            "    run_name = RUN_NAME or run_dir.name\nelse:\n"
            "    run_name, run_dir = make_run_dir(OUTPUT_DIR, 'gaae_logreg', DATA_INFO)")
    append(nb, 23, wb_cv_summary())
    replace(nb, 25, "patch_run_summary(LOGREG_RUN_DIR, {\n    'test_auc':           float(te_auc),",
            "patch_run_summary(LOGREG_RUN_DIR, {\n"
            + metrics_block("te_auc", "te_f1", "te_sens", "te_spec")
            + "    'test_auc':           float(te_auc),")
    append(nb, 25, wb_test_finish("te_auc", "te_f1", "te_sens", "te_spec"))


def wire_first_n(nb):
    # Checkpoint: let the runner supply the GAAE encoder (default '' = none, as before).
    replace(nb, 5,
            "GAAE_CKPT_PATH   = ''   # set to GAAE encoder checkpoint produced by GAAE_DELCODE_WHOLE_BRAIN",
            "GAAE_CKPT_PATH   = GAAE_CHECKPOINT_PATH or ''   # runner-supplied GAAE encoder, else none")
    # W&B per-epoch inside run_cv (cell 9) + finish after the summary table.
    replace(nb, 9, "def run_cv(items, labels, sids, n_folds=N_FOLDS):",
            "from common import tracking\n"
            "_wb_exp = {'id': EXPERIMENT_ID or 'gelstm-early-detection-first-n', 'mode': MODE or 'longitudinal',\n"
            "           'model': MODEL or 'GELSTM', 'dataset': DATASET, 'seed': SEED, 'wandb': WANDB_ENABLED}\n"
            "wandb_run = tracking.init_run(_wb_exp, {**(RESOLVED_CONFIG or {})})\n\n"
            "def run_cv(items, labels, sids, n_folds=N_FOLDS):")
    replace(nb, 9,
            "            r    = evaluate(m, va_b, device)\n            if r['auc'] > best_auc:",
            "            r    = evaluate(m, va_b, device)\n"
            "            tracking.log_metrics(wandb_run, {'fold': fold+1, 'epoch': epoch+1, 'val_auc': r['auc']})\n"
            "            if r['auc'] > best_auc:")
    append(nb, 9,
           "\ntry:\n"
           "    for _r in rows:\n"
           "        tracking.log_metrics(wandb_run, {f\"{_r['window']}_cv_auc\": _r['cv_auc_mean'], "
           "f\"{_r['window']}_test_auc\": _r['test_auc']})\n"
           "    if RUN_DIR:\n"
           "        from common.provenance import write_run_summary, capture_git_provenance, capture_env\n"
           "        write_run_summary(RUN_DIR, {'experiment_id': EXPERIMENT_ID, 'timestamp': RUN_NAME,\n"
           "            'git': capture_git_provenance(), 'env': capture_env(),\n"
           "            'metrics': {r['window'] + '_test_auc': r['test_auc'] for r in rows}})\n"
           "    tracking.finish_run(wandb_run)\n"
           "except NameError:\n"
           "    pass\n")


def wire_sanity_gelstm(nb):
    # Guard the inline GELSTM-checkpoint prompt (cell 3). Headless: pick latest, or the
    # runner-supplied checkpoint if given.
    replace(nb, 3,
            "selected_idx = int(input('Select checkpoint index: '))\n"
            "GELSTM_RUN_NAME, GELSTM_CKPT_PATH, GELSTM_RUN_DIR = checkpoint_candidates[selected_idx]",
            "if GAAE_CHECKPOINT_PATH is not None:\n"
            "    _t = str(Path(GAAE_CHECKPOINT_PATH).resolve())\n"
            "    _m = [c for c in checkpoint_candidates if str(Path(c[1]).resolve()) == _t]\n"
            "    if not _m:\n"
            "        raise FileNotFoundError(f'GAAE_CHECKPOINT_PATH={GAAE_CHECKPOINT_PATH!r} not among candidates')\n"
            "    GELSTM_RUN_NAME, GELSTM_CKPT_PATH, GELSTM_RUN_DIR = _m[0]\n"
            "elif RUN_DIR is not None:\n"
            "    GELSTM_RUN_NAME, GELSTM_CKPT_PATH, GELSTM_RUN_DIR = checkpoint_candidates[-1]\n"
            "    print(f'Headless: using latest GELSTM checkpoint {GELSTM_RUN_NAME}')\n"
            "else:\n"
            "    selected_idx = int(input('Select checkpoint index: '))\n"
            "    GELSTM_RUN_NAME, GELSTM_CKPT_PATH, GELSTM_RUN_DIR = checkpoint_candidates[selected_idx]")


def wire_gaae(nb):
    # Guard the train-vs-load prompt (cell 6) so headless runs never block.
    replace(nb, 6,
            '    _idx_str = input("Select [0=train new / 1,2,...=load existing, Enter=train new]: ").strip()\n'
            "    _idx = int(_idx_str) if _idx_str.isdigit() else 0",
            "    if GAAE_CHECKPOINT_PATH is not None:\n"
            "        _idx_str = ''   # runner supplied an explicit checkpoint (handled below)\n"
            "    elif RUN_DIR is not None:\n"
            "        _idx_str = '0'  # headless default: train a new model\n"
            "    else:\n"
            '        _idx_str = input("Select [0=train new / 1,2,...=load existing, Enter=train new]: ").strip()\n'
            "    _idx = int(_idx_str) if _idx_str.isdigit() else 0")


# (wirer, idempotency-marker) — the marker is a string the wirer guarantees to
# add, so a second run detects an already-wired notebook and skips it.
WIRERS = {
    "LONGITUDINAL/LONGITUDINAL_GELSTM_DELCODE.ipynb": (wire_gelstm_flagship, WIRED_MARKER),
    "LONGITUDINAL/LONGITUDINAL_GELSTM_FDR_FILTERED_DELCODE.ipynb": (wire_gelstm_fdr, WIRED_MARKER),
    "LONGITUDINAL/LONGITUDINAL_GEC_DELCODE.ipynb": (wire_gec, WIRED_MARKER),
    "STATIC/STATIC_LOGREG_DELCODE_WHOLE_BRAIN.ipynb": (wire_logreg, WIRED_MARKER),
    "LONGITUDINAL/LONGITUDINAL_GELSTM_FIRST_N_DELCODE.ipynb": (wire_first_n, WIRED_MARKER),
    "SANITY/SANITY_LONGITUDINAL_GELSTM.ipynb": (wire_sanity_gelstm, "Headless: using latest GELSTM checkpoint"),
    "STATIC/STATIC_GAAE_DELCODE_WHOLE_BRAIN.ipynb": (wire_gaae, "headless default: train a new model"),
}


def main() -> int:
    import py_compile  # noqa: F401  (compile check via compile())
    for rel, (fn, marker) in WIRERS.items():
        path = NB_ROOT / rel
        nb = json.loads(path.read_text())
        if already_wired(nb, marker):
            print(f"  already-wired  {rel}")
            continue
        fn(nb)
        # Static check: every code cell must still compile.
        for i, c in enumerate(nb["cells"]):
            if c["cell_type"] == "code":
                code = src(nb, i)
                try:
                    compile(code, f"{rel}::cell{i}", "exec")
                except SyntaxError as e:
                    raise SystemExit(f"SYNTAX ERROR in {rel} cell {i}: {e}\n---\n{code}") from e
        path.write_text(json.dumps(nb, indent=1))
        print(f"  wired          {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
