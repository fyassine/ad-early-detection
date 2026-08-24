"""Backfill the winning fold's normalization statistics into TFGN checkpoints.

Why this exists
---------------
``TFGNAdapter.load_state`` needs four numbers that are not model weights — the
``log Δt`` StandardScaler's mean/scale and the strength-centrality z-scoring
mean/std, all fitted on the *winning fold's training subjects*
(``adapters/tfgn.py``'s ``train_fold``). Until ``checkpoint_extras`` was wired,
``model_state_for_save`` dropped them at save time, so every TFGN run completed
before that fix carries a checkpoint that cannot be re-scored on a held-out
split. This is the Tier-4 blocker in ``DOCS/flipped/PLAN.md`` section G.

Why a backfill is exact, not an approximation
---------------------------------------------
The statistics are a deterministic function of the winning fold's training
subjects, and that subject set is fully recoverable from artifacts already on
disk:

* ``run_summary.json["best_fold"]`` records which fold won.
* ``oof_predictions.csv`` records every subject's fold assignment. The
  ``StratifiedGroupKFold`` in ``common/crossval.run_kfold_cv`` takes no seed and
  no shuffle, so the map is identical across every seed and every arm.

So the winning fold's training set is exactly the rows with
``fold != best_fold``, and refitting on them reproduces the original numbers
rather than approximating them. ``--validate`` proves that claim per run by
re-scoring the held-out winning fold and checking the result against the
predictions the original run wrote.

Usage
-----
    python scripts/backfill_tfgn_norm_stats.py --dry-run --id tfgn-s1-flip-pooled-seed42
    python scripts/backfill_tfgn_norm_stats.py --id tfgn-s1-flip-pooled-seed{42..45}
    python scripts/backfill_tfgn_norm_stats.py --validate --id ...
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

_CLASSIFIER_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _CLASSIFIER_ROOT.parent
for _p in (str(_REPO_ROOT), str(_CLASSIFIER_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

NORM_KEYS = (
    "log_dt_scaler_mean",
    "log_dt_scaler_scale",
    "cent_mean",
    "cent_std",
)


def resolve_run_dir(exp_id: str) -> Path:
    run_dir = _CLASSIFIER_ROOT / "outputs" / exp_id / "latest"
    if not run_dir.is_dir():
        raise FileNotFoundError(f"{exp_id}: no outputs/<id>/latest directory.")
    return run_dir.resolve()


def checkpoint_path(run_dir: Path) -> Path:
    matches = sorted(run_dir.glob("checkpoint_*.pth"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"{run_dir}: expected exactly one checkpoint_*.pth, found {len(matches)}."
        )
    return matches[0]


def winning_fold_train_subjects(run_dir: Path, best_fold: int) -> List[str]:
    """Subject ids the winning fold trained on: every OOF row not in that fold."""
    oof_path = run_dir / "oof_predictions.csv"
    if not oof_path.is_file():
        raise FileNotFoundError(
            f"{run_dir}: no oof_predictions.csv — the fold map cannot be recovered, "
            "so the winning fold's training split is unknown. Re-run this arm instead."
        )
    oof = pd.read_csv(oof_path)
    if best_fold not in set(oof["fold"]):
        raise ValueError(
            f"{run_dir}: best_fold={best_fold} is absent from oof_predictions.csv "
            f"(folds present: {sorted(set(oof['fold']))})."
        )
    return oof.loc[oof["fold"] != best_fold, "subject_id"].astype(str).tolist()


def build_adapter_and_items(run_dir: Path, subject_ids: List[str], device: str):
    """Rebuild the run's adapter and the TFGNItems for `subject_ids`.

    Adapter construction goes through ``common.frozen_read.build_adapter_from_run``
    — the same path the frozen read itself uses — so the backfill cannot refit
    against a differently-configured adapter than the one that will consume it.
    """
    from common.frozen_read import build_adapter_from_run

    summary = json.loads((run_dir / "run_summary.json").read_text())
    pooled = _REPO_ROOT / "DATA" / "POOLED_ADNI_DELCODE"
    splits = pooled / "SPLITS" / "downstream"
    frame = pd.concat(
        [pd.read_csv(splits / f"{s}.csv") for s in ("train", "val")], ignore_index=True
    )
    wanted = {str(s) for s in subject_ids}
    frame = frame[frame["subject_id"].astype(str).isin(wanted)].copy()
    missing = wanted - set(frame["subject_id"].astype(str))
    if missing:
        raise ValueError(
            f"{run_dir}: {len(missing)} winning-fold training subjects are absent "
            f"from the pooled CV split CSVs (e.g. {sorted(missing)[:3]}). The split "
            "files have changed since the run — refusing to refit on a different pool."
        )

    adapter, summary = build_adapter_from_run(
        run_dir,
        adapter_key="tfgn",
        data_root=str(pooled / "__fc_wholebrain_sch200_flat__" / "matrices"),
        cohorts_csv=None,
        gaae_ckpt_path=summary.get("gaae_checkpoint") or "",
        gaae_hp={},
        device=device,
    )
    bundle = adapter.prepare_data(frame)
    items = adapter._prepare_tfgn_items(bundle.items)
    if len(items) != len(wanted):
        raise ValueError(
            f"{run_dir}: built {len(items)} items for {len(wanted)} winning-fold "
            "training subjects — the refit pool does not match the original."
        )
    return adapter, items, summary


def recompute_stats(items) -> Dict[str, Any]:
    """Refit exactly what adapters/tfgn.py::train_fold fits on its training items."""
    from sklearn.preprocessing import StandardScaler

    all_log_dt = np.concatenate([it.log_dt.numpy() for it in items], axis=0)
    scaler = StandardScaler().fit(all_log_dt.reshape(-1, 1))
    all_cent = np.concatenate([it.strength_centrality.numpy() for it in items], axis=0)
    return {
        "log_dt_scaler_mean": scaler.mean_.tolist(),
        "log_dt_scaler_scale": scaler.scale_.tolist(),
        "cent_mean": float(all_cent.mean()),
        "cent_std": float(max(all_cent.std(), 1e-8)),
    }


def validate(
    run_dir: Path, adapter, stats: Dict[str, Any], summary, device: str
) -> Dict[str, float]:
    """Re-score the winning fold with the recomputed stats and compare to the record.

    If these statistics are the originals, re-scoring the held-out winning fold
    must reproduce the probabilities the original run wrote to
    ``oof_predictions.csv`` for that fold — and hence its AUC. Any mismatch means
    the refit did not recover the original scaling; the caller must not proceed
    to a frozen read on that basis.
    """
    from adapters import load_run_checkpoint, model_state_from_checkpoint
    from sklearn.metrics import roc_auc_score

    best_fold = int(summary["best_fold"])
    oof = pd.read_csv(run_dir / "oof_predictions.csv")
    held = oof[oof["fold"] == best_fold].copy()

    pooled = _REPO_ROOT / "DATA" / "POOLED_ADNI_DELCODE"
    splits = pooled / "SPLITS" / "downstream"
    frame = pd.concat(
        [pd.read_csv(splits / f"{s}.csv") for s in ("train", "val")], ignore_index=True
    )
    frame = frame[frame["subject_id"].astype(str).isin(set(held["subject_id"].astype(str)))]

    ckpt = load_run_checkpoint(run_dir, device=device)
    state = {"model_state": model_state_from_checkpoint(ckpt), **stats}
    bundle = adapter.prepare_data(frame)
    metrics = adapter.eval_split(state, bundle, float(ckpt["best_threshold"]), device=device)

    scored = pd.DataFrame(
        {"subject_id": [str(s) for s in metrics["subject_ids"]], "prob_new": metrics["probs"]}
    )
    merged = held.merge(scored, on="subject_id", how="inner", validate="one_to_one")
    if len(merged) != len(held):
        raise ValueError(
            f"{run_dir}: re-scored {len(merged)} of {len(held)} winning-fold subjects."
        )
    max_abs_diff = float(np.abs(merged["prob"] - merged["prob_new"]).max())
    return {
        "n": len(merged),
        "auc_recorded": float(roc_auc_score(merged["label"], merged["prob"])),
        "auc_rescored": float(roc_auc_score(merged["label"], merged["prob_new"])),
        "val_auc_in_checkpoint": float(ckpt.get("val_auc", float("nan"))),
        "max_abs_prob_diff": max_abs_diff,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--id", dest="ids", action="append", required=True, help="experiment id (repeatable)"
    )
    ap.add_argument("--dry-run", action="store_true", help="recompute and report; write nothing")
    ap.add_argument(
        "--validate",
        action="store_true",
        help="re-score the winning fold and compare against its recorded OOF probs",
    )
    ap.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="max allowed |prob_recorded - prob_rescored| (default 1e-5: the "
        "model emits float32 and oof_predictions.csv round-trips through "
        "text, so exact equality is not achievable; a wrong scaler moves "
        "probabilities by ~1e-1, orders of magnitude above this floor)",
    )
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    failures: List[str] = []
    for exp_id in args.ids:
        print(f"\n=== {exp_id} ===")
        run_dir = resolve_run_dir(exp_id)
        summary = json.loads((run_dir / "run_summary.json").read_text())
        best_fold = int(summary["best_fold"])
        train_ids = winning_fold_train_subjects(run_dir, best_fold)
        print(f"  best_fold={best_fold}  winning-fold training subjects: {len(train_ids)}")

        adapter, items, summary = build_adapter_and_items(run_dir, train_ids, args.device)
        stats = recompute_stats(items)
        print(f"  log_dt mean={stats['log_dt_scaler_mean']} scale={stats['log_dt_scaler_scale']}")
        print(f"  cent mean={stats['cent_mean']:.6f} std={stats['cent_std']:.6f}")

        if args.validate:
            v = validate(run_dir, adapter, stats, summary, args.device)
            # Two criteria. The AUC equality is the decision-relevant one -- it is
            # what every downstream number is computed from, and a mis-scaled feature
            # cannot leave it invariant. The per-subject bound then confirms the
            # agreement is pointwise, not a coincidence of tied ranks.
            auc_delta = abs(v["auc_rescored"] - v["auc_recorded"])
            ok = auc_delta <= 1e-9 and v["max_abs_prob_diff"] <= args.tolerance
            print(
                f"  validation: n={v['n']} recorded_auc={v['auc_recorded']:.6f} "
                f"rescored_auc={v['auc_rescored']:.6f} "
                f"ckpt_val_auc={v['val_auc_in_checkpoint']:.6f} "
                f"max|Δprob|={v['max_abs_prob_diff']:.3e} "
                f"|Δauc|={auc_delta:.2e} -> {'PASS' if ok else 'FAIL'}"
            )
            if not ok:
                failures.append(exp_id)
                print("  refusing to write: recomputed statistics do not reproduce the run.")
                continue

        if args.dry_run:
            print("  --dry-run: checkpoint not modified.")
            continue

        ckpt_path = checkpoint_path(run_dir)
        backup = ckpt_path.with_suffix(".pth.pre-backfill")
        if not backup.exists():
            shutil.copy2(ckpt_path, backup)
        import torch
        from adapters import load_run_checkpoint

        ckpt = load_run_checkpoint(run_dir, device="cpu")
        already = [k for k in NORM_KEYS if k in ckpt]
        if already:
            print(f"  checkpoint already carries {already}; leaving it untouched.")
            continue
        ckpt.update(stats)
        torch.save(ckpt, ckpt_path)
        print(f"  wrote {list(NORM_KEYS)} into {ckpt_path.name} (backup: {backup.name})")

    if failures:
        print(f"\nFAILED validation: {failures}")
        return 1
    print("\nAll requested runs processed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
