#!/usr/bin/env python3
"""
verify_matched_cohort_parity.py — B.1.1/B.1.2 gate for the GELSTM vs BrainTokenGT
matched-cohort head-to-head (DOCS/timeline/MASTER_PLAN.md §3, Phase B).

Builds the CV-pool Bundle each adapter would see under the registered matched-cohort
experiments (``recon-ablation-gelstm-pretrained-frozen-2to3v-seed42`` and
``braintokengt-delcode-whole-brain-repaired-fix-stabilized``, both min_visits=2,
max_visits=3) via the exact same ``load_experiment`` / ``build_config`` /
``prepare_data`` path the notebooks and run_experiment.py use, then checks:

  1. Same subject-ID SET (B.1.1 item 4 — "if they don't match exactly, the 'fair'
     claim is dead on arrival — fix the mismatch before running anything").
  2. Same subject-ID ORDER (B.1.2 — StratifiedGroupKFold(shuffle=False) in
     common/crossval.py.run_kfold_cv splits on ``bundle.groups`` order/content
     only, not on seed, so identical order is what actually makes the two
     models' folds paired).

Exit 0 on full parity (set + order), exit 1 otherwise with a diagnostic.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CLASSIFIER_ROOT = _REPO_ROOT / "CLASSIFIER"
_BRAINTOKENGT_ROOT = _REPO_ROOT / "BRAINTOKENGT"
for _p in (str(_REPO_ROOT), str(_CLASSIFIER_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd  # noqa: E402

from BRAINTOKENGT.adapter import BrainTokenGTAdapter  # noqa: E402
from CLASSIFIER.adapters import get_adapter  # noqa: E402
from CLASSIFIER.common.crossval import Bundle  # noqa: E402
from CLASSIFIER.common.experiment_utils import build_config, load_experiment  # noqa: E402
from DATA.DELCODE.src.splitting.load_splits import splits_dir  # noqa: E402

GELSTM_EXP_ID = "recon-ablation-gelstm-pretrained-frozen-2to3v-seed42"
BRAINTOKENGT_EXP_ID = "braintokengt-delcode-whole-brain-repaired-fix-stabilized"

WB_DATA_ROOT = str(_REPO_ROOT / "DATA/DELCODE/__fc_wholebrain_sch200_flat__/matrices")
COHORTS_CSV = str(
    _REPO_ROOT
    / "DATA/DELCODE/__fc_wholebrain_sch200_flat__/metadata/cohorts_with_scans_on_disk.csv"
)


def _load_cv_pool() -> pd.DataFrame:
    """Identical to both notebooks' data cell: CV pool = train + val, test held out."""
    sd = splits_dir("downstream")
    train_df = pd.read_csv(sd / "train.csv")
    val_df = pd.read_csv(sd / "val.csv")
    return pd.concat([train_df, val_df], ignore_index=True)


def _require_matched_window(adapter, exp_id: str) -> None:
    if int(adapter.min_visits) != 2 or int(adapter.max_visits) != 3:
        raise ValueError(
            f"{exp_id!r} does not resolve to min_visits=2,max_visits=3 "
            f"(got min_visits={adapter.min_visits!r}, max_visits={adapter.max_visits!r}) "
            "— this is not the matched-cohort configuration this check is meant to verify."
        )


def build_gelstm_bundle(cv_pool_df: pd.DataFrame) -> Bundle:
    exp = load_experiment(_CLASSIFIER_ROOT / "experiments", GELSTM_EXP_ID)
    train_config = build_config(exp, _CLASSIFIER_ROOT)
    adapter_cls = get_adapter(exp.get("adapter") or exp["model"])
    adapter = adapter_cls(
        gaae_ckpt_path="unused (subject-parity check does not build a model)",
        gaae_hp={},
        train_config=train_config,
        data_root=WB_DATA_ROOT,
        cohorts_csv=COHORTS_CSV,
        device="cpu",
        rng=None,
    )
    _require_matched_window(adapter, GELSTM_EXP_ID)
    return adapter.prepare_data(cv_pool_df)


def build_braintokengt_bundle(cv_pool_df: pd.DataFrame) -> Bundle:
    exp = load_experiment(_BRAINTOKENGT_ROOT / "experiments", BRAINTOKENGT_EXP_ID)
    train_config = build_config(exp, _BRAINTOKENGT_ROOT)
    adapter = BrainTokenGTAdapter(
        gaae_ckpt_path="none (end-to-end)",
        gaae_hp={},
        train_config=train_config,
        data_root=WB_DATA_ROOT,
        cohorts_csv=COHORTS_CSV,
        device="cpu",
        rng=None,
    )
    _require_matched_window(adapter, BRAINTOKENGT_EXP_ID)
    return adapter.prepare_data(cv_pool_df)


def check_parity(gelstm_ids: list, braintokengt_ids: list) -> bool:
    """Print a diagnostic and return True iff the two ID sequences match exactly."""
    gset, bset = set(gelstm_ids), set(braintokengt_ids)
    if gset != bset:
        only_g = sorted(gset - bset)
        only_b = sorted(bset - gset)
        print("MISMATCH: cohort sets differ — the 'fair' claim is dead on arrival.")
        if only_g:
            print(f"  only in GELSTM ({len(only_g)}): {only_g}")
        if only_b:
            print(f"  only in BrainTokenGT ({len(only_b)}): {only_b}")
        return False

    if list(gelstm_ids) != list(braintokengt_ids):
        print(
            "MISMATCH: identical subject SET but different ORDER — "
            "StratifiedGroupKFold folds will not be paired between the two models."
        )
        for i, (g, b) in enumerate(zip(gelstm_ids, braintokengt_ids, strict=False)):
            if g != b:
                print(f"  first divergence at index {i}: GELSTM={g!r}  BrainTokenGT={b!r}")
                break
        return False

    print(
        f"OK: {len(gelstm_ids)} subjects — identical set AND identical order. "
        "Folds are paired by construction (StratifiedGroupKFold(shuffle=False))."
    )
    return True


def main() -> int:
    cv_pool_df = _load_cv_pool()
    gelstm_bundle = build_gelstm_bundle(cv_pool_df)
    braintokengt_bundle = build_braintokengt_bundle(cv_pool_df)

    print(f"GELSTM CV-pool subjects:       {len(gelstm_bundle.groups)}")
    print(f"BrainTokenGT CV-pool subjects: {len(braintokengt_bundle.groups)}")
    ok = check_parity(gelstm_bundle.groups, braintokengt_bundle.groups)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
