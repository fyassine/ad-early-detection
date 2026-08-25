"""One-off Tier-4 frozen read for two arms outside the pre-registered primary/
secondary set: the matched-window winner (tfgn-w3-gelstm-frozen-pooled) and the
adversarial-conditioning escalation (tfgn-s1-advcohort-pooled).

Reuses common.frozen_read.score_frozen_split exactly as
notebooks/COMPARISON/COMPARISON_TEMPORAL_FIRST_LADDER.ipynb's run_frozen_reads
does, so the estimate is produced by the identical code path as every other
Tier-4 read -- this is a targeted CLI invocation of that path, not a new one.

DOCS/temporal-first-ablation.md's Tier-4 protocol did not pre-register either of
these two reads. This script exists because the user explicitly asked for both
numbers after being told that; it deliberately does NOT write the results into
any doc -- that recording step is a separate, explicit action.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path("/mnt/e/fyassine/ad-early-detection")
MODEL_ROOT = REPO_ROOT / "CLASSIFIER"
for _p in (str(REPO_ROOT), str(MODEL_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from adapters.explain import resolve_source_run  # noqa: E402
from common.frozen_read import score_frozen_split  # noqa: E402

SEEDS = [42, 43, 44, 45]

ADAPTER_KEY_MAP = {
    "tfgnclassifier": "tfgn",
    "logregdriftadapter": "logregdrift",
    "gelstmclassifier": "gelstm",
    "braintokengtclassifier": "braintokengt",
}


def load_summary(run_dir: Path):
    path = run_dir / "run_summary.json"
    return json.loads(path.read_text()) if path.is_file() else None


def run_frozen_reads(id_prefix: str, label: str) -> pd.DataFrame:
    gaae_hp_path = MODEL_ROOT / "configs" / "gaae_delcode_whole_brain.json"
    gaae_hp = json.loads(gaae_hp_path.read_text()) if gaae_hp_path.is_file() else {}
    pooled_dir = REPO_ROOT / "DATA" / "POOLED_ADNI_DELCODE"
    in_domain_test_df = pd.read_csv(pooled_dir / "SPLITS" / "downstream" / "test.csv")
    oasis_splits = REPO_ROOT / "DATA" / "OASIS3" / "__metadata__" / "SPLITS" / "downstream"
    oasis_test_df = pd.concat(
        [pd.read_csv(oasis_splits / f"{s}.csv") for s in ("train", "val", "test")],
        ignore_index=True,
    )
    oasis_test_df["cohort"] = "oasis3"

    results = {}
    for seed in SEEDS:
        exp_id = f"{id_prefix}-seed{seed}"
        run_dir = resolve_source_run(exp_id, classifier_root=MODEL_ROOT)
        summary = load_summary(run_dir)
        if summary is None:
            raise FileNotFoundError(f"{exp_id}: no run_summary.json yet -- not ready for a frozen read.")
        adapter_key = str(summary.get("model_config", {}).get("model_type", "")).lower()
        adapter_key = ADAPTER_KEY_MAP.get(adapter_key, adapter_key)
        common_kwargs = dict(
            adapter_key=adapter_key,
            data_root=str(pooled_dir / "__fc_wholebrain_sch200_flat__" / "matrices"),
            cohorts_csv=None,
            gaae_ckpt_path=summary.get("gaae_checkpoint") or "",
            gaae_hp=gaae_hp,
            device="cpu",
        )
        test_metrics = score_frozen_split(run_dir, in_domain_test_df, record_as="test", **common_kwargs)
        ext_metrics = score_frozen_split(
            run_dir, oasis_test_df, record_as="external", cohort="oasis3", **common_kwargs
        )
        results[exp_id] = {"test_auc": test_metrics["auc"], "ext_oasis3_auc": ext_metrics["auc"]}
        print(f"[{label}] {exp_id}: test_auc={test_metrics['auc']:.4f}  ext_oasis3_auc={ext_metrics['auc']:.4f}")

    frozen_df = pd.DataFrame(results).T
    print()
    print(f"[{label}] Frozen reads across seeds ({id_prefix}):")
    print(frozen_df)
    print()
    print(f"[{label}] In-domain test AUC: {frozen_df['test_auc'].mean():.4f} +/- {frozen_df['test_auc'].std():.4f}")
    print(f"[{label}] OASIS-3 AUC:        {frozen_df['ext_oasis3_auc'].mean():.4f} +/- {frozen_df['ext_oasis3_auc'].std():.4f}")
    return frozen_df


if __name__ == "__main__":
    print("=" * 70)
    run_frozen_reads("tfgn-w3-gelstm-frozen-pooled", "AD-HOC (matched-window winner, not pre-registered)")
    print()
    print("=" * 70)
    run_frozen_reads("tfgn-s1-advcohort-pooled", "AD-HOC (rejected adversarial arm, not pre-registered)")
