"""common/pooled_data.py — multi-cohort Bundle construction for pooled training.

`LongitudinalSubjectDataset` (`model/GELSTM/dataset.py`) reads one cohort's FC
matrices from one root directory with one visit-parsing convention (DELCODE's
nominal protocol months vs ADNI/OASIS-3's elapsed days — see
`common/visits.py`). A pooled ADNI+DELCODE training pool therefore needs one
dataset instance per cohort, built from the pooled split CSV's `cohort`
column, concatenated into a single `Bundle` (`common/crossval.py`).

See `DOCS/flipped/PLAN.md` Phase 1 and `DATA/manifest/build_pooled_assets.py`
for the split CSVs this consumes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

# Make `common.*` / `model.*` importable regardless of caller (mirrors
# model/GELSTM/dataset.py and adapters/__init__.py's sys.path setup — needed
# whether this module is reached as `CLASSIFIER.common.pooled_data` (tests,
# repo-root callers) or `common.pooled_data` (notebooks, which already put
# CLASSIFIER/ on sys.path).
_CLASSIFIER_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _CLASSIFIER_ROOT.parent
for _p in (str(_REPO_ROOT), str(_CLASSIFIER_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.crossval import Bundle  # noqa: E402
from model.GELSTM.dataset import LongitudinalSubjectDataset  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Cohort -> its FC matrices root. Each cohort's dataset is built against its
# own directory, never a merged one — the merged symlink farm under
# DATA/POOLED_ADNI_DELCODE/ exists only for the single-root GAAE static loader.
COHORT_ROOTS: Dict[str, str] = {
    "delcode": str(_REPO_ROOT / "DATA" / "DELCODE" / "__fc_wholebrain_sch200_flat__" / "matrices"),
    "adni": str(_REPO_ROOT / "DATA" / "ADNI" / "__fc_wholebrain_sch200_flat__" / "matrices"),
    "oasis3": str(_REPO_ROOT / "DATA" / "OASIS3" / "__fc_wholebrain_sch200_flat__" / "matrices"),
}

# Which allow-list column is native to each cohort's visit-time convention
# (common/visits.py: DELCODE nominal protocol months vs ADNI/OASIS-3 elapsed
# days). The other column must be dropped before constructing the per-cohort
# dataset, or LongitudinalSubjectDataset's first-match column pick
# (dataset.py's allow_col loop) could silently pick up an empty/foreign
# column and disable the leakage filter instead of raising.
_NATIVE_ALLOW_COLUMN: Dict[str, str] = {
    "delcode": "allowed_months",
    "adni": "allowed_days",
    "oasis3": "allowed_days",
}
_ALLOW_COLUMNS = {"allowed_days", "allowed_months"}


def build_multicohort_bundle(
    df: pd.DataFrame,
    *,
    cohort_roots: Optional[Dict[str, str]] = None,
    **dataset_kwargs,
) -> Bundle:
    """Build one Bundle spanning every cohort present in ``df['cohort']``.

    Splits ``df`` by its ``cohort`` column, builds one
    ``LongitudinalSubjectDataset`` per cohort (that cohort's root + native
    allow-list column + ``cohort=`` tag for visit-time parsing), and
    concatenates the resulting items into a single ``Bundle``. Each item gets
    a ``cohort`` key added (used by the TFGN cohort-decoding probe;
    ``LongitudinalSubjectDataset`` itself never emits one).

    A ``df`` without a ``cohort`` column is not multi-cohort input — callers
    should fall through to the existing single-cohort path in that case
    (this function raises rather than guessing a default).

    ``dataset_kwargs`` are forwarded to every per-cohort
    ``LongitudinalSubjectDataset`` (``adjacency_k``, ``file_variant``,
    ``min_visits``, ``max_visits``, ...); ``cohorts_csv`` and ``cohort`` must
    not be passed here — they are resolved per cohort internally.
    """
    if "cohort" not in df.columns:
        raise ValueError(
            "build_multicohort_bundle requires a 'cohort' column; the caller "
            "should route single-cohort frames through the existing "
            "single-cohort dataset construction instead."
        )
    for forbidden in ("cohort", "cohorts_csv"):
        if forbidden in dataset_kwargs:
            raise ValueError(
                f"build_multicohort_bundle resolves {forbidden!r} per cohort; "
                "do not pass it in dataset_kwargs."
            )
    roots = cohort_roots or COHORT_ROOTS

    all_labels: list = []
    all_groups: list = []
    all_items: list = []

    for cohort_name, sub_df in df.groupby("cohort", sort=True):
        cohort_key = str(cohort_name).lower()
        if cohort_key not in roots:
            raise ValueError(f"Unknown cohort {cohort_name!r}; known roots: {sorted(roots)}")

        native_col = _NATIVE_ALLOW_COLUMN[cohort_key]
        sub_df = sub_df.drop(columns=[c for c in _ALLOW_COLUMNS if c in sub_df.columns and c != native_col])
        if native_col in sub_df.columns:
            populated = sub_df[native_col].astype(str).str.len().gt(0)
            if len(sub_df) > 0 and not populated.any():
                raise ValueError(
                    f"cohort={cohort_key!r}: native allow-list column {native_col!r} is "
                    "entirely empty for every subject in this frame — refusing to build "
                    "a dataset that would silently drop the post-conversion leakage "
                    "filter. Check the pooled split CSV."
                )

        ds = LongitudinalSubjectDataset(
            roots[cohort_key],
            sub_df,
            cohort=cohort_key,
            **dataset_kwargs,
        )
        items = [ds[i] for i in range(len(ds))]
        for item in items:
            item["cohort"] = cohort_key
        all_labels.extend(ds.get_labels())
        all_groups.extend(ds.get_subject_ids())
        all_items.extend(items)

    return Bundle(all_labels, all_groups, all_items)
