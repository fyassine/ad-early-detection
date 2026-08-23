"""Build the pooled ADNI+DELCODE training assets for the temporal-first (TFGN)
ablation: `DOCS/flipped/PLAN.md` Phase 1 / `DOCS/temporal-first-ablation.md`.

Four artefacts, each idempotent and re-runnable:

1. ``DATA/ADNI/__metadata__/SPLITS/pretrain/{train,val,test}.csv`` — ADNI's
   missing pretrain split, built with the same leakage rule as
   ``DATA/DELCODE/src/splitting/create_pretrain_data_splits.py``: downstream
   val/test subjects are forced into pretrain val/test, so
   ``pretrain train ∩ downstream {val,test} = ∅``. Unlike DELCODE's ADNI has no
   healthy/AD-only pool — every manifest row already carries a
   converter/stable label — so the "available for pretrain" pool is simply
   every ADNI subject with ``>= 1`` session (``min_sessions=1``), which is a
   superset of the downstream ``min_sessions=2`` pool.
2. ``DATA/POOLED_ADNI_DELCODE/SPLITS/pretrain/{train,val,test}.csv`` —
   concatenation of DELCODE's and ADNI's pretrain splits, harmonised to the
   ``Pseudonym,diagnosis,sex,age,n_scans`` schema
   ``GraphDatasetInMemoryFiltered`` (``CLASSIFIER/model/GAAE/dataset.py``)
   requires verbatim — its ``filter_csv_path``/``patient_info_path`` loader
   hardcodes the column name ``Pseudonym``, so this is the one pooled CSV that
   does NOT use ``subject_id``.
3. ``DATA/POOLED_ADNI_DELCODE/SPLITS/downstream/{train,val,test}.csv`` —
   union of DELCODE's and ADNI's downstream splits, harmonised to
   ``subject_id,cohort,converter_status,sex,age,n_scans,allowed_days,
   allowed_months`` (``LongitudinalSubjectDataset`` accepts either
   ``subject_id`` or ``Pseudonym``, so ``subject_id`` is used here for
   consistency with ADNI/OASIS-3's native column). Each row keeps only its
   cohort-native allow-list column populated; ``--min-visits`` (default 2)
   drops subjects below the floor.
4. ``DATA/POOLED_ADNI_DELCODE/__fc_wholebrain_sch200_flat__/matrices/`` —
   a symlink farm into both cohorts' ``.npz`` files, needed only by the
   single-directory static (GAAE) pretraining loader. Subject-id prefixes are
   disjoint (``sub-ADNI...`` / ``sub-<pseudonym>...``) so one glob root works.

The downstream *classification* path (TFGN/GELSTM adapters via
``CLASSIFIER.common.pooled_data.build_multicohort_bundle``) reads each
cohort's FC matrices from its own on-disk root directly — it does not need
the symlink farm, which exists solely for the GAAE static loader's
single-``root`` constructor.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from DATA.manifest.build_cohort_splits import build_subject_table

_REPO_ROOT = Path(__file__).resolve().parents[2]

ADNI_MANIFEST = _REPO_ROOT / "DATA" / "ADNI" / "__metadata__" / "cohort_manifest.csv"
ADNI_DEMOGRAPHICS = _REPO_ROOT / "DATA" / "ADNI" / "__metadata__" / "adni_demographics.csv"
ADNI_DOWNSTREAM_DIR = _REPO_ROOT / "DATA" / "ADNI" / "__metadata__" / "SPLITS" / "downstream"
ADNI_PRETRAIN_DIR = _REPO_ROOT / "DATA" / "ADNI" / "__metadata__" / "SPLITS" / "pretrain"
ADNI_MATRICES_DIR = _REPO_ROOT / "DATA" / "ADNI" / "__fc_wholebrain_sch200_flat__" / "matrices"

DELCODE_DOWNSTREAM_DIR = _REPO_ROOT / "DATA" / "DELCODE" / "__metadata__" / "SPLITS" / "downstream"
DELCODE_PRETRAIN_DIR = _REPO_ROOT / "DATA" / "DELCODE" / "__metadata__" / "SPLITS" / "pretrain"
DELCODE_MATRICES_DIR = _REPO_ROOT / "DATA" / "DELCODE" / "__fc_wholebrain_sch200_flat__" / "matrices"

# Not `__pooled__` — SHARED.provenance.region_from_data_root scans path parts for
# the first `__<name>__`-shaped component to identify the dataset directory
# (e.g. `__fc_wholebrain_sch200_flat__`); a double-underscore-wrapped pooled root
# would match that regex itself and shadow the real one, leaving region/atlas as
# None in every pooled run's metadata.
POOLED_ROOT = _REPO_ROOT / "DATA" / "POOLED_ADNI_DELCODE"
POOLED_DOWNSTREAM_DIR = POOLED_ROOT / "SPLITS" / "downstream"
POOLED_PRETRAIN_DIR = POOLED_ROOT / "SPLITS" / "pretrain"
POOLED_MATRICES_DIR = POOLED_ROOT / "__fc_wholebrain_sch200_flat__" / "matrices"

POOLED_DOWNSTREAM_COLUMNS = [
    "subject_id",
    "cohort",
    "converter_status",
    "sex",
    "age",
    "n_scans",
    "allowed_days",
    "allowed_months",
]
POOLED_PRETRAIN_COLUMNS = ["Pseudonym", "diagnosis", "sex", "age", "n_scans"]


# ---------------------------------------------------------------------------
# 1. ADNI's missing pretrain split
# ---------------------------------------------------------------------------


def _stratified_split_by_scans(scan_map: dict, test_size: float, random_state: int):
    """Mirrors ``create_pretrain_data_splits._stratified_split_by_scans``."""
    if len(scan_map) < 2:
        return list(scan_map.keys()), []
    ids = sorted(scan_map, key=lambda p: scan_map[p], reverse=True)
    actual_test = min(test_size, (len(ids) - 1) / len(ids))
    return train_test_split(ids, test_size=actual_test, random_state=random_state)


def build_adni_pretrain_splits(*, seed: int = 42) -> dict[str, pd.DataFrame]:
    """ADNI pretrain split with the same leakage rule as DELCODE's builder.

    ``available`` = every ADNI subject NOT already reserved by the downstream
    val/test split (``min_sessions=1``, so this includes the 76 single-session
    subjects the downstream ``min_sessions=2`` split drops entirely). Reserved
    subjects are forced into the matching pretrain split; the rest are
    stratified 60/20/20 by label, mirroring
    ``create_pretrain_data_splits.py``'s ``_stratified_split_by_scans`` policy.
    """
    manifest = pd.read_csv(ADNI_MANIFEST)
    demographics = pd.read_csv(ADNI_DEMOGRAPHICS)
    subjects = build_subject_table(manifest, demographics, min_sessions=1)

    downstream_val_ids = set(pd.read_csv(ADNI_DOWNSTREAM_DIR / "val.csv")["subject_id"].astype(str))
    downstream_test_ids = set(pd.read_csv(ADNI_DOWNSTREAM_DIR / "test.csv")["subject_id"].astype(str))

    reserved_val = subjects[subjects["subject_id"].isin(downstream_val_ids)]
    reserved_test = subjects[subjects["subject_id"].isin(downstream_test_ids)]
    available = subjects[~subjects["subject_id"].isin(downstream_val_ids | downstream_test_ids)]

    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = [reserved_val]
    test_parts: list[pd.DataFrame] = [reserved_test]

    for _label, group_df in available.groupby("label"):
        scan_map = dict(zip(group_df["subject_id"], group_df["n_scans"], strict=False))
        ids = list(scan_map.keys())
        if len(ids) >= 5:
            trainval_ids, test_ids = _stratified_split_by_scans(scan_map, 0.20, seed)
            trainval_map = {p: scan_map[p] for p in trainval_ids}
            train_ids, val_ids = _stratified_split_by_scans(trainval_map, 0.25, seed)
        elif len(ids) >= 2:
            train_ids, val_ids = _stratified_split_by_scans(scan_map, 0.33, seed)
            test_ids = []
        else:
            train_ids, val_ids, test_ids = ids, [], []

        train_parts.append(group_df[group_df["subject_id"].isin(train_ids)])
        val_parts.append(group_df[group_df["subject_id"].isin(val_ids)])
        test_parts.append(group_df[group_df["subject_id"].isin(test_ids)])

    empty = pd.DataFrame(columns=subjects.columns)
    train = pd.concat(train_parts, ignore_index=True) if train_parts else empty
    val = pd.concat(val_parts, ignore_index=True) if val_parts else empty
    test = pd.concat(test_parts, ignore_index=True) if test_parts else empty

    train_ids_out, val_ids_out, test_ids_out = (
        set(train["subject_id"]),
        set(val["subject_id"]),
        set(test["subject_id"]),
    )
    assert not (train_ids_out & downstream_val_ids), "LEAK: downstream val subjects in pretrain train"
    assert not (train_ids_out & downstream_test_ids), "LEAK: downstream test subjects in pretrain train"
    assert not (val_ids_out & downstream_test_ids), "LEAK: downstream test subjects in pretrain val"
    assert downstream_val_ids.issubset(val_ids_out), "downstream val not fully covered by pretrain val"
    assert downstream_test_ids.issubset(test_ids_out), "downstream test not fully covered by pretrain test"

    return {"train": train, "val": val, "test": test}


# ---------------------------------------------------------------------------
# 2. Pooled pretrain split (Pseudonym schema — GAAE static loader contract)
# ---------------------------------------------------------------------------


def _delcode_pretrain_to_pooled(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in POOLED_PRETRAIN_COLUMNS:
        if col not in out.columns:
            raise ValueError(f"DELCODE pretrain split missing expected column {col!r}")
    return out[POOLED_PRETRAIN_COLUMNS]


def _adni_pretrain_to_pooled(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "Pseudonym": df["subject_id"].astype(str),
            "diagnosis": df["label"],
            "sex": df["sex"],
            "age": df["age"],
            "n_scans": df["n_scans"],
        }
    )
    return out[POOLED_PRETRAIN_COLUMNS]


def build_pooled_pretrain_splits(*, seed: int = 42) -> dict[str, pd.DataFrame]:
    if not (DELCODE_PRETRAIN_DIR / "train.csv").exists():
        raise FileNotFoundError(
            f"{DELCODE_PRETRAIN_DIR} is missing — run "
            "DATA/DELCODE/src/splitting/create_pretrain_data_splits.py first."
        )
    if not (ADNI_PRETRAIN_DIR / "train.csv").exists():
        build_and_write_adni_pretrain_splits(seed=seed)

    pooled: dict[str, pd.DataFrame] = {}
    for split_name in ("train", "val", "test"):
        delcode_df = _delcode_pretrain_to_pooled(pd.read_csv(DELCODE_PRETRAIN_DIR / f"{split_name}.csv"))
        adni_df = _adni_pretrain_to_pooled(pd.read_csv(ADNI_PRETRAIN_DIR / f"{split_name}.csv"))
        pooled[split_name] = pd.concat([delcode_df, adni_df], ignore_index=True)

    train_ids, val_ids, test_ids = (
        set(pooled["train"]["Pseudonym"]),
        set(pooled["val"]["Pseudonym"]),
        set(pooled["test"]["Pseudonym"]),
    )
    assert not (train_ids & val_ids), "LEAK: subject in both pooled pretrain train and val"
    assert not (train_ids & test_ids), "LEAK: subject in both pooled pretrain train and test"
    assert not (val_ids & test_ids), "LEAK: subject in both pooled pretrain val and test"

    return pooled


# ---------------------------------------------------------------------------
# 3. Pooled downstream split (subject_id schema — LongitudinalSubjectDataset)
# ---------------------------------------------------------------------------


def _delcode_downstream_to_pooled(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "subject_id": df["Pseudonym"].astype(str),
            "cohort": "delcode",
            "converter_status": df["converter_status"],
            "sex": df["sex"],
            "age": df["age"],
            "n_scans": df["n_scans"],
            "allowed_days": "",
            "allowed_months": df.get("allowed_months", ""),
        }
    )
    return out[POOLED_DOWNSTREAM_COLUMNS]


def _dayscoded_downstream_to_pooled(df: pd.DataFrame, *, cohort: str) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "subject_id": df["subject_id"].astype(str),
            "cohort": cohort,
            "converter_status": df["converter_status"],
            "sex": df["sex"],
            "age": df["age"],
            "n_scans": df["n_scans"],
            "allowed_days": df.get("allowed_days", ""),
            "allowed_months": "",
        }
    )
    return out[POOLED_DOWNSTREAM_COLUMNS]


def build_pooled_downstream_splits(*, min_visits: int = 2) -> dict[str, pd.DataFrame]:
    pooled: dict[str, pd.DataFrame] = {}
    for split_name in ("train", "val", "test"):
        delcode_df = _delcode_downstream_to_pooled(pd.read_csv(DELCODE_DOWNSTREAM_DIR / f"{split_name}.csv"))
        adni_df = _dayscoded_downstream_to_pooled(
            pd.read_csv(ADNI_DOWNSTREAM_DIR / f"{split_name}.csv"), cohort="adni"
        )
        merged = pd.concat([delcode_df, adni_df], ignore_index=True)
        n_before = len(merged)
        merged = merged[merged["n_scans"] >= min_visits].reset_index(drop=True)
        n_dropped = n_before - len(merged)
        if n_dropped:
            print(f"  [{split_name}] dropped {n_dropped} subject(s) below min_visits={min_visits}")
        pooled[split_name] = merged

    train_ids, val_ids, test_ids = (
        set(pooled["train"]["subject_id"]),
        set(pooled["val"]["subject_id"]),
        set(pooled["test"]["subject_id"]),
    )
    assert not (train_ids & val_ids), "LEAK: subject in both pooled downstream train and val"
    assert not (train_ids & test_ids), "LEAK: subject in both pooled downstream train and test"
    assert not (val_ids & test_ids), "LEAK: subject in both pooled downstream val and test"

    for split_name, df in pooled.items():
        both_populated = df["allowed_days"].astype(str).str.len().gt(0) & df["allowed_months"].astype(
            str
        ).str.len().gt(0)
        if both_populated.any():
            raise ValueError(
                f"{split_name}: {both_populated.sum()} row(s) carry both allowed_days and "
                "allowed_months — the leakage filter would be ambiguous."
            )

    return pooled


# ---------------------------------------------------------------------------
# 4. Symlink farm (static/GAAE pretraining path only)
# ---------------------------------------------------------------------------


def build_symlink_farm(*, dry_run: bool = False) -> tuple[int, int]:
    """Symlink every DELCODE + ADNI ``.npz`` into one flat pooled directory.

    Idempotent: existing correct symlinks are left alone; a stale symlink
    pointing elsewhere is refused (fail loud) rather than silently replaced.
    Returns ``(n_created, n_already_present)``.
    """
    POOLED_MATRICES_DIR.mkdir(parents=True, exist_ok=True)
    n_created = 0
    n_present = 0
    for source_dir in (DELCODE_MATRICES_DIR, ADNI_MATRICES_DIR):
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Source matrices dir not found: {source_dir}")
        for src in sorted(source_dir.glob("*.npz")):
            dst = POOLED_MATRICES_DIR / src.name
            if dst.exists() or dst.is_symlink():
                if not dst.is_symlink() or os.path.realpath(dst) != os.path.realpath(src):
                    raise ValueError(f"{dst} exists and does not point at {src}; refusing to overwrite.")
                n_present += 1
                continue
            if not dry_run:
                dst.symlink_to(src)
            n_created += 1
    return n_created, n_present


# ---------------------------------------------------------------------------
# Orchestration / CLI
# ---------------------------------------------------------------------------


def build_and_write_adni_pretrain_splits(*, seed: int = 42) -> dict[str, pd.DataFrame]:
    splits = build_adni_pretrain_splits(seed=seed)
    ADNI_PRETRAIN_DIR.mkdir(parents=True, exist_ok=True)
    lines = [f"adni pretrain splits (seed={seed}, min_sessions=1)", ""]
    for split_name in ("train", "val", "test"):
        df = splits[split_name][["subject_id", "label", "sex", "age", "n_scans"]]
        df.to_csv(ADNI_PRETRAIN_DIR / f"{split_name}.csv", index=False)
        counts = dict(df["label"].value_counts()) if len(df) else {}
        line = f"{split_name:5}: {len(df):3}  {counts}"
        print(f"  {line}")
        lines.append(line)
    (ADNI_PRETRAIN_DIR / "split_report.txt").write_text("\n".join(lines) + "\n")
    return splits


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-visits", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-symlinks", action="store_true")
    args = parser.parse_args(argv)

    print("=== 1. ADNI pretrain split ===")
    build_and_write_adni_pretrain_splits(seed=args.seed)

    print("\n=== 2. Pooled pretrain split (Pseudonym schema) ===")
    POOLED_PRETRAIN_DIR.mkdir(parents=True, exist_ok=True)
    pretrain_splits = build_pooled_pretrain_splits(seed=args.seed)
    for split_name, df in pretrain_splits.items():
        df.to_csv(POOLED_PRETRAIN_DIR / f"{split_name}.csv", index=False)
        print(f"  {split_name:5}: {len(df):4}")

    print(f"\n=== 3. Pooled downstream split (min_visits={args.min_visits}) ===")
    POOLED_DOWNSTREAM_DIR.mkdir(parents=True, exist_ok=True)
    downstream_splits = build_pooled_downstream_splits(min_visits=args.min_visits)
    report_lines = [f"pooled adni+delcode downstream splits (min_visits={args.min_visits})", ""]
    for split_name, df in downstream_splits.items():
        df.to_csv(POOLED_DOWNSTREAM_DIR / f"{split_name}.csv", index=False)
        by_cohort = dict(df["cohort"].value_counts())
        by_label = dict(df["converter_status"].value_counts())
        line = f"{split_name:5}: {len(df):4}  cohort={by_cohort}  converter_status={by_label}"
        print(f"  {line}")
        report_lines.append(line)
    (POOLED_DOWNSTREAM_DIR / "split_report.txt").write_text("\n".join(report_lines) + "\n")

    if not args.skip_symlinks:
        print("\n=== 4. Symlink farm ===")
        n_created, n_present = build_symlink_farm()
        print(f"  created={n_created}  already_present={n_present}  -> {POOLED_MATRICES_DIR}")

    print(f"\nDone. Pooled assets under {POOLED_ROOT}")


if __name__ == "__main__":
    main()
