"""Build DATA/{ADNI,OASIS3}/__metadata__/SPLITS/downstream/{train,val,test}.csv.

A.4 (`DOCS/timeline/MASTER_PLAN.md` §3): subject-level,
stratified-by-label, 60/20/20 splits from the cohort manifest, mirroring
DELCODE's protocol in
`DATA/DELCODE/src/splitting/create_downstream_data_splits.py::_stratified_split`
(same two-stage `train_test_split` call, same seed policy) — reimplemented here
rather than imported, so that module (frozen for the A.2 DELCODE reproduction
gate) is never touched by this work.

Deliberately **not** aliased to DELCODE's `Pseudonym`/`allowed_months` column
names: ADNI/OASIS-3 use day-coded sessions, not nominal protocol months, and
GELSTM's `diagnosis in {mci, converter}` filter doesn't apply to a cohort with
no `mci` label. Consumer-side generalization (id column, label vocabulary,
`allowed_days` vs `allowed_months`) is A.1-wiring's remaining scope, not this
module's — see A.4 in the comparison plan.

A row's eligibility for a split requires ``fc_path`` to be populated, i.e. A.3
extraction to have already run for that session — same coupling as DELCODE's
splitter, which globs its FC matrices directory
(``create_downstream_data_splits.py::_on_disk_months``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

_REPO_ROOT = Path(__file__).resolve().parents[2]

_COHORT_PATHS = {
    "adni": {
        "manifest": _REPO_ROOT / "DATA" / "ADNI" / "__metadata__" / "cohort_manifest.csv",
        "demographics": _REPO_ROOT / "DATA" / "ADNI" / "__metadata__" / "adni_demographics.csv",
        "output_dir": _REPO_ROOT / "DATA" / "ADNI" / "__metadata__" / "SPLITS" / "downstream",
    },
    "oasis3": {
        "manifest": _REPO_ROOT / "DATA" / "OASIS3" / "__metadata__" / "cohort_manifest.csv",
        "demographics": _REPO_ROOT / "DATA" / "OASIS3" / "__metadata__" / "oasis3_demographics.csv",
        "output_dir": _REPO_ROOT / "DATA" / "OASIS3" / "__metadata__" / "SPLITS" / "downstream",
    },
}

SPLIT_COLUMNS = ["subject_id", "label", "converter_status", "sex", "age", "n_scans", "allowed_days"]

_ALLOWED_LABELS = {"converter", "stable"}


def _stratified_split(ids: list, test_size: float, random_state: int):
    """Same policy as DELCODE's ``create_downstream_data_splits._stratified_split``."""
    if len(ids) < 2:
        return ids, []
    actual_test = min(test_size, (len(ids) - 1) / len(ids))
    return train_test_split(ids, test_size=actual_test, random_state=random_state)


def build_subject_table(
    manifest: pd.DataFrame, demographics: pd.DataFrame, *, min_sessions: int
) -> pd.DataFrame:
    """One row per eligible subject: label, demographics, session count, allowed days."""
    unrecognized = set(manifest["label"].dropna().unique()) - _ALLOWED_LABELS
    if unrecognized:
        raise ValueError(
            f"Unrecognized label value(s) in manifest: {sorted(unrecognized)}; "
            f"expected only {_ALLOWED_LABELS} or null."
        )

    labeled = manifest.dropna(subset=["label"])
    n_dropped_unlabeled = len(manifest) - len(labeled)
    if n_dropped_unlabeled:
        print(f"  dropping {n_dropped_unlabeled} row(s) with no label (neither converter nor stable)")

    eligible = labeled[labeled["fc_path"].notna()]
    n_dropped_no_fc = len(labeled) - len(eligible)
    if n_dropped_no_fc:
        print(f"  dropping {n_dropped_no_fc} row(s) with no fc_path (A.3 extraction not yet run for them)")

    rows: list[dict] = []
    for subject_id, group in eligible.groupby("subject_id"):
        if len(group) < min_sessions:
            continue
        labels = set(group["label"].unique())
        if len(labels) > 1:
            raise ValueError(f"Subject {subject_id} has conflicting labels across sessions: {labels}")
        days = sorted(int(d) for d in group["days_from_baseline"])
        rows.append(
            {
                "subject_id": subject_id,
                "label": next(iter(labels)),
                "n_scans": len(group),
                "allowed_days": ";".join(str(d) for d in days),
            }
        )

    subjects = pd.DataFrame(rows, columns=["subject_id", "label", "n_scans", "allowed_days"])
    subjects["converter_status"] = (subjects["label"] == "converter").astype(int)

    merged = subjects.merge(demographics[["subject_id", "sex", "age_at_baseline"]], on="subject_id", how="left")
    missing_demog = merged[merged["sex"].isna()]
    if not missing_demog.empty:
        raise ValueError(
            f"{len(missing_demog)} eligible subject(s) missing demographics: "
            f"{sorted(missing_demog['subject_id'])[:20]}. Run DATA.manifest.demographics first."
        )
    merged = merged.rename(columns={"age_at_baseline": "age"})
    return merged[SPLIT_COLUMNS]


def split_subjects(subjects: pd.DataFrame, *, seed: int) -> dict[str, pd.DataFrame]:
    train_parts, val_parts, test_parts = [], [], []
    for _label, group_df in subjects.groupby("label"):
        ids = group_df["subject_id"].tolist()
        train_ids, temp = _stratified_split(ids, test_size=0.4, random_state=seed)
        val_ids, test_ids = _stratified_split(temp, test_size=0.5, random_state=seed)
        train_parts.append(group_df[group_df["subject_id"].isin(train_ids)])
        val_parts.append(group_df[group_df["subject_id"].isin(val_ids)])
        test_parts.append(group_df[group_df["subject_id"].isin(test_ids)])

    empty = pd.DataFrame(columns=subjects.columns)
    return {
        "train": pd.concat(train_parts, ignore_index=True) if train_parts else empty,
        "val": pd.concat(val_parts, ignore_index=True) if val_parts else empty,
        "test": pd.concat(test_parts, ignore_index=True) if test_parts else empty,
    }


def build_cohort_splits(
    cohort: str,
    *,
    manifest_csv: Path | None = None,
    demographics_csv: Path | None = None,
    seed: int = 42,
    min_sessions: int = 2,
) -> dict[str, pd.DataFrame]:
    paths = _COHORT_PATHS[cohort]
    manifest = pd.read_csv(manifest_csv or paths["manifest"])
    demographics = pd.read_csv(demographics_csv or paths["demographics"])

    subjects = build_subject_table(manifest, demographics, min_sessions=min_sessions)
    return split_subjects(subjects, seed=seed)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=("adni", "oasis3"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-sessions", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    print(f"{args.cohort}:")
    splits = build_cohort_splits(args.cohort, seed=args.seed, min_sessions=args.min_sessions)

    output_dir = args.output_dir or _COHORT_PATHS[args.cohort]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = [f"{args.cohort} downstream splits (seed={args.seed}, min_sessions={args.min_sessions})", ""]
    for split_name in ("train", "val", "test"):
        df = splits[split_name]
        df.to_csv(output_dir / f"{split_name}.csv", index=False)
        counts = dict(df["label"].value_counts()) if len(df) else {}
        line = f"{split_name:5}: {len(df):3}  {counts}"
        print(f"  {line}")
        lines.append(line)
    (output_dir / "split_report.txt").write_text("\n".join(lines) + "\n")
    print(f"  -> {output_dir}")


if __name__ == "__main__":
    main()
