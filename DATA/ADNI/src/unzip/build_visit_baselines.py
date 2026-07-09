#!/usr/bin/env python3
"""
build_visit_baselines.py
=========================
Computes each ADNI subject's baseline visit date from the DXSUM diagnosis
summary CSV, so that later scans can be expressed as days-since-baseline —
the same "ses-d<NNNN>" convention already used by DATA/OASIS3/__bold_and_smri__.

For each subject (PTID): the baseline date is the EXAMDATE of the VISCODE
"bl" row; if that row is missing or has no date, falls back to "4_bl", then
to the earliest EXAMDATE across all of that subject's rows. The fallback
tier actually used is recorded per subject so it stays auditable rather
than silently degrading.

Usage
-----
    python build_visit_baselines.py
"""

from __future__ import annotations

import csv
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2]  # .../ADNI/
DEFAULT_DXSUM_CSV = DATA_DIR / "__metadata__" / "All_Subjects_DXSUM_12May2026.csv"
DEFAULT_OUTPUT_CSV = DATA_DIR / "__metadata__" / "adni_visit_baselines.csv"

BASELINE_VISCODES = ("bl", "4_bl")

OUTPUT_COLUMNS = ["subject_id", "baseline_date", "baseline_source"]


def build_baselines(dxsum_csv: Path) -> list[dict[str, str]]:
    """
    Returns one row per subject_id (PTID) with the resolved baseline_date and
    the baseline_source tier that produced it: "bl", "4_bl", or
    "earliest_examdate" (fallback when no bl/4_bl row has a date).
    """
    if not dxsum_csv.exists():
        raise FileNotFoundError(dxsum_csv)

    rows_by_subject: dict[str, list[tuple[str, str]]] = {}
    with dxsum_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subject_id = row["PTID"].strip()
            examdate = row["EXAMDATE"].strip()
            if not subject_id or not examdate:
                continue
            rows_by_subject.setdefault(subject_id, []).append((row["VISCODE"].strip(), examdate))

    results: list[dict[str, str]] = []
    for subject_id, visits in sorted(rows_by_subject.items()):
        baseline_date = None
        baseline_source = None
        for viscode in BASELINE_VISCODES:
            candidates = [date for vc, date in visits if vc == viscode]
            if candidates:
                baseline_date = min(candidates)
                baseline_source = viscode
                break
        if baseline_date is None:
            baseline_date = min(date for _, date in visits)
            baseline_source = "earliest_examdate"

        results.append(
            {
                "subject_id": subject_id,
                "baseline_date": baseline_date,
                "baseline_source": baseline_source,
            }
        )
    return results


def write_baselines(rows: list[dict[str, str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dxsum-csv", default=str(DEFAULT_DXSUM_CSV))
    p.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    args = p.parse_args()

    rows = build_baselines(Path(args.dxsum_csv))
    write_baselines(rows, Path(args.output_csv))

    n_bl = sum(1 for r in rows if r["baseline_source"] == "bl")
    n_4bl = sum(1 for r in rows if r["baseline_source"] == "4_bl")
    n_fallback = sum(1 for r in rows if r["baseline_source"] == "earliest_examdate")
    print(f"Wrote {len(rows)} subject baselines -> {args.output_csv}")
    print(f"  bl: {n_bl}  4_bl: {n_4bl}  earliest_examdate fallback: {n_fallback}")


if __name__ == "__main__":
    main()
