#!/usr/bin/env python3
"""One-off diagnostic: can sex be decoded from baseline FC matrices, per cohort?

Not part of the model/adapter pipeline. Standalone script, run once, result
recorded in DOCS/timeline/MASTER_PLAN.md §4 (Phase 1) and not wired into any
experiment registry.

Sex is reliably decodable from resting-state FC across cohorts in the
literature. This is a label-free positive control: it uses neither the
conversion label nor any cross-cohort comparison, so a null result here
isolates "the extracted matrices carry no signal" from "conversion doesn't
transfer across cohorts."
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]

COHORTS = {
    "DELCODE": dict(
        matrices_dir=REPO_ROOT / "DATA/DELCODE/__fc_wholebrain_sch200_flat__/matrices",
        splits_dir=REPO_ROOT / "DATA/DELCODE/__metadata__/SPLITS/downstream",
        id_col="Pseudonym",
    ),
    "ADNI": dict(
        matrices_dir=REPO_ROOT / "DATA/ADNI/__fc_wholebrain_sch200_flat__/matrices",
        splits_dir=REPO_ROOT / "DATA/ADNI/__metadata__/SPLITS/downstream",
        id_col="subject_id",
    ),
    "OASIS3": dict(
        matrices_dir=REPO_ROOT / "DATA/OASIS3/__fc_wholebrain_sch200_flat__/matrices",
        splits_dir=REPO_ROOT / "DATA/OASIS3/__metadata__/SPLITS/downstream",
        id_col="subject_id",
    ),
}

SUFFIX = "_whole_brain_correlation_matrix_z_transformed.npz"


def load_subject_df(splits_dir: Path) -> pd.DataFrame:
    return pd.concat(
        [pd.read_csv(splits_dir / f"{s}.csv") for s in ("train", "val", "test")],
        ignore_index=True,
    )


def earliest_file_per_subject(matrices_dir: Path, subject_ids: set[str]) -> dict[str, str]:
    """One file per subject: the earliest session by filename sort order.

    Filenames sort chronologically within a subject (ses-01 < ses-02 for
    DELCODE; ses-d<days> is zero-padded and increasing for ADNI/OASIS-3), so
    a plain sort gives the baseline scan without parsing visit codes.
    """
    files = sorted(f for f in os.listdir(matrices_dir) if f.endswith(SUFFIX))
    chosen: dict[str, str] = {}
    for f in files:
        m = re.match(r"sub-([^_]+)_", f)
        if not m:
            continue
        sid = m.group(1)
        if sid in subject_ids and sid not in chosen:
            chosen[sid] = f
    return chosen


def build_xy(cohort: str, cfg: dict) -> tuple[np.ndarray, np.ndarray, int, int]:
    df = load_subject_df(cfg["splits_dir"])
    df = df[df["sex"].isin(["m", "f"])].copy()
    id_col = cfg["id_col"]
    df[id_col] = df[id_col].astype(str)

    file_map = earliest_file_per_subject(cfg["matrices_dir"], set(df[id_col]))
    df = df[df[id_col].isin(file_map)]

    n_missing = len(load_subject_df(cfg["splits_dir"])) - len(df)

    triu_idx = None
    X, y = [], []
    for _, row in df.iterrows():
        arr = np.load(cfg["matrices_dir"] / file_map[row[id_col]])["array"]
        if triu_idx is None:
            triu_idx = np.triu_indices_from(arr, k=1)
        X.append(arr[triu_idx])
        y.append(1 if row["sex"] == "m" else 0)

    return np.array(X), np.array(y), len(df), n_missing


def main() -> None:
    print(f"{'cohort':10s} {'n_subj':>7s} {'missing_fc':>10s} {'n_male':>7s} {'n_female':>8s} {'sex_AUC (5-fold CV)':>22s}")
    results = {}
    for cohort, cfg in COHORTS.items():
        if not cfg["matrices_dir"].is_dir():
            print(f"{cohort:10s}  matrices dir missing: {cfg['matrices_dir']}")
            continue
        X, y, n, n_missing = build_xy(cohort, cfg)
        n_male, n_female = int(y.sum()), int((1 - y).sum())

        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=0.01, class_weight="balanced"),
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
        auc_mean, auc_sd = scores.mean(), scores.std()
        results[cohort] = (auc_mean, auc_sd)
        print(
            f"{cohort:10s} {n:7d} {n_missing:10d} {n_male:7d} {n_female:8d} "
            f"{auc_mean:14.4f} ± {auc_sd:.4f}"
        )

    print()
    print("Decision rule (DOCS/timeline/MASTER_PLAN.md §4, Phase 1):")
    print("  DELCODE high + ADNI/OASIS-3 high  -> features carry signal; at-chance")
    print("                                        conversion result is a finding")
    print("  DELCODE high + ADNI/OASIS-3 chance -> A.3 extraction is broken;")
    print("                                        external results are void")


if __name__ == "__main__":
    sys.exit(main())
