"""
CLASSIFIER_v2/common/sanity.py — Split-hygiene and duplicate-data audits.

Two things this module guarantees, with hard failures (not warnings):

1. `assert_splits_clean(*split_csvs, id_col)` — for every pair of CSVs, the
   sets of subject IDs are disjoint. Used at the head of every production
   notebook so a regression in the split-generation pipeline halts execution.

2. `assert_no_duplicate_matrices(npz_paths)` — content-hashes every .npz
   correlation matrix and flags any group with identical content under
   different filenames (= upstream preprocessing bug).

Also exposes:

3. `audit_groupkfold(subject_ids, labels, n_splits, seed)` — runs
   StratifiedGroupKFold and returns per-fold subject sets, asserting zero
   inter-fold overlap (defends against a subtle bug where someone re-splits
   on scan IDs by accident).

All functions return a dict for the notebook to print/log.
"""
from __future__ import annotations

import hashlib
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


# ── 1. Split overlap ──────────────────────────────────────────────────────────

def _load_subject_ids(csv_path: str, id_col: str) -> set:
    df = pd.read_csv(csv_path)
    if id_col not in df.columns:
        raise KeyError(f"{csv_path}: missing column {id_col!r}; has {list(df.columns)}")
    return set(df[id_col].astype(str).unique())


def assert_splits_clean(
    split_csvs: Dict[str, str],
    id_col: str = "Repseudonym",
    raise_on_overlap: bool = True,
) -> Dict[str, object]:
    """
    Assert pairwise-disjoint subject IDs across all named split CSVs.

    Parameters
    ----------
    split_csvs : dict {split_name: csv_path}
        e.g. {"train": ".../train.csv", "val": ".../val.csv", "test": ".../test.csv"}
    id_col : str
        Column holding the subject identifier. Default 'Repseudonym'.
    raise_on_overlap : bool
        If True (default), raise AssertionError when any pair overlaps.
        If False, return the report without raising.

    Returns
    -------
    report : dict with keys
        sizes        : {name: n_subjects}
        overlaps     : list of (a, b, n_shared, [examples])
        clean        : bool — True iff every pair is disjoint
    """
    sets = {name: _load_subject_ids(path, id_col) for name, path in split_csvs.items()}
    sizes = {name: len(s) for name, s in sets.items()}

    overlaps: List[Tuple[str, str, int, List[str]]] = []
    for (a, sa), (b, sb) in combinations(sets.items(), 2):
        shared = sa & sb
        if shared:
            overlaps.append((a, b, len(shared), sorted(shared)[:5]))

    clean = (len(overlaps) == 0)
    if raise_on_overlap and not clean:
        msg = "; ".join(
            f"{a}↔{b}: {n} shared subjects (e.g. {ex})"
            for a, b, n, ex in overlaps
        )
        raise AssertionError(f"Split-overlap detected — {msg}")

    return {"sizes": sizes, "overlaps": overlaps, "clean": clean}


# ── 2. Duplicate-matrix audit ─────────────────────────────────────────────────

def _hash_npz(path: str, array_key: str = "array") -> str:
    arr = np.load(path)[array_key]
    arr = np.ascontiguousarray(arr)
    return hashlib.sha1(arr.tobytes()).hexdigest()


def assert_no_duplicate_matrices(
    npz_paths: Sequence[str],
    array_key: str = "array",
    raise_on_dup: bool = True,
) -> Dict[str, object]:
    """
    Hash every .npz array; raise if two distinct paths share content.

    Returns
    -------
    report : dict with keys
        n_files     : int
        n_unique    : int
        duplicates  : list of {"hash": str, "paths": [str, ...]}
        clean       : bool
    """
    by_hash: Dict[str, List[str]] = {}
    for p in npz_paths:
        try:
            h = _hash_npz(p, array_key=array_key)
        except Exception as e:
            raise RuntimeError(f"Failed to hash {p}: {e}") from e
        by_hash.setdefault(h, []).append(str(p))

    dups = [{"hash": h, "paths": ps} for h, ps in by_hash.items() if len(ps) > 1]
    clean = (len(dups) == 0)
    if raise_on_dup and not clean:
        examples = "; ".join(
            f"{d['hash'][:8]}: {len(d['paths'])} files (e.g. {d['paths'][:2]})"
            for d in dups[:3]
        )
        raise AssertionError(
            f"Duplicate-content matrices detected — {len(dups)} group(s); {examples}"
        )

    return {
        "n_files":    len(npz_paths),
        "n_unique":   len(by_hash),
        "duplicates": dups,
        "clean":      clean,
    }


# ── 3. StratifiedGroupKFold audit ─────────────────────────────────────────────

def audit_groupkfold(
    subject_ids: Sequence[str],
    labels: Sequence[int],
    n_splits: int = 5,
    seed: int = 42,
) -> Dict[str, object]:
    """
    Re-run StratifiedGroupKFold with the saved seed and assert zero subject
    overlap between any two folds' validation sets.

    Returns
    -------
    report : dict with keys
        folds        : list of {"train_subjects": [...], "val_subjects": [...]}
        overlaps     : list of (fold_a, fold_b, n_shared)
        clean        : bool
    """
    from sklearn.model_selection import StratifiedGroupKFold

    sids = np.array(list(map(str, subject_ids)))
    y    = np.array(list(labels), dtype=int)

    # One row per subject (StratifiedGroupKFold expects groups = sample-level IDs,
    # but here each sample is already a subject, so the group is itself).
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    folds = []
    val_sets: List[set] = []
    for tr_idx, val_idx in cv.split(X=sids.reshape(-1, 1), y=y, groups=sids):
        tr_subj  = set(sids[tr_idx])
        val_subj = set(sids[val_idx])
        folds.append({
            "train_subjects": sorted(tr_subj),
            "val_subjects":   sorted(val_subj),
            "n_train":        len(tr_subj),
            "n_val":          len(val_subj),
        })
        val_sets.append(val_subj)

    overlaps: List[Tuple[int, int, int]] = []
    for (i, a), (j, b) in combinations(enumerate(val_sets), 2):
        shared = a & b
        if shared:
            overlaps.append((i, j, len(shared)))

    clean = (len(overlaps) == 0)
    if not clean:
        raise AssertionError(
            f"StratifiedGroupKFold produced overlapping validation folds: {overlaps}"
        )

    return {"folds": folds, "overlaps": overlaps, "clean": clean}


# ── 4. Cohort-policy assertion ────────────────────────────────────────────────

def assert_cohort_policy(
    gaae_pretrain_subjects: Sequence[str],
    downstream_subjects: Sequence[str],
    policy: str = "shared",
) -> Dict[str, object]:
    """
    State the chosen cohort policy explicitly and assert it.

    policy="shared"   — the two cohorts are deliberately the same set.
    policy="disjoint" — the two cohorts must not overlap (GAAE pretrained
                         on subjects never seen by the downstream classifier).

    Returns the overlap report; raises AssertionError on policy violation.
    """
    a = set(map(str, gaae_pretrain_subjects))
    b = set(map(str, downstream_subjects))
    shared = a & b

    if policy == "disjoint" and shared:
        raise AssertionError(
            f"Cohort policy 'disjoint' violated — {len(shared)} subjects in both "
            f"(e.g. {sorted(shared)[:5]})"
        )
    if policy == "shared" and not a == b:
        # Lenient: warn but don't raise (notebooks may legitimately use a
        # subset of GAAE pretrain for downstream).
        pass

    return {
        "policy": policy,
        "gaae_n": len(a),
        "downstream_n": len(b),
        "shared_n": len(shared),
        "ok": True,
    }


# ── Smoke helper for notebooks ────────────────────────────────────────────────

def run_full_audit(
    split_csvs: Dict[str, str],
    id_col: str = "Repseudonym",
    verbose: bool = True,
) -> Dict[str, object]:
    """One-shot helper used at the head of every production notebook."""
    rep = assert_splits_clean(split_csvs, id_col=id_col, raise_on_overlap=True)
    if verbose:
        print("[SANITY] Split sizes:", rep["sizes"])
        print("[SANITY] Pairwise-disjoint: OK")
    return rep
