"""Cohort-manifest schema and build-time assertions.

Schema and assertion list per
DOCS/meetings/ninth-meeting/comparison-plan-v2.md §3 (Phase A.0). Every
assertion here fails loudly (raises) rather than silently dropping rows —
that is the entire point of A.0: two "progress counter globs a directory,
counts a strays" bugs have already shipped (the ``.html`` glob and the
OASIS-3 46-empty-subject-dir bug), and a manifest that quietly excludes the
same class of row would just be a third one.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

MANIFEST_COLUMNS = [
    "cohort",
    "subject_id",
    "session_id",
    "days_from_baseline",
    "visit_index",
    "protocol_month",
    "delta_t_months",
    "bold_path",
    "fc_path",
    "label",
    "scanner_vendor",
    "scanner_model",
    "site",
]

_PATH_COLUMNS = ("bold_path", "fc_path")


def assert_schema(df: pd.DataFrame) -> None:
    """Assert ``df`` has exactly the manifest columns (order-insensitive)."""
    missing = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in MANIFEST_COLUMNS]
    if missing or extra:
        raise ValueError(
            f"Manifest schema mismatch: missing={missing}, unexpected={extra}. "
            f"Expected exactly {MANIFEST_COLUMNS}."
        )


def assert_paths_exist_and_nonempty(
    df: pd.DataFrame, path_columns: tuple[str, ...] = _PATH_COLUMNS
) -> None:
    """Every non-null path in ``path_columns`` must exist on disk and be non-empty.

    Kills the empty-dir bug class (§1.3): a directory existing is not the same
    as it containing usable data.
    """
    bad: list[str] = []
    for col in path_columns:
        if col not in df.columns:
            continue
        for value in df[col].dropna():
            p = Path(str(value))
            if not p.exists():
                bad.append(f"{col}={value} (missing)")
            elif p.stat().st_size == 0:
                bad.append(f"{col}={value} (empty)")
    if bad:
        preview = "\n  ".join(bad[:20])
        more = f"\n  ... and {len(bad) - 20} more" if len(bad) > 20 else ""
        raise ValueError(
            f"{len(bad)} manifest path(s) missing or empty on disk:\n  {preview}{more}"
        )


def assert_every_subject_dir_contributes_sessions(
    df: pd.DataFrame,
    subject_dirs_on_disk: set[str],
    *,
    cohort: str,
    known_empty_subjects: frozenset[str] = frozenset(),
) -> list[str]:
    """Every subject directory on disk must contribute >= 1 manifest row.

    Kills the class of bug in §1.3: 46 OASIS-3 subject directories exist
    (``fmriprep`` output ran) but hold zero postprocessed BOLD files, and a
    progress counter that globs directory existence reports them as
    "settled" anyway. A subject directory with zero sessions is a build-time
    error here, not a silently-dropped row — unless explicitly acknowledged
    via ``known_empty_subjects`` (e.g. while the §5 triage is still pending).

    Returns the acknowledged-but-empty subject IDs (for logging), and raises
    if any *unacknowledged* empty subject directory is found.
    """
    contributing = set(df["subject_id"].astype(str).unique())
    empty = subject_dirs_on_disk - contributing
    unacknowledged = sorted(empty - known_empty_subjects)
    if unacknowledged:
        preview = ", ".join(unacknowledged[:20])
        more = f", ... and {len(unacknowledged) - 20} more" if len(unacknowledged) > 20 else ""
        raise ValueError(
            f"{cohort}: {len(unacknowledged)} subject director{'y' if len(unacknowledged) == 1 else 'ies'} "
            f"on disk contribute zero sessions to the manifest: {preview}{more}. "
            "This is the empty-dir bug class from §1.3 — either the data genuinely "
            "needs a postprocessing/triage pass, or pass their IDs in "
            "known_empty_subjects to acknowledge the gap explicitly."
        )
    return sorted(empty & known_empty_subjects)


def assert_counts_match(
    df: pd.DataFrame, *, cohort: str, expected_subjects: int, expected_sessions: int
) -> None:
    """Per-cohort subject/session totals must match the reproduced §7 counts."""
    actual_subjects = df["subject_id"].nunique()
    actual_sessions = len(df)
    if actual_subjects != expected_subjects or actual_sessions != expected_sessions:
        raise ValueError(
            f"{cohort}: manifest counts drifted from the §7-reproduced totals. "
            f"Expected subjects={expected_subjects}, sessions={expected_sessions}; "
            f"got subjects={actual_subjects}, sessions={actual_sessions}. "
            "Re-run DOCS/meetings/ninth-meeting/comparison-plan-v2.md §7's count "
            "block and update the expected counts if this drift is real (e.g. a "
            "postprocessing pass landed), rather than assuming the manifest is wrong."
        )


def assert_visit_index_contiguous(df: pd.DataFrame) -> None:
    """``visit_index`` must be 0-based and contiguous per subject."""
    bad: list[str] = []
    for subject_id, group in df.groupby("subject_id"):
        indices = sorted(group["visit_index"].tolist())
        if indices != list(range(len(indices))):
            bad.append(f"{subject_id}: {indices}")
    if bad:
        preview = "\n  ".join(bad[:20])
        raise ValueError(f"{len(bad)} subject(s) have non-contiguous visit_index:\n  {preview}")


def assert_delta_t_monotonic(
    df: pd.DataFrame, *, known_duplicate_day_subjects: frozenset[str] = frozenset()
) -> list[str]:
    """``delta_t_months`` must be strictly increasing per subject, sorted by visit_index.

    A same-day repeat scan (two BOLD acquisitions on the identical elapsed
    day — e.g. OASIS-3's paired ``task-restingstate`` / ``task-restingstateMB4``
    protocol variants) makes this impossible to satisfy by construction: both
    rows share the same ``days_from_baseline`` and therefore the same
    ``delta_t_months``, regardless of tie-break ordering. Which of the two
    scans is canonical is a data/modeling decision (which BOLD feeds FC
    extraction), not something this manifest builder should pick silently —
    so it is surfaced here exactly like the empty-dir class in
    ``assert_every_subject_dir_contributes_sessions``: fails loudly by
    default, listing the affected subjects, unless explicitly acknowledged.

    Returns the acknowledged-but-duplicated subject IDs.
    """
    bad: list[str] = []
    for subject_id, group in df.groupby("subject_id"):
        ordered = group.sort_values("visit_index")["delta_t_months"].tolist()
        if ordered != sorted(set(ordered)) or len(set(ordered)) != len(ordered):
            bad.append(str(subject_id))
    unacknowledged = sorted(set(bad) - known_duplicate_day_subjects)
    if unacknowledged:
        raise ValueError(
            f"{len(unacknowledged)} subject(s) have same-day duplicate scans, which makes "
            f"delta_t_months non-strictly-increasing: {unacknowledged}. Decide which scan "
            "per duplicated day is canonical, or pass their IDs in "
            "known_duplicate_day_subjects to acknowledge the gap explicitly."
        )
    return sorted(set(bad) & known_duplicate_day_subjects)


def assert_no_cross_label_duplicates(
    converter_ids: set[str], stable_ids: set[str], *, cohort: str
) -> None:
    """No subject may appear in both the converter and stable/non-converter CSVs."""
    both = converter_ids & stable_ids
    if both:
        raise ValueError(
            f"{cohort}: {len(both)} subject(s) appear in both converter and stable "
            f"cohort CSVs: {sorted(both)[:20]}. A subject cannot carry two labels."
        )


def validate_manifest(
    df: pd.DataFrame,
    *,
    cohort: str,
    subject_dirs_on_disk: set[str],
    expected_subjects: int | None = None,
    expected_sessions: int | None = None,
    known_empty_subjects: frozenset[str] = frozenset(),
    known_duplicate_day_subjects: frozenset[str] = frozenset(),
) -> dict:
    """Run every A.0 build-time assertion. Returns a small summary dict on success."""
    assert_schema(df)
    assert_paths_exist_and_nonempty(df)
    acknowledged_empty = assert_every_subject_dir_contributes_sessions(
        df, subject_dirs_on_disk, cohort=cohort, known_empty_subjects=known_empty_subjects
    )
    if expected_subjects is not None and expected_sessions is not None:
        assert_counts_match(
            df,
            cohort=cohort,
            expected_subjects=expected_subjects,
            expected_sessions=expected_sessions,
        )
    assert_visit_index_contiguous(df)
    acknowledged_duplicate_day = assert_delta_t_monotonic(
        df, known_duplicate_day_subjects=known_duplicate_day_subjects
    )
    return {
        "cohort": cohort,
        "subjects": df["subject_id"].nunique(),
        "sessions": len(df),
        "acknowledged_empty_subjects": acknowledged_empty,
        "acknowledged_duplicate_day_subjects": acknowledged_duplicate_day,
    }
