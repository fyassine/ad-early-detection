"""
Integration tests for the DATA/manifest cohort builders, run against the real
on-disk flat directories and metadata CSVs. Auto-skips per-cohort when its
source data isn't present (mirrors
DATA/DELCODE/src/splitting/tests/test_split_integrity.py's Section B pattern).

These pin the §7-reproduced counts from
DOCS/meetings/ninth-meeting/comparison-plan-v2.md so a drift back into the
"progress counter globs a directory, not content" bug class is caught here
first, not three weeks later in a fourth incarnation of it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from CLASSIFIER.common.visits import DAYS_PER_MONTH
from DATA.manifest.build_adni_manifest import (
    DEFAULT_FMRI_ROOT as ADNI_FMRI_ROOT,
    EXPECTED_SESSIONS as ADNI_EXPECTED_SESSIONS,
    EXPECTED_SUBJECTS as ADNI_EXPECTED_SUBJECTS,
    build_adni_manifest,
)
from DATA.manifest.build_delcode_manifest import (
    DEFAULT_FMRI_ROOT as DELCODE_FMRI_ROOT,
    build_delcode_manifest,
)
from DATA.manifest.build_oasis3_manifest import (
    DEFAULT_FMRI_ROOT as OASIS3_FMRI_ROOT,
    EXPECTED_SESSIONS as OASIS3_EXPECTED_SESSIONS,
    EXPECTED_SUBJECTS as OASIS3_EXPECTED_SUBJECTS,
    build_oasis3_manifest,
)
from DATA.manifest._day_coded import subject_dirs_on_disk
from DATA.manifest.schema import MANIFEST_COLUMNS, validate_manifest


def _skip_unless_dir(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"Flat directory not found: {path}")


# ---------------------------------------------------------------------------
# DELCODE
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def delcode_manifest():
    _skip_unless_dir(DELCODE_FMRI_ROOT)
    return build_delcode_manifest()


def test_delcode_manifest_has_schema_columns(delcode_manifest):
    assert list(delcode_manifest.columns) == MANIFEST_COLUMNS


def test_delcode_manifest_nonempty(delcode_manifest):
    assert len(delcode_manifest) > 0
    assert delcode_manifest["subject_id"].nunique() > 0


def test_delcode_baseline_visit_has_protocol_month_zero_when_m0_present(delcode_manifest):
    # visit_index==0 is a subject's *earliest on-disk* visit, which is M0 only
    # for subjects who actually have an M0 scan — not guaranteed for everyone.
    m0_subjects = delcode_manifest.loc[delcode_manifest["protocol_month"] == 0, "subject_id"]
    baseline_rows = delcode_manifest[
        (delcode_manifest["visit_index"] == 0) & (delcode_manifest["subject_id"].isin(m0_subjects))
    ]
    assert len(baseline_rows) > 0
    assert (baseline_rows["protocol_month"] == 0).all()


def test_delcode_delta_t_equals_protocol_month(delcode_manifest):
    assert (
        delcode_manifest["delta_t_months"] == delcode_manifest["protocol_month"].astype(float)
    ).all()


def test_delcode_manifest_passes_validation(delcode_manifest):
    subject_dirs = subject_dirs_on_disk(DELCODE_FMRI_ROOT)
    summary = validate_manifest(delcode_manifest, cohort="delcode", subject_dirs_on_disk=subject_dirs)
    assert summary["sessions"] == len(delcode_manifest)


# ---------------------------------------------------------------------------
# ADNI
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def adni_manifest():
    _skip_unless_dir(ADNI_FMRI_ROOT)
    return build_adni_manifest()


def test_adni_manifest_has_schema_columns(adni_manifest):
    assert list(adni_manifest.columns) == MANIFEST_COLUMNS


def test_adni_matches_section_7_counts(adni_manifest):
    assert adni_manifest["subject_id"].nunique() == ADNI_EXPECTED_SUBJECTS
    assert len(adni_manifest) == ADNI_EXPECTED_SESSIONS


def test_adni_labels_are_converter_or_stable_no_strays(adni_manifest):
    # §7: "the ADNI flat product *is* the MCI cohort, no strays" — every
    # subject must resolve to exactly one of the two labels.
    assert adni_manifest["label"].isin({"converter", "stable"}).all()


def test_adni_baseline_day_zero_delta_t_zero(adni_manifest):
    baseline_rows = adni_manifest[adni_manifest["visit_index"] == 0]
    zero_day = baseline_rows[baseline_rows["days_from_baseline"] == 0]
    assert (zero_day["delta_t_months"] == 0.0).all()


def test_adni_delta_t_matches_days_formula(adni_manifest):
    expected = adni_manifest["days_from_baseline"] / DAYS_PER_MONTH
    assert (adni_manifest["delta_t_months"] - expected).abs().max() < 1e-9


def test_adni_manifest_passes_validation(adni_manifest):
    summary = validate_manifest(
        adni_manifest,
        cohort="adni",
        subject_dirs_on_disk=subject_dirs_on_disk(ADNI_FMRI_ROOT),
        expected_subjects=ADNI_EXPECTED_SUBJECTS,
        expected_sessions=ADNI_EXPECTED_SESSIONS,
    )
    assert summary["acknowledged_empty_subjects"] == []


# ---------------------------------------------------------------------------
# OASIS-3
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def oasis3_manifest():
    _skip_unless_dir(OASIS3_FMRI_ROOT)
    return build_oasis3_manifest()


def test_oasis3_manifest_has_schema_columns(oasis3_manifest):
    assert list(oasis3_manifest.columns) == MANIFEST_COLUMNS


def test_oasis3_matches_section_7_counts(oasis3_manifest):
    # As of 2026-08-21 the §1.3 empty-dir gap is closed (see module docstring
    # in build_oasis3_manifest.py) — all 128 on-disk dirs now contribute.
    assert oasis3_manifest["subject_id"].nunique() == OASIS3_EXPECTED_SUBJECTS
    assert len(oasis3_manifest) == OASIS3_EXPECTED_SESSIONS


def test_oasis3_protocol_month_is_always_none(oasis3_manifest):
    # No clean scheduled-visit-month code exists for OASIS-3 (see module
    # docstring) — must be None everywhere, never a guessed value.
    assert oasis3_manifest["protocol_month"].isna().all()


def test_oasis3_build_fails_loudly_on_empty_dirs_by_default(oasis3_manifest):
    subject_dirs = subject_dirs_on_disk(OASIS3_FMRI_ROOT)
    contributing = set(oasis3_manifest["subject_id"].unique())
    empty = subject_dirs - contributing
    if not empty:
        pytest.skip("No empty OASIS-3 subject directories currently on disk (§5 triage landed?).")
    with pytest.raises(ValueError, match="empty-dir bug class"):
        validate_manifest(oasis3_manifest, cohort="oasis3", subject_dirs_on_disk=subject_dirs)


def _duplicate_day_subjects(df):
    counts = df.groupby(["subject_id", "days_from_baseline"]).size()
    return {subject_id for subject_id, _ in counts[counts > 1].index}


def test_oasis3_build_fails_loudly_on_duplicate_day_sessions_by_default(oasis3_manifest):
    duplicated = _duplicate_day_subjects(oasis3_manifest)
    if not duplicated:
        pytest.skip("No same-day duplicate OASIS-3 scans currently on disk.")
    subject_dirs = subject_dirs_on_disk(OASIS3_FMRI_ROOT)
    with pytest.raises(ValueError, match="same-day duplicate scans"):
        validate_manifest(oasis3_manifest, cohort="oasis3", subject_dirs_on_disk=subject_dirs)


def test_oasis3_build_passes_when_all_gaps_acknowledged(oasis3_manifest):
    subject_dirs = subject_dirs_on_disk(OASIS3_FMRI_ROOT)
    contributing = set(oasis3_manifest["subject_id"].unique())
    known_empty = frozenset(subject_dirs - contributing)
    known_duplicate_day = frozenset(_duplicate_day_subjects(oasis3_manifest))
    summary = validate_manifest(
        oasis3_manifest,
        cohort="oasis3",
        subject_dirs_on_disk=subject_dirs,
        expected_subjects=OASIS3_EXPECTED_SUBJECTS,
        expected_sessions=OASIS3_EXPECTED_SESSIONS,
        known_empty_subjects=known_empty,
        known_duplicate_day_subjects=known_duplicate_day,
    )
    assert set(summary["acknowledged_empty_subjects"]) == known_empty
    assert set(summary["acknowledged_duplicate_day_subjects"]) == known_duplicate_day
