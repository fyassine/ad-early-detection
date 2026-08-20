"""Shared flat-directory session discovery for the day-coded cohorts (ADNI, OASIS-3).

Both cohorts share the same on-disk convention: ``__fmri_wholebrain_sch200_flat__
/fmri/sub-<id>/sub-<id>_ses-d<days>_task-rest_..._bold_reoriented.nii.gz``, where
``<days>`` is elapsed days from that subject's baseline scan (see
``DATA/ADNI/src/unzip/build_visit_baselines.py``). This module is the one place
that walks that directory layout, so ADNI's and OASIS-3's manifest builders stay
identical apart from their label-CSV and scanner-metadata joins.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, NamedTuple

from CLASSIFIER.common.visits import parse_day


class FlatSession(NamedTuple):
    subject_id: str  # 'sub-' prefix stripped
    day: int
    bold_path: Path


def iter_flat_sessions(fmri_root: Path) -> Iterator[FlatSession]:
    """Yield one ``FlatSession`` per BOLD file under a day-coded flat root.

    Subject directories that exist but contribute zero sessions (the §1.3
    empty-dir bug) are simply absent from this iterator's output — callers
    must compare against ``os.listdir(fmri_root)`` themselves (see
    ``schema.assert_every_subject_dir_contributes_sessions``) rather than
    relying on this function to notice the gap, since silently noticing and
    dropping is exactly the failure mode A.0 exists to kill.
    """
    for subject_dir in sorted(fmri_root.glob("sub-*")):
        if not subject_dir.is_dir():
            continue
        subject_id = subject_dir.name.removeprefix("sub-")
        for bold_path in sorted(subject_dir.glob("*.nii.gz")):
            day = parse_day(bold_path.name)
            if day is None:
                continue
            yield FlatSession(subject_id=subject_id, day=day, bold_path=bold_path)


def subject_dirs_on_disk(fmri_root: Path) -> set[str]:
    """Subject IDs (no 'sub-' prefix) with a directory under ``fmri_root``, empty or not."""
    return {p.name.removeprefix("sub-") for p in fmri_root.glob("sub-*") if p.is_dir()}
