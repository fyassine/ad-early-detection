"""Visit-month parsing helpers shared across dataset classes.

Downstream split CSVs carry an ``allowed_months`` column listing the
pre-conversion visit months (e.g. ``"0;12;24"``) each patient may contribute.
For converter patients, visits relabelled ``ad`` in ``cohorts_with_scans_on_disk.csv``
(the post-conversion, already-demented scans) are excluded from that list to
prevent label leakage into the converter-vs-MCI classifiers.

These helpers parse the visit month from an ``.npz`` filename and the allow-list
cell so the month filter is identical across the GAAE, GEC, GELSTM and common
dataset classes. Filenames encode the visit month as an ``_M<int>_`` token, e.g.
``sub-XXX_ses-01_M12_..._whole_brain_correlation_matrix_z_transformed.npz``.

## Cohort-aware visit identity (DELCODE vs ADNI / OASIS-3)

DELCODE encodes a *nominal protocol month* in its filenames (``_M12_`` — the
scheduled visit label, drawn from a small discrete set: 0/12/24/36/48/60).
ADNI and OASIS-3 encode *actual elapsed days from baseline* instead
(``ses-d0381`` — continuous and irregular; see
``DATA/ADNI/src/unzip/build_visit_baselines.py``). Collapsing elapsed days onto
the nearest DELCODE-shaped protocol month would throw away the irregularity
GELSTM's ``use_time_delta`` exists to exploit; keeping days continuous for
DELCODE would break the ``allowed_months`` set-membership filter. The three
concepts below are therefore kept separate everywhere, with one function per
concept dispatching on cohort:

- ``visit_index``    — 0-based rank, used for ordering / window selection.
- ``protocol_month``  — nominal scheduled-visit label, used only by the
  ``allowed_months`` leakage filter. ``None`` when a session cannot be
  confidently mapped onto a scheduled visit (e.g. an ADNI unscheduled/"v"
  visit code) — never guessed.
- ``delta_t_months``  — float, cumulative elapsed months since baseline, used
  as model input (``use_time_delta``). Strictly increasing per subject by
  construction, so one code path (the existing inter-visit-diff computation
  in ``GELSTM/dataset.py``) serves all three cohorts. For DELCODE this equals
  the nominal protocol month exactly, so its values reproduce byte-for-byte
  under the refactor (see the A.2 gate in
  ``DOCS/meetings/ninth-meeting/comparison-plan-v2.md``).
"""

from __future__ import annotations

import math
import re
from typing import Literal, Sequence

import pandas as pd

Cohort = Literal["delcode", "adni", "oasis3"]

# Visit month token in FC-matrix filenames, e.g. '..._M12_...' -> 12.
_MONTH_RE = re.compile(r"_(M\d+)_")

# Elapsed-days-from-baseline token in ADNI / OASIS-3 filenames, e.g.
# '..._ses-d0381_...' -> 381. Both cohorts share this convention (see
# DATA/ADNI/src/unzip/build_visit_baselines.py's docstring).
_DAY_RE = re.compile(r"_ses-d(\d+)_")

# ADNI viscodes that name a nominal scheduled-visit month: 'bl' (baseline) or
# 'm<N>' (month N). Unscheduled / non-month codes ('v01', 'sc', 'scmri', ...)
# do not encode a protocol month and must not be parsed as one.
_ADNI_VISCODE_MONTH_RE = re.compile(r"^m(\d+)$")

DAYS_PER_MONTH = 30.44


def parse_month(filename: str) -> int | None:
    """Return the visit month int from an npz filename, or None if absent."""
    m = _MONTH_RE.search(filename)
    return int(m.group(1)[1:]) if m else None


def parse_allowed_months(cell) -> set[int] | None:
    """Parse an ``allowed_months`` CSV cell into a set of month ints.

    ``"0;12;24"`` -> ``{0, 12, 24}``. Returns ``None`` when the value is
    genuinely absent (NaN / empty string), signalling "no month filtering for
    this patient". Raises ``ValueError`` on a malformed non-empty value rather
    than silently dropping scans.
    """
    if cell is None:
        return None
    if isinstance(cell, float) and math.isnan(cell):
        return None
    text = str(cell).strip()
    if not text:
        return None
    try:
        return {int(tok) for tok in text.split(";") if tok.strip() != ""}
    except ValueError as exc:
        raise ValueError(
            f"Malformed allowed_months value {cell!r}; expected "
            "';'-separated integers like '0;12;24'."
        ) from exc


def allowed_months_map(
    filter_df: pd.DataFrame,
    id_col: str = "Pseudonym",
    column: str = "allowed_months",
) -> dict[str, set[int] | None] | None:
    """Build ``{pseudonym/subject_id -> set[int] | None}`` from a split DataFrame.

    Returns ``None`` when the ``allowed_months`` (or ``allowed_days``) column is absent so callers can
    treat a legacy CSV as "no month filtering" without changing behaviour.
    """
    if id_col not in filter_df.columns and "subject_id" in filter_df.columns:
        id_col = "subject_id"
    if column not in filter_df.columns and "allowed_days" in filter_df.columns:
        column = "allowed_days"
    if column not in filter_df.columns:
        return None
    return {
        str(pid): parse_allowed_months(val)
        for pid, val in zip(filter_df[id_col].astype(str), filter_df[column], strict=True)
    }


def month_allowed(
    filename: str, allowed: set[int] | None, cohort: Cohort = "delcode"
) -> bool:
    """True if the file's visit month/day is permitted for its subject.

    ``allowed=None`` means the subject has no restriction (kept). When a
    restriction is present, a file is kept only if its parsed month (or elapsed
    day for ADNI/OASIS-3) is in the allow-list — a file with no parseable
    visit value is dropped, since it cannot be positively verified as
    pre-conversion.
    """
    if allowed is None:
        return True
    if cohort == "delcode":
        m = parse_month(filename)
        if m is not None:
            return m in allowed
    d = parse_day(filename)
    if d is not None:
        return d in allowed
    m = parse_month(filename)
    return m in allowed if m is not None else False


# ---------------------------------------------------------------------------
# Cohort-aware visit identity (ADNI / OASIS-3 elapsed-days + the three-field
# split from DOCS/meetings/ninth-meeting/comparison-plan-v2.md §2).
# ---------------------------------------------------------------------------


def parse_day(filename: str) -> int | None:
    """Return the elapsed-days-from-baseline int from an ADNI/OASIS-3 filename.

    Filenames encode this as a ``ses-d<int>`` token, e.g.
    ``sub-ADNI002S2043_ses-d0381_task-rest_..._bold_reoriented.nii.gz`` -> 381.
    Returns ``None`` if absent (e.g. a DELCODE ``_M<n>_``-style filename).
    """
    m = _DAY_RE.search(filename)
    return int(m.group(1)) if m else None


def parse_adni_protocol_month(viscode: str) -> int | None:
    """Parse an ADNI viscode into a nominal protocol month, or ``None``.

    ``'bl'`` -> 0, ``'m<N>'`` -> N (case-insensitive). Any other viscode
    (``'v01'``, ``'sc'``, ``'scmri'``, ...) names an unscheduled or otherwise
    non-month visit and returns ``None`` rather than a guess: ADNI's
    ``'v'``-prefixed codes are sequential visit numbers, not elapsed months,
    and parsing them as months would silently corrupt the ``allowed_months``
    leakage filter with the wrong number.
    """
    code = viscode.strip().lower()
    if code == "bl":
        return 0
    m = _ADNI_VISCODE_MONTH_RE.match(code)
    return int(m.group(1)) if m else None


def visit_identity(
    cohort: Cohort, raw_values: Sequence[int | float]
) -> tuple[list[int], list[float]]:
    """Compute ``(visit_index, delta_t_months)`` for one subject's sorted visits.

    ``raw_values`` must already be sorted ascending and hold the cohort-native
    time value per visit: DELCODE protocol months (ints from :func:`parse_month`)
    or ADNI/OASIS-3 elapsed days from baseline (ints from :func:`parse_day`).

    ``visit_index`` is the 0-based rank. ``delta_t_months`` is the cumulative
    elapsed time in months since baseline: for DELCODE this equals the
    protocol month itself, so its values reproduce byte-for-byte under the
    visit-parsing refactor (the A.2 gate); for ADNI/OASIS-3 it is
    ``days / DAYS_PER_MONTH``. Strictly increasing per subject provided
    ``raw_values`` has no duplicates — a duplicate visit value is a data bug
    and is left for the caller to assert separately rather than silently
    collapsed here.
    """
    values = list(raw_values)
    if values != sorted(values):
        raise ValueError(f"visit_identity requires raw_values sorted ascending; got {values}.")
    visit_index = list(range(len(values)))
    if cohort == "delcode":
        delta_t_months = [float(v) for v in values]
    elif cohort in ("adni", "oasis3"):
        delta_t_months = [v / DAYS_PER_MONTH for v in values]
    else:
        raise ValueError(f"Unknown cohort {cohort!r}; expected 'delcode', 'adni', or 'oasis3'.")
    return visit_index, delta_t_months
