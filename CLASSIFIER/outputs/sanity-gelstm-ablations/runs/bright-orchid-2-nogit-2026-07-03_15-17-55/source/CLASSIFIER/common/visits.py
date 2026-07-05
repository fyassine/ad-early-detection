"""Visit-month parsing helpers shared across dataset classes.

Downstream split CSVs carry an ``allowed_months`` column listing the
pre-conversion visit months (e.g. ``"0;12;24"``) each patient may contribute.
For converter patients, visits relabelled ``ad`` in ``cohorts_with_scans.csv``
(the post-conversion, already-demented scans) are excluded from that list to
prevent label leakage into the converter-vs-MCI classifiers.

These helpers parse the visit month from an ``.npz`` filename and the allow-list
cell so the month filter is identical across the GAAE, GEC, GELSTM and common
dataset classes. Filenames encode the visit month as an ``_M<int>_`` token, e.g.
``sub-XXX_ses-01_M12_..._whole_brain_correlation_matrix_z_transformed.npz``.
"""

from __future__ import annotations

import math
import re

import pandas as pd

# Visit month token in FC-matrix filenames, e.g. '..._M12_...' -> 12.
_MONTH_RE = re.compile(r"_(M\d+)_")


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
    """Build ``{pseudonym -> set[int] | None}`` from a split DataFrame.

    Returns ``None`` when the ``allowed_months`` column is absent so callers can
    treat a legacy CSV as "no month filtering" without changing behaviour.
    """
    if column not in filter_df.columns:
        return None
    return {
        str(pid): parse_allowed_months(val)
        for pid, val in zip(filter_df[id_col].astype(str), filter_df[column], strict=False)
    }


def month_allowed(filename: str, allowed: set[int] | None) -> bool:
    """True if the file's visit month is permitted for its subject.

    ``allowed=None`` means the subject has no month restriction (kept). When a
    restriction is present, a file is kept only if its parsed month is in the
    allow-list — a file with no parseable month is dropped, since it cannot be
    positively verified as pre-conversion.
    """
    if allowed is None:
        return True
    return parse_month(filename) in allowed
