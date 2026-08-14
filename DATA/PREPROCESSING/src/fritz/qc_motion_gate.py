#!/usr/bin/env python3
"""Motion/FD QC gate for one subject's fMRIPrep output (Fritz-side).

Operationalizes the mean-FD exclusion guideline (reused from the reference
``DATA/__artifacts__/PREPROCESSING/scripts/04_postprocessing/qc_motion_table.py``):

  * mean framewise displacement > 0.5 mm  -> exclude that session
  * optional per-volume scrub fraction at FD > 0.2 mm (reported, not gated)
  * usable scan time < 5 min after dummy drop -> flag (reported, not gated)

Framewise displacement is read directly from fMRIPrep's
``*_desc-confounds_timeseries.tsv`` (the same file the postprocessing container
consumes), so no separate MRIQC IQM JSON is needed.

Scope: one fMRIPrep-flat subject dir, ``<fmriprep_root>/sub-<ID>/ses-*/func/``.
A subject can have several sessions; each rest BOLD run is judged independently
(the flat product is per-session, so gating is per-session).

Behaviour:
  * appends one row per session/run to ``--qc-csv`` (created with header if absent)
  * prints the ``ses-<X>`` id of every PASSING session to stdout, one per line,
    for the orchestrator to consume
  * exits 0 on success; raises loudly (non-zero) on a malformed/missing confounds
    file rather than silently passing or failing a session

This module deliberately does NOT choose a fallback threshold silently — the
0.5 mm cutoff is an explicit, named constant matching the reference convention.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

# Named convention, matching the reference qc_motion_table.py. Not a silent default:
# override via --fd-threshold if a run needs a different cutoff.
MEAN_FD_EXCLUDE_THRESHOLD_MM = 0.5
DEFAULT_SCRUB_THRESHOLD_MM = 0.2
MIN_USABLE_MINUTES = 5.0

CONFOUNDS_SUFFIX = "_desc-confounds_timeseries.tsv"

QC_FIELDS = [
    "dataset",
    "subject",
    "session",
    "confounds_file",
    "n_volumes",
    "mean_fd_mm",
    "fd_threshold_mm",
    "usable_minutes",
    "pct_scrubbed",
    "excluded_high_motion",
    "flagged_short_scan",
    "verdict",
]


def find_confounds(subject_dir: Path) -> list[Path]:
    """All rest-BOLD confounds TSVs under sub-<ID>/ses-*/func/, sorted."""
    return sorted(subject_dir.glob(f"ses-*/func/*{CONFOUNDS_SUFFIX}"))


def session_of(confounds_path: Path) -> str:
    """Recover the ses-<X> id from the confounds path (…/ses-<X>/func/…)."""
    for part in confounds_path.parts:
        if part.startswith("ses-"):
            return part
    raise ValueError(f"could not find a ses-* component in {confounds_path}")


def read_tr(confounds_path: Path) -> float | None:
    """RepetitionTime from the run's BOLD JSON sidecar, if present.

    fMRIPrep names the sidecar ``<prefix>desc-preproc_bold.json`` (BIDS
    derivatives), not a bare ``<prefix>bold.json`` — try the derivatives name
    first, then the raw-BIDS name, then any ``*_bold.json`` for the same run.
    """
    prefix = confounds_path.name[: -len(CONFOUNDS_SUFFIX)]
    candidates = [
        confounds_path.with_name(f"{prefix}_desc-preproc_bold.json"),
        confounds_path.with_name(f"{prefix}_bold.json"),
    ]
    candidates += sorted(confounds_path.parent.glob(f"{prefix}*_bold.json"))
    for sidecar in candidates:
        if sidecar.exists():
            tr = json.loads(sidecar.read_text()).get("RepetitionTime")
            if tr is not None:
                return float(tr)
    return None


def score_session(
    confounds_path: Path,
    *,
    fd_threshold: float,
    scrub_threshold: float | None,
    dummy: int,
) -> dict:
    df = pd.read_csv(confounds_path, sep="\t")
    if "framewise_displacement" not in df.columns:
        raise ValueError(
            f"{confounds_path} has no 'framewise_displacement' column — "
            "cannot QC this session. Is this really an fMRIPrep confounds TSV?"
        )

    # Drop the dummy scans the postprocessing container also drops, so the FD we
    # judge matches the FD of the volumes that actually survive into the product.
    fd = df["framewise_displacement"].iloc[dummy:].fillna(0.0)
    n_volumes = int(len(fd))
    if n_volumes == 0:
        raise ValueError(f"{confounds_path}: 0 volumes left after dropping {dummy} dummy scans.")

    mean_fd = float(fd.mean())
    tr = read_tr(confounds_path)
    usable_minutes = (n_volumes * tr / 60.0) if tr else None

    pct_scrubbed = None
    if scrub_threshold is not None:
        pct_scrubbed = float((fd > scrub_threshold).mean() * 100.0)

    excluded = mean_fd > fd_threshold
    flagged_short = usable_minutes is not None and usable_minutes < MIN_USABLE_MINUTES

    return {
        "confounds_file": confounds_path.name,
        "n_volumes": n_volumes,
        "mean_fd_mm": round(mean_fd, 4),
        "fd_threshold_mm": fd_threshold,
        "usable_minutes": round(usable_minutes, 2) if usable_minutes is not None else "",
        "pct_scrubbed": round(pct_scrubbed, 2) if pct_scrubbed is not None else "",
        "excluded_high_motion": excluded,
        "flagged_short_scan": flagged_short,
        "verdict": "EXCLUDE" if excluded else "PASS",
    }


def append_qc_rows(qc_csv: Path, rows: list[dict]) -> None:
    qc_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not qc_csv.exists()
    with qc_csv.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=QC_FIELDS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in QC_FIELDS})


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--fmriprep-root", type=Path, required=True, help="dir containing sub-* fMRIPrep output"
    )
    parser.add_argument("--subject", required=True, help="subject id WITHOUT the sub- prefix")
    parser.add_argument("--dataset", default="", help="cohort label recorded in the QC CSV")
    parser.add_argument("--qc-csv", type=Path, required=True, help="QC ledger to append to")
    parser.add_argument(
        "--fd-threshold",
        type=float,
        default=MEAN_FD_EXCLUDE_THRESHOLD_MM,
        help=f"mean-FD exclusion cutoff in mm (default {MEAN_FD_EXCLUDE_THRESHOLD_MM})",
    )
    parser.add_argument(
        "--scrub-threshold",
        type=float,
        default=None,
        help=f"if given, report pct of volumes above this FD (reference {DEFAULT_SCRUB_THRESHOLD_MM}mm)",
    )
    parser.add_argument(
        "--dummy",
        type=int,
        default=10,
        help="dummy scans dropped before FD is judged (match the container's --dummy)",
    )
    args = parser.parse_args()

    subject_dir = args.fmriprep_root / f"sub-{args.subject}"
    if not subject_dir.is_dir():
        sys.exit(f"ERROR: no subject dir at {subject_dir}")

    confounds = find_confounds(subject_dir)
    if not confounds:
        sys.exit(
            f"ERROR: no *{CONFOUNDS_SUFFIX} under {subject_dir}/ses-*/func/ — "
            "fMRIPrep sometimes fails to emit confounds; cannot QC this subject."
        )

    rows: list[dict] = []
    passing_sessions: list[str] = []
    for confounds_path in confounds:
        session = session_of(confounds_path)
        result = score_session(
            confounds_path,
            fd_threshold=args.fd_threshold,
            scrub_threshold=args.scrub_threshold,
            dummy=args.dummy,
        )
        result.update({"dataset": args.dataset, "subject": args.subject, "session": session})
        rows.append(result)
        if result["verdict"] == "PASS":
            passing_sessions.append(session)

    append_qc_rows(args.qc_csv, rows)

    # stdout is the machine-readable channel for the orchestrator: passing ses ids only.
    for session in passing_sessions:
        print(session)


if __name__ == "__main__":
    main()
