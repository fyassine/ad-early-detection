#!/usr/bin/env python3
"""Stage 4 QC: turn the mean-FD exclusion guideline into a concrete per-subject table.

Operationalizes gap #7 from the online research: mean FD > 0.5mm -> exclude; optional
per-volume scrub flag at FD > 0.2mm (off by default, per "optional" framing); usable scan time
< 5 minutes after exclusion -> flag. Reads framewise_displacement directly from fMRIPrep's
confounds TSV (needed for scrubbing anyway, so reuse it rather than re-deriving from MRIQC's
separate IQM JSON).

Usage:
    python qc_motion_table.py --confounds-tsv <path> --subject <id> --session <id> \\
        [--t-r 2.58] [--scrub-threshold 0.2] [--append-to qc_summary.tsv]
"""
import argparse
from pathlib import Path

import pandas as pd

MEAN_FD_EXCLUDE_THRESHOLD = 0.5   # mm
DEFAULT_SCRUB_THRESHOLD = 0.2     # mm
MIN_USABLE_MINUTES = 5.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confounds-tsv", type=Path, required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--t-r", type=float, default=None, help="seconds; read from sidecar if omitted")
    parser.add_argument("--scrub-threshold", type=float, default=None,
                         help=f"mm; if given, compute pct of volumes above this FD (default off, "
                              f"reference value is {DEFAULT_SCRUB_THRESHOLD}mm if you enable it)")
    parser.add_argument("--append-to", type=Path, default=None)
    args = parser.parse_args()

    confounds_df = pd.read_csv(args.confounds_tsv, sep="\t")
    fd = confounds_df["framewise_displacement"].fillna(0.0)

    t_r = args.t_r
    if t_r is None:
        import json
        sidecar = args.confounds_tsv.with_name(
            args.confounds_tsv.name.replace("desc-confounds_timeseries.tsv", "bold.json")
        )
        t_r = json.loads(sidecar.read_text())["RepetitionTime"] if sidecar.exists() else None

    mean_fd = float(fd.mean())
    n_volumes = len(fd)
    usable_minutes = (n_volumes * t_r / 60.0) if t_r else None

    pct_scrubbed = None
    if args.scrub_threshold is not None:
        pct_scrubbed = float((fd > args.scrub_threshold).mean() * 100)

    row = {
        "subject": args.subject,
        "session": args.session,
        "n_volumes": n_volumes,
        "mean_fd_mm": round(mean_fd, 4),
        "excluded_mean_fd": mean_fd > MEAN_FD_EXCLUDE_THRESHOLD,
        "usable_minutes": round(usable_minutes, 2) if usable_minutes is not None else None,
        "flagged_short_scan": (usable_minutes is not None and usable_minutes < MIN_USABLE_MINUTES),
        "pct_scrubbed": round(pct_scrubbed, 2) if pct_scrubbed is not None else None,
    }

    print(row)

    if args.append_to:
        df_row = pd.DataFrame([row])
        if args.append_to.exists():
            df_row.to_csv(args.append_to, sep="\t", mode="a", header=False, index=False)
        else:
            args.append_to.parent.mkdir(parents=True, exist_ok=True)
            df_row.to_csv(args.append_to, sep="\t", mode="w", header=True, index=False)
        print(f"Appended to {args.append_to}")


if __name__ == "__main__":
    main()
