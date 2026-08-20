"""CLI entrypoint: build cohort_manifest.csv for one or all of DELCODE / ADNI / OASIS-3.

Usage:
    python -m DATA.manifest.build_cohort_manifest --cohort delcode
    python -m DATA.manifest.build_cohort_manifest --cohort adni
    python -m DATA.manifest.build_cohort_manifest --cohort oasis3 --acknowledge-empty-subjects
    python -m DATA.manifest.build_cohort_manifest --cohort all
"""

from __future__ import annotations

import argparse

from DATA.manifest import build_adni_manifest, build_delcode_manifest, build_oasis3_manifest

_COHORTS = ("delcode", "adni", "oasis3")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=(*_COHORTS, "all"), required=True)
    parser.add_argument(
        "--acknowledge-empty-subjects",
        action="store_true",
        help="OASIS-3 only: see DATA.manifest.build_oasis3_manifest for what this acknowledges.",
    )
    parser.add_argument(
        "--acknowledge-duplicate-day-sessions",
        action="store_true",
        help="OASIS-3 only: see DATA.manifest.build_oasis3_manifest for what this acknowledges.",
    )
    args = parser.parse_args()

    cohorts = _COHORTS if args.cohort == "all" else (args.cohort,)
    for cohort in cohorts:
        if cohort == "delcode":
            build_delcode_manifest.main([])
        elif cohort == "adni":
            build_adni_manifest.main([])
        elif cohort == "oasis3":
            oasis3_argv = []
            if args.acknowledge_empty_subjects:
                oasis3_argv.append("--acknowledge-empty-subjects")
            if args.acknowledge_duplicate_day_sessions:
                oasis3_argv.append("--acknowledge-duplicate-day-sessions")
            build_oasis3_manifest.main(oasis3_argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
