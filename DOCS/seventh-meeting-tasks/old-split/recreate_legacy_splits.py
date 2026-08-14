"""
Re-runs the Feb-2026 legacy split generators against the preserved raw graph data
and scores the result against the preserved split JSONs sitting next to this file.

Answers the question "can the old split be recreated from code?" empirically rather
than by assertion. Read-only: writes nothing outside --out.

Usage:
    python recreate_legacy_splits.py            # report fidelity only
    python recreate_legacy_splits.py --out DIR  # also dump the regenerated splits

Requires the sibling checkout /mnt/e/fyassine/_ad-early-detection to still be on disk.
"""

import argparse
import json
import re
from pathlib import Path

from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
LEGACY_DATA = Path("/mnt/e/fyassine/_ad-early-detection/data/Data-Delcode")

# Reproduces create_gec/gaae_data_splits.py exactly, including RANDOM_SEED = 42.
RANDOM_SEED = 42

RAW_DIRS = {
    "ad": LEGACY_DATA / "Delcode_AD_graph_data" / "raw",
    "healthy": LEGACY_DATA / "Delcode_healthy_graph_data_demographics" / "raw",
    "converter": LEGACY_DATA / "Delcode_Converter_graph_data" / "raw",
    "mci": LEGACY_DATA / "Delcode_MCI_SCD_exclude_converter_graph_data" / "raw",
}


def extract_patient_id(filename: str) -> str | None:
    match = re.match(r"(sub-[a-f0-9]+)", filename)
    return match.group(1) if match else None


def get_patient_files(directory: Path, pattern: str) -> dict[str, list[str]]:
    if not directory.exists():
        raise FileNotFoundError(
            f"Legacy raw directory missing: {directory}. The sibling checkout "
            f"_ad-early-detection must be present to recreate these splits."
        )
    patient_files: dict[str, list[str]] = {}
    for f in directory.glob(pattern):
        pid = extract_patient_id(f.name)
        if pid:
            patient_files.setdefault(pid, []).append(f.name)
    return patient_files


def split_ids(patient_ids: list[str], test_size: float) -> tuple[list[str], list[str]]:
    if len(patient_ids) < 2:
        return patient_ids, []
    return train_test_split(patient_ids, test_size=test_size, random_state=RANDOM_SEED)


def split_by_files(patient_dict: dict, test_size: float) -> tuple[list[str], list[str]]:
    """create_gaae_data_splits.py::stratified_split_by_files.

    Sorts by file count descending, then splits. The sort is stable, so patients with
    equal file counts keep their glob() order -- this is the order dependence that makes
    the GAAE split unreproducible (see README).
    """
    if len(patient_dict) < 2:
        return list(patient_dict.keys()), []
    sorted_patients = sorted(patient_dict.items(), key=lambda x: len(x[1]), reverse=True)
    patient_ids = [p[0] for p in sorted_patients]
    actual_test_size = min(test_size, (len(patient_ids) - 1) / len(patient_ids))
    return train_test_split(patient_ids, test_size=actual_test_size, random_state=RANDOM_SEED)


def recreate_gec() -> dict[str, set[str]]:
    mci = get_patient_files(RAW_DIRS["mci"], "*.npz")
    conv = get_patient_files(RAW_DIRS["converter"], "*.npz")

    mci_train, mci_temp = split_ids(list(mci.keys()), test_size=0.4)
    mci_val, mci_test = split_ids(mci_temp, test_size=0.5)
    conv_train, conv_temp = split_ids(list(conv.keys()), test_size=0.4)
    conv_val, conv_test = split_ids(conv_temp, test_size=0.5)

    return {
        "train": set(mci_train) | set(conv_train),
        "validation": set(mci_val) | set(conv_val),
        "test": set(mci_test) | set(conv_test),
    }


def recreate_gaae(gec_test_ids: set[str]) -> dict[str, set[str]]:
    splits: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}

    for raw_dir in RAW_DIRS.values():
        patient_files = get_patient_files(raw_dir, "*pearson*.npz")

        # Step 1: GEC test patients are reserved into GAAE test.
        # Note: GEC *validation* is never reserved -- this is the leak.
        splits["test"] |= {p for p in patient_files if p in gec_test_ids}

        available = {p: f for p, f in patient_files.items() if p not in gec_test_ids}
        available_ids = list(available.keys())

        if len(available_ids) >= 5:
            trainval_ids, test_ids = split_by_files(available, test_size=0.20)
            train_ids, val_ids = split_by_files(
                {p: available[p] for p in trainval_ids}, test_size=0.25
            )
        elif len(available_ids) >= 2:
            train_ids, val_ids = split_by_files(available, test_size=0.33)
            test_ids = []
        else:
            train_ids, val_ids, test_ids = available_ids, [], []

        splits["train"] |= set(train_ids)
        splits["validation"] |= set(val_ids)
        splits["test"] |= set(test_ids)

    return splits


def score(name: str, recreated: dict[str, set[str]], preserved: dict) -> None:
    print(f"\n=== {name}: recreated vs preserved artifact ===")
    for key in ("train", "validation", "test"):
        got = recreated[key]
        want = set(preserved[key].keys())
        jaccard = len(got & want) / len(got | want) if (got | want) else 1.0
        flag = "EXACT" if got == want else f"DIFFERS (+{len(got - want)} / -{len(want - got)})"
        print(
            f"  {key:11}: recreated={len(got):3} preserved={len(want):3} "
            f"jaccard={jaccard:.3f}  {flag}"
        )


def report_leak(gaae_train: set[str], gec: dict) -> None:
    gec_test = set(gec["test"].keys())
    gec_val = set(gec["validation"].keys())
    print("\n=== Leak metric (the finding this whole exercise exists to defend) ===")
    print(
        f"  GAAE train n GEC test = {len(gaae_train & gec_test):3}"
        f"                  (0 expected -- test was protected)"
    )
    pct = 100 * len(gaae_train & gec_val) / len(gec_val) if gec_val else 0
    print(
        f"  GAAE train n GEC val  = {len(gaae_train & gec_val):3} / {len(gec_val)} "
        f"({pct:.0f}%)  <-- the leak: val was never reserved"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional directory to dump the regenerated split ID lists.",
    )
    args = parser.parse_args()

    preserved_gec = json.load(open(HERE / "gec_data_splits.json"))
    preserved_gaae = json.load(open(HERE / "gaae_data_splits.json"))

    gec = recreate_gec()
    score("GEC (downstream)", gec, preserved_gec)

    # GAAE consumed the GEC test set from the *preserved* JSON, not our recreation --
    # using the recreation here would compound GEC's one-patient drift into GAAE.
    gaae = recreate_gaae(set(preserved_gec["test"].keys()))
    score("GAAE (pretrain)", gaae, preserved_gaae)

    print("\n--- leak, from the PRESERVED artifacts (authoritative) ---")
    report_leak(set(preserved_gaae["train"].keys()), preserved_gec)
    print("\n--- leak, from the RECREATED splits (robustness check) ---")
    report_leak(gaae["train"], preserved_gec)

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for name, data in (("gec", gec), ("gaae", gaae)):
            path = args.out / f"recreated_{name}_splits.json"
            json.dump({k: sorted(v) for k, v in data.items()}, open(path, "w"), indent=2)
            print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
