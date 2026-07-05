"""
Create STRATIFIED train/validation/test splits for the GAAE pretraining model.
Patients: healthy, AD, MCI (non-converter), converter. SCD-only patients are excluded.
A patient's group is derived from their full visit history:
  converter → ever had a 'converter' visit
  mci       → had 'mci' visits, never a 'converter' visit
  ad        → had 'ad' visits, never mci or converter
  healthy   → only healthy visits

Leakage-prevention rule (GAAE is pretrained first; its frozen embeddings feed the
downstream classifiers):
  pretrain train ∩ downstream val  = ∅
  pretrain train ∩ downstream test = ∅
  pretrain val   ∩ downstream test = ∅
  → downstream val  patients are forced into pretrain val  (excluded from pretrain train)
  → downstream test patients are forced into pretrain test (consistent holdout)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DELCODE_DIR = Path(__file__).parents[3] / "DATA" / "DELCODE" / "__fc_wholebrain_sch200_flat__"
MATRICES_DIR = DELCODE_DIR / "matrices"
COHORTS_CSV = DELCODE_DIR / "metadata" / "cohorts.csv"
DOWNSTREAM_SPLITS_DIR = Path(__file__).parents[3] / "DATA" / "DELCODE" / "SPLITS" / "downstream"
OUTPUT_DIR = Path(__file__).parents[3] / "DATA" / "DELCODE" / "SPLITS" / "pretrain"
# Demographics from the downstream run; brthdat fallback is used if not present.
PATIENT_INFO_CSV = DOWNSTREAM_SPLITS_DIR / "_all_splits_patient_info.csv"

RANDOM_SEED = 42


def _patient_groups(cohorts: pd.DataFrame) -> pd.Series:
    """
    Assign each patient a group from their full visit history.
    Returns a Series indexed by Pseudonym; SCD-only/relative patients are excluded (None dropped).
    """
    def classify(g):
        d = set(g["diagnosis"])
        if "converter" in d:
            return "converter"
        if "mci" in d:
            return "mci"
        if "healthy" in d:
            return "healthy"
        if "ad" in d:
            return "ad"
        return None  # SCD-only or relatives → excluded

    return cohorts.groupby("Pseudonym").apply(classify, include_groups=False).dropna()


def _count_scans(patient_id: str) -> int:
    return len(list(MATRICES_DIR.glob(f"sub-{patient_id}_*_z_transformed.npz")))


def _age_from_brthdat(brthdat: str, scan_date: str) -> int:
    """Approximate age — brthdat is pseudonymized so result may be off by several years."""
    try:
        birth_year = int(str(brthdat).split("-")[0])
        scan_year = int(str(scan_date).split("-")[2])  # DD-MM-YYYY
        return scan_year - birth_year
    except (ValueError, IndexError):
        return -1


def _stratified_split_by_scans(patient_dict: dict, test_size: float, random_state: int):
    """Split patients; sort by n_scans descending for balanced scan-count distribution."""
    if len(patient_dict) < 2:
        return list(patient_dict.keys()), []
    ids = sorted(patient_dict, key=lambda p: patient_dict[p], reverse=True)
    actual_test = min(test_size, (len(ids) - 1) / len(ids))
    return train_test_split(ids, test_size=actual_test, random_state=random_state)


def main():
    np.random.seed(RANDOM_SEED)

    # Load downstream holdout sets — these must not appear in pretrain train or pretrain val.
    downstream_test_ids: set = set()
    downstream_val_ids: set = set()
    if (DOWNSTREAM_SPLITS_DIR / "test.csv").exists():
        downstream_test_ids = set(pd.read_csv(DOWNSTREAM_SPLITS_DIR / "test.csv")["Pseudonym"])
    if (DOWNSTREAM_SPLITS_DIR / "val.csv").exists():
        downstream_val_ids = set(pd.read_csv(DOWNSTREAM_SPLITS_DIR / "val.csv")["Pseudonym"])

    print(f"Downstream test patients (→ pretrain test): {len(downstream_test_ids)}")
    print(f"Downstream val  patients (→ pretrain val):  {len(downstream_val_ids)}")

    cohorts = pd.read_csv(COHORTS_CSV)
    groups = _patient_groups(cohorts)

    first_visits = (
        cohorts.drop_duplicates("Pseudonym", keep="first")
        .set_index("Pseudonym")[["sex", "brthdat", "scan_date"]]
    )

    info: dict = {}
    if PATIENT_INFO_CSV.exists():
        _df = pd.read_csv(PATIENT_INFO_CSV)
        _id = "Pseudonym" if "Pseudonym" in _df.columns else "Repseudonym"
        info = _df.set_index(_id)[["sex", "age"]].to_dict("index")

    rows = []
    for pid, group in groups.items():
        n = _count_scans(pid)
        if n == 0:
            continue
        dem = info.get(pid)
        if dem:
            sex = dem["sex"]
            age = dem["age"]
        else:
            fv = first_visits.loc[pid] if pid in first_visits.index else None
            sex = fv["sex"] if fv is not None else ""
            age = _age_from_brthdat(
                fv["brthdat"] if fv is not None else "",
                fv["scan_date"] if fv is not None else "",
            )
        rows.append({
            "Pseudonym": pid,
            "diagnosis": group,
            "sex": sex,
            "age": age,
            "n_scans": n,
        })

    df = pd.DataFrame(rows)
    print(f"\nTotal patients with scans: {len(df)}")
    print(df["diagnosis"].value_counts().to_string())

    # Separate the downstream holdout from patients available for free splitting.
    downstream_test_reserved = df[df["Pseudonym"].isin(downstream_test_ids)]
    downstream_val_reserved = df[df["Pseudonym"].isin(downstream_val_ids)]
    available = df[~df["Pseudonym"].isin(downstream_test_ids | downstream_val_ids)]

    print(f"\nReserved → pretrain test: {len(downstream_test_reserved)}")
    print(f"Reserved → pretrain val:  {len(downstream_val_reserved)}")
    print(f"Available for free split: {len(available)}")

    # Stratified 60/20/20 split of available patients, one cohort at a time.
    train_parts, val_parts, test_parts = [], [downstream_val_reserved], [downstream_test_reserved]

    for _diagnosis, group_df in available.groupby("diagnosis"):
        scan_map = dict(zip(group_df["Pseudonym"], group_df["n_scans"], strict=False))
        ids = list(scan_map.keys())

        if len(ids) >= 5:
            trainval_ids, test_ids = _stratified_split_by_scans(
                scan_map, test_size=0.20, random_state=RANDOM_SEED
            )
            trainval_map = {p: scan_map[p] for p in trainval_ids}
            train_ids, val_ids = _stratified_split_by_scans(
                trainval_map, test_size=0.25, random_state=RANDOM_SEED
            )
        elif len(ids) >= 2:
            train_ids, val_ids = _stratified_split_by_scans(
                scan_map, test_size=0.33, random_state=RANDOM_SEED
            )
            test_ids = []
        else:
            train_ids = ids
            val_ids, test_ids = [], []

        train_parts.append(group_df[group_df["Pseudonym"].isin(train_ids)])
        val_parts.append(group_df[group_df["Pseudonym"].isin(val_ids)])
        test_parts.append(group_df[group_df["Pseudonym"].isin(test_ids)])

    train = pd.concat(train_parts, ignore_index=True)
    val = pd.concat(val_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)

    print(f"\nTRAIN : {len(train):4}  {dict(train['diagnosis'].value_counts())}")
    print(f"VAL   : {len(val):4}  {dict(val['diagnosis'].value_counts())}")
    print(f"TEST  : {len(test):4}  {dict(test['diagnosis'].value_counts())}")

    print("\n=== Per-cohort distribution (free-split patients only) ===")
    holdout = downstream_test_ids | downstream_val_ids
    for cohort in sorted(df["diagnosis"].unique()):
        tf = len(train[train["diagnosis"].eq(cohort) & ~train["Pseudonym"].isin(holdout)])
        vf = len(val[val["diagnosis"].eq(cohort) & ~val["Pseudonym"].isin(holdout)])
        xf = len(test[test["diagnosis"].eq(cohort) & ~test["Pseudonym"].isin(holdout)])
        total = tf + vf + xf
        if total > 0:
            print(f"  {cohort:10}: train={tf} ({tf/total*100:.0f}%), val={vf} ({vf/total*100:.0f}%), test={xf} ({xf/total*100:.0f}%)")

    # Leakage assertions — fail loud if violated.
    pretrain_train_ids = set(train["Pseudonym"])
    pretrain_val_ids_out = set(val["Pseudonym"])
    pretrain_test_ids_out = set(test["Pseudonym"])
    assert len(pretrain_train_ids & downstream_val_ids) == 0,  "LEAK: downstream val patients in pretrain train"
    assert len(pretrain_train_ids & downstream_test_ids) == 0, "LEAK: downstream test patients in pretrain train"
    assert len(pretrain_val_ids_out & downstream_test_ids) == 0, "LEAK: downstream test patients in pretrain val"
    assert downstream_test_ids.issubset(pretrain_test_ids_out), "downstream test not fully covered by pretrain test"
    assert downstream_val_ids.issubset(pretrain_val_ids_out),   "downstream val not fully covered by pretrain val"
    print("\nLeakage assertions passed.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUTPUT_DIR / "train.csv", index=False)
    val.to_csv(OUTPUT_DIR / "val.csv", index=False)
    test.to_csv(OUTPUT_DIR / "test.csv", index=False)

    # Merged sex/age lookup over all splits (consumed by GraphDatasetInMemoryFiltered).
    pd.concat([train, val, test], ignore_index=True)[["Pseudonym", "diagnosis", "sex", "age"]] \
        .drop_duplicates("Pseudonym").reset_index(drop=True) \
        .to_csv(OUTPUT_DIR / "_all_splits_patient_info.csv", index=False)

    print(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
