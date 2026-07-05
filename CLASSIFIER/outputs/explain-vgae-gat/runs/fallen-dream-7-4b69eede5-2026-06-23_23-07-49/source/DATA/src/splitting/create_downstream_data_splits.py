"""
Create STRATIFIED train/validation/test splits for downstream classifiers
(GEC, GELSTM, Long-GEC-MLP, LogReg, PROGNOSER) consuming frozen GAAE embeddings.
Patients: MCI (non-converter) vs Converter. AD, healthy, and SCD are excluded.
A patient is a converter if ANY of their visits in cohorts.csv is labeled 'converter'.
A patient is MCI if they have MCI visits but never converted.
Splits are at subject level with stratification by cohort (60/20/20 each).
All scans from the same patient stay together.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DELCODE_DIR = Path(__file__).parents[3] / "DATA" / "DELCODE" / "__fc_wholebrain_sch200_flat__"
MATRICES_DIR = DELCODE_DIR / "matrices"
COHORTS_CSV = DELCODE_DIR / "metadata" / "cohorts.csv"
OUTPUT_DIR = Path(__file__).parents[3] / "DATA" / "DELCODE" / "SPLITS" / "downstream"
# Pre-computed demographics; on first run this file doesn't exist — brthdat fallback is used.
PATIENT_INFO_CSV = OUTPUT_DIR / "_all_splits_patient_info.csv"

RANDOM_SEED = 42


def _patient_groups(cohorts: pd.DataFrame) -> pd.Series:
    """
    Assign each patient a group from their full visit history.
    - converter: ever had a 'converter' visit
    - mci: had 'mci' visits, never a 'converter' visit
    Returns a Series indexed by Pseudonym; patients outside {mci, converter} are dropped.
    """
    def classify(g):
        d = set(g["diagnosis"])
        if "converter" in d:
            return "converter"
        if "mci" in d:
            return "mci"
        return None

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


def _stratified_split(ids: list, test_size: float, random_state: int):
    if len(ids) < 2:
        return ids, []
    actual_test = min(test_size, (len(ids) - 1) / len(ids))
    return train_test_split(ids, test_size=actual_test, random_state=random_state)


def main():
    cohorts = pd.read_csv(COHORTS_CSV)
    groups = _patient_groups(cohorts)

    # Demographics: take first (chronologically earliest) visit per patient.
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
            "converter_status": 1 if group == "converter" else 0,
            "sex": sex,
            "age": age,
            "n_scans": n,
        })

    df = pd.DataFrame(rows)
    print(f"Patients with scans: {len(df)}")
    print(df["diagnosis"].value_counts().to_string())

    train_parts, val_parts, test_parts = [], [], []
    for _diagnosis, group_df in df.groupby("diagnosis"):
        ids = group_df["Pseudonym"].tolist()
        train_ids, temp = _stratified_split(ids, test_size=0.4, random_state=RANDOM_SEED)
        val_ids, test_ids = _stratified_split(temp, test_size=0.5, random_state=RANDOM_SEED)
        train_parts.append(group_df[group_df["Pseudonym"].isin(train_ids)])
        val_parts.append(group_df[group_df["Pseudonym"].isin(val_ids)])
        test_parts.append(group_df[group_df["Pseudonym"].isin(test_ids)])

    train = pd.concat(train_parts, ignore_index=True)
    val = pd.concat(val_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)

    print(f"\nTRAIN : {len(train):3}  {dict(train['diagnosis'].value_counts())}")
    print(f"VAL   : {len(val):3}  {dict(val['diagnosis'].value_counts())}")
    print(f"TEST  : {len(test):3}  {dict(test['diagnosis'].value_counts())}")

    print("\n=== Per-cohort 60/20/20 check ===")
    for cohort in ["mci", "converter"]:
        tf = len(train[train["diagnosis"] == cohort])
        vf = len(val[val["diagnosis"] == cohort])
        xf = len(test[test["diagnosis"] == cohort])
        total = tf + vf + xf
        if total > 0:
            print(f"  {cohort:10}: train={tf} ({tf/total*100:.0f}%), val={vf} ({vf/total*100:.0f}%), test={xf} ({xf/total*100:.0f}%)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUTPUT_DIR / "train.csv", index=False)
    val.to_csv(OUTPUT_DIR / "val.csv", index=False)
    test.to_csv(OUTPUT_DIR / "test.csv", index=False)

    # Merged sex/age lookup over all splits (consumed by GraphDatasetInMemoryFiltered).
    pd.concat([train, val, test], ignore_index=True)[["Pseudonym", "diagnosis", "sex", "age"]] \
        .drop_duplicates("Pseudonym").reset_index(drop=True) \
        .to_csv(OUTPUT_DIR / "_all_splits_patient_info.csv", index=False)

    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
