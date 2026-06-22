from pathlib import Path

import pandas as pd

FILES = [
    Path("DATA/DELCODE/__fmri_wholebrain_sch200_flat__/metadata/cohorts.csv"),
    Path("DATA/DELCODE/__fc_wholebrain_sch200_flat__/metadata/cohorts.csv"),
    Path("DATA/DELCODE/__fc_dmn_sch200_flat__/metadata/cohorts.csv"),
]


def parse_date_series(series: pd.Series) -> pd.Series:
    series = series.astype(str).str.strip()
    series = series.replace({"": pd.NA, ".": pd.NA, "nan": pd.NA, "NaN": pd.NA})
    return pd.to_datetime(series, dayfirst=True, errors="coerce")


def missing_visit_series(series: pd.Series) -> pd.Series:
    series = series.astype(str).str.strip()
    return series.eq("") | series.eq(".") | series.str.lower().eq("nan")


def subject_column(df: pd.DataFrame) -> str:
    for col in ["Pseudonym", "Repseudonym", "pseudonym", "subject_id"]:
        if col in df.columns:
            return col
    raise ValueError("Subject column not found")


def fill_visits(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int, pd.DataFrame]:
    subject_col = subject_column(df)
    df = df.copy()
    df["_row_id"] = range(len(df))

    visdate = parse_date_series(df["visdate"]) if "visdate" in df.columns else pd.Series([pd.NaT] * len(df))
    scan_date = parse_date_series(df["scan_date"]) if "scan_date" in df.columns else pd.Series([pd.NaT] * len(df))
    df["_date_key"] = visdate.fillna(scan_date)

    target_visits = [""] * len(df)

    for _, group in df.groupby(subject_col, sort=False):
        group_sorted = group.sort_values(by=["_date_key", "_row_id"], na_position="last")
        targets = ["M" + str(i * 12) for i in range(len(group_sorted))]
        for row_id, target in zip(group_sorted["_row_id"].tolist(), targets):
            target_visits[row_id] = target

    target_series = pd.Series(target_visits, index=df.index)

    if "visit" in df.columns:
        missing = missing_visit_series(df["visit"])
    else:
        df["visit"] = ""
        missing = pd.Series([True] * len(df))

    conflicts = (~missing) & (df["visit"].astype(str).str.strip() != target_series)
    conflict_cols = [subject_col, "visdate", "scan_date", "visit", "visnam"]
    conflict_cols = [c for c in conflict_cols if c in df.columns]
    conflict_rows = df.loc[conflicts, conflict_cols].copy()
    conflict_rows["target_visit"] = target_series[conflicts].values

    df.loc[missing, "visit"] = target_series[missing].values

    df = df.drop(columns=["_row_id", "_date_key"])

    after_missing = missing_visit_series(df["visit"]).sum()
    return df, int(missing.sum()), int(after_missing), conflict_rows


def main() -> None:
    for path in FILES:
        df = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
        )
        updated, missing_before, missing_after, conflicts = fill_visits(df)
        updated.to_csv(path, index=False)
        print(f"{path}: missing visit before={missing_before} after={missing_after} conflicts={len(conflicts)}")
        if len(conflicts) > 0:
            sample = conflicts.head(10)
            print(sample.to_string(index=False))
            print("")


if __name__ == "__main__":
    main()
