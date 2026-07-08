"""
Build `DATA/DELCODE/__metadata__/cohorts.csv` — the per-visit metadata table consumed
by the CLASSIFIER pipeline (GELSTM dataset, STATIC data-journey notebook).

Consolidates two previously separate, hand-run notebooks into one reproducible script
(no `split` column — nothing downstream produces or consumes one; the real
train/val/test splits live under `DATA/DELCODE/__metadata__/SPLITS/`):

1. Extract phase (formerly `extract_cohorts.ipynb`)
   Reads the raw DELCODE baseline + follow-up Excel exports and the data dictionary,
   selects a fixed set of columns (MMSE, GDS, FAQ, CDR, ApoE, CSF biomarkers,
   ADAS-Cog...), combines baseline + follow-up into one flat table, and merges in
   scan dates from the resting-state scan-dates CSV.

2. Build phase (formerly `create_metadata.ipynb`)
   Takes that combined table plus the converter workbook, excludes relatives
   (prmdiag=100), and derives the `diagnosis` column via the converter rule: a visit
   is `converter` when it matches the converter workbook by (Pseudonym, visdate);
   every later visit for that subject becomes `ad`; everything else maps prmdiag
   0/1/2/5 -> healthy/scd/mci/ad.

3. Fill-missing-visits phase (formerly `DATA/DELCODE/src/transfering/fill_missing_visits.py`)
   Phase 1 only populates `visit`/`scan_date` where a real MRI scan record matches;
   every other row is left blank. This phase fills those blanks with each subject's
   ordinal visit position (`M{12*i}` for the i-th visit chronologically, by
   visdate falling back to scan_date) — rows that already have a scan-matched
   `visit` value are left untouched even if they conflict with that numbering;
   conflicts are only reported, not corrected.

Inputs
------
- DATA/DELCODE/__metadata__/documentation/Antrag_462_Ruat_..._Baseline_repseudonymisiert.xlsx
- DATA/DELCODE/__metadata__/documentation/Antrag_462_Ruat_...Follow-up_repseudonymisiert.xlsx
- DATA/DELCODE/__metadata__/documentation/Dokumentation/20231205_Data_Dictionary_DELCODE.xlsx
- DATA/DELCODE/__metadata__/cohorts_csv/Converter_allvisits.xlsx
- DATA/DELCODE/__metadata__/__artifacts__/restingstate_scan_dates_M0_M60.csv

Outputs
-------
- DATA/DELCODE/__metadata__/cohorts_csv/extracted/baseline_selected_columns.csv
- DATA/DELCODE/__metadata__/cohorts_csv/extracted/followup_selected_columns.csv
- DATA/DELCODE/__metadata__/cohorts_csv/extracted/all_visits_selected_columns.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

REPO_ROOT = Path(__file__).resolve().parents[4]
METADATA_DIR = REPO_ROOT / "DATA" / "DELCODE" / "__metadata__"
DOCS_DIR = METADATA_DIR / "documentation"
DOKUMENTATION_DIR = DOCS_DIR / "Dokumentation"
COHORTS_WORK_DIR = METADATA_DIR / "cohorts_csv"
ARTIFACTS_DIR = METADATA_DIR / "__artifacts__"
EXTRACTED_DIR = COHORTS_WORK_DIR / "extracted"

BL_EXCEL = DOCS_DIR / "Antrag_462_Ruat_DysConnectivity Index_20240320_DELCODE_Baseline_repseudonymisiert.xlsx"
FU_EXCEL = DOCS_DIR / "Antrag_462_Ruat_DysConnectivity Index_20240320DELCODE_Follow-up_repseudonymisiert.xlsx"
NICHT_FREIGEGEBEN_EXCEL = DOCS_DIR / "Antrag 462_Ruat_MRT-Subjects_nicht-freigegeben.xlsx"
DOC_DICT_PATH = DOKUMENTATION_DIR / "20231205_Data_Dictionary_DELCODE.xlsx"
CONVERTER_WORKBOOK = COHORTS_WORK_DIR / "Converter_allvisits.xlsx"
SCAN_DATES_CSV = METADATA_DIR / "restingstate_scan_dates_M0_M60.csv"

BL_SELECTED_PATH = EXTRACTED_DIR / "baseline_selected_columns.csv"
FU_SELECTED_PATH = EXTRACTED_DIR / "followup_selected_columns.csv"
COMBINED_SELECTED_PATH = EXTRACTED_DIR / "all_visits_selected_columns.csv"

OUTPUT_ALL_CSV = METADATA_DIR / "cohorts_all.csv"
OUTPUT_WITH_SCANS_ON_DISK_CSV = METADATA_DIR / "cohorts_with_scans_on_disk.csv"
FMRI_DIR = REPO_ROOT / "DATA" / "DELCODE" / "__fmri_wholebrain_sch200_flat__" / "fmri"

CONVERTER_ID_COLUMN = "Repseudonym"
CONVERTER_DATE_COLUMN = "visdat"
ALL_VISITS_ID_COLUMN = "Pseudonym"
ALL_VISITS_DATE_COLUMN = "visdate"

VISIT_MAP = {
    "M0": "BASELINE",
    "M12": "Follow-up 1",
    "M24": "Follow-up 2",
    "M36": "Follow-up 3",
    "M48": "Follow-up 4",
    "M60": "Follow-up 5",
}

SHARED_GROUPS = {
    "MMSE": ["mmstot", "mmsort", "mmsorp", "mmsreg", "mmsac", "mmsrl", "mmslng", "mmsdw"],
    "GDS": ["gdstot", "gdsframe"] + [f"gds{i}" for i in range(1, 16)],
    "FAQ": ["faqtot"] + [f"faq{i}" for i in range(1, 11)] + ["primsrc", "prmsoth"],
    "CDR": ["cdrtot", "cdrglobal"] + [f"cdr010{i}" for i in range(1, 7)],
    "ApoE": ["ApoE"],
    "Neurodegenerationsmarker": [
        "Abeta38", "Abeta40", "Abeta42",
        "totaltau", "phosphotau181",
        "ratio_Abeta42_40", "ratio_Abeta42_phosphotau181",
    ],
    "ADAS-Cog": ["AD11SUM", "AD13SUM"],
    "Abeta 40/42 (MSD)": ["AB40-MSD pg/ml", "AB42-MSD pg/ml", "Ratio42_40-MSD"],
}

BL_COLUMNS = {
    # Basisdaten
    0: "Pseudonym", 3: "visdate", 6: "sex", 7: "brthdat", 17: "prmdiag",
    # MMSE
    165: "mmstot", 166: "mmsort", 167: "mmsorp", 168: "mmsreg",
    169: "mmsac", 170: "mmsrl", 171: "mmslng", 172: "mmsdw",
    # GDS
    173: "gdstot", 174: "gdsframe",
    175: "gds1", 176: "gds2", 177: "gds3", 178: "gds4", 179: "gds5",
    180: "gds6", 181: "gds7", 182: "gds8", 183: "gds9", 184: "gds10",
    185: "gds11", 186: "gds12", 187: "gds13", 188: "gds14", 189: "gds15",
    # FAQ
    248: "faqtot",
    249: "faq1", 250: "faq2", 251: "faq3", 252: "faq4", 253: "faq5",
    254: "faq6", 255: "faq7", 256: "faq8", 257: "faq9", 258: "faq10",
    259: "primsrc", 260: "prmsoth",
    # CDR
    261: "cdrtot", 262: "cdrglobal",
    263: "cdr0101", 264: "cdr0102", 265: "cdr0103",
    266: "cdr0104", 267: "cdr0105", 268: "cdr0106",
    # ApoE
    777: "ApoE",
    # Neurodegenerationsmarker
    778: "Abeta38", 780: "Abeta40", 782: "Abeta42",
    784: "totaltau", 786: "phosphotau181",
    788: "ratio_Abeta42_40", 789: "ratio_Abeta42_phosphotau181",
    # ADAS-Cog
    790: "AD11SUM", 791: "AD13SUM",
    # Abeta 40/42 (MSD)
    822: "AB40-MSD pg/ml", 823: "AB42-MSD pg/ml", 824: "Ratio42_40-MSD",
}

FU_COLUMNS = {
    # Basisdaten
    0: "Pseudonym", 2: "visnam", 3: "visdate", 7: "sex", 8: "brthdat", 18: "prmdiag",
    # MMSE
    577: "mmstot", 578: "mmsort", 579: "mmsorp", 580: "mmsreg",
    581: "mmsac", 582: "mmsrl", 583: "mmslng", 584: "mmsdw",
    # GDS
    326: "gdstot", 327: "gdsframe",
    328: "gds1", 329: "gds2", 330: "gds3", 331: "gds4", 332: "gds5",
    333: "gds6", 334: "gds7", 335: "gds8", 336: "gds9", 337: "gds10",
    338: "gds11", 339: "gds12", 340: "gds13", 341: "gds14", 342: "gds15",
    # FAQ
    282: "faqtot",
    283: "faq1", 284: "faq2", 285: "faq3", 286: "faq4", 287: "faq5",
    288: "faq6", 289: "faq7", 290: "faq8", 291: "faq9", 292: "faq10",
    610: "primsrc", 611: "prmsoth",
    # CDR
    167: "cdrtot", 168: "cdrglobal",
    169: "cdr0101", 170: "cdr0102", 171: "cdr0103",
    172: "cdr0104", 173: "cdr0105", 174: "cdr0106",
    # Neurodegenerationsmarker
    941: "Abeta38", 943: "Abeta40", 945: "Abeta42",
    947: "totaltau", 949: "phosphotau181",
    951: "ratio_Abeta42_40", 952: "ratio_Abeta42_phosphotau181",
    # ADAS-Cog
    938: "AD11SUM", 939: "AD13SUM",
    # Follow-up extras
    940: "pacc5", 953: "ND_Assay", 954: "Kommentar",
}


def _to_dd_mm_yyyy(series: pd.Series) -> pd.Series:
    """Parse mixed ISO (YYYY-MM-DD) / German (DD-MM-YYYY) dates, normalize to DD-MM-YYYY."""
    s = series.astype(str).str.strip().str.slice(0, 10)
    dt_iso = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    dt_dmy = pd.to_datetime(s, format="%d-%m-%Y", errors="coerce")
    dt = dt_iso.fillna(dt_dmy)
    return dt.dt.strftime("%d-%m-%Y").fillna("")


def read_excel_with_header(path: Path) -> pd.DataFrame:
    """Read a DELCODE Excel export where row 0 = section headers, row 1 = column names."""
    raw = pd.read_excel(path, header=None)
    header = raw.iloc[1].values
    df = raw.iloc[2:].copy()
    df.columns = header
    df = df.reset_index(drop=True)
    print(f"  {Colors.CYAN}Loaded {path.name}: raw {raw.shape} -> data {df.shape}{Colors.RESET}")
    return df


def read_documented_variables(path: Path) -> set:
    docs = pd.read_excel(path)
    return set(docs["Variablenname"].dropna().astype(str))


def report_documentation(variable_names, documented_variables, label) -> None:
    documented = [name for name in variable_names if name in documented_variables]
    undocumented = [name for name in variable_names if name not in documented_variables]
    print(f"  {label}: documented {len(documented)} / {len(variable_names)} selected variables")
    if undocumented:
        print(f"  {Colors.YELLOW}Not found in data dictionary:{Colors.RESET}")
        for v in undocumented:
            print(f"    {Colors.YELLOW}• {v}{Colors.RESET}")


def select_columns(df: pd.DataFrame, columns_by_index: dict) -> pd.DataFrame:
    """Select columns by positional index and rename them."""
    out = df.iloc[:, list(columns_by_index.keys())].copy()
    out.columns = list(columns_by_index.values())
    return out


def _build_group_map(groups_dict: dict) -> dict:
    """Build {column: group_name} from a dict of {group: [columns]}."""
    group_map = {}
    for group_name, columns in groups_dict.items():
        for column_name in columns:
            group_map[column_name] = group_name
    return group_map


def save_with_multiindex(df: pd.DataFrame, group_map: dict, path: Path) -> None:
    """Save a copy of the DataFrame with a MultiIndex column header."""
    df_to_save = df.copy()
    group_row = [group_map[column_name] for column_name in df_to_save.columns]
    df_to_save.columns = pd.MultiIndex.from_arrays([group_row, df_to_save.columns])
    df_to_save.to_csv(path, index=False)
    print(f"  {Colors.GREEN}✓ Saved: {path}{Colors.RESET}")
    print(f"    {len(df_to_save)} rows × {len(df_to_save.columns)} cols")
    print(f"    Read back with: pd.read_csv('{path}', header=[0, 1])")


def combine_baseline_followup(df_bl: pd.DataFrame, df_fu: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Tag, union, sort, and save baseline + follow-up selected columns."""
    bl = df_bl.copy()
    fu = df_fu.copy()
    bl["file"] = "BASELINE"
    fu["file"] = "FOLLOWUP"

    bl_cols = [column_name for column_name in bl.columns if column_name != "file"]
    fu_extra = [column_name for column_name in fu.columns if column_name not in bl_cols and column_name != "file"]
    ordered_columns = ["file"] + bl_cols + fu_extra

    bl = bl.reindex(columns=ordered_columns)
    fu = fu.reindex(columns=ordered_columns)
    combined = pd.concat([bl, fu], ignore_index=True)

    combined["_dt"] = pd.to_datetime(combined["visdate"], errors="coerce")
    combined["_fo"] = combined["file"].map({"BASELINE": 0, "FOLLOWUP": 1}).fillna(9)
    combined = combined.sort_values(["Pseudonym", "_dt", "_fo"], kind="mergesort").reset_index(drop=True)
    combined = combined.drop(columns=["_dt", "_fo"])
    combined.to_csv(out_path, index=False)

    print(f"  {Colors.GREEN}✓ Saved combined selected table: {out_path}{Colors.RESET}")
    print(f"    {len(combined)} rows × {len(combined.columns)} cols")
    print()
    for label, count in combined["file"].value_counts().items():
        print(f"    {label:<12} {count:>5} rows")
    return combined


def get_all_documented_pseudonyms() -> set:
    """Extract all unique patient pseudonyms across all sheets of the main documentation files."""
    files = [BL_EXCEL, FU_EXCEL, NICHT_FREIGEGEBEN_EXCEL]
    all_pseudonyms = set()
    
    for file_path in files:
        if not file_path.exists():
            continue
        try:
            xls = pd.ExcelFile(file_path)
            for sheet in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet, header=None)
                
                # Locate columns that contain 'pseudonym' or 'repseudonym' in the first 5 rows
                col_indices = set()
                for row_idx in range(min(5, len(df))):
                    for col_idx in df.columns:
                        val = str(df.iat[row_idx, col_idx]).lower().strip()
                        if val in ('pseudonym', 'repseudonym'):
                            col_indices.add(col_idx)
                
                # Extract all values from those columns
                for col_idx in col_indices:
                    series = df.iloc[:, col_idx].dropna().astype(str).str.strip()
                    all_pseudonyms.update(series.tolist())
        except Exception as e:
            print(f"{Colors.YELLOW}Warning: Could not read {file_path.name} fully: {e}{Colors.RESET}")
            
    # Clean up the set — remove sentinel strings and obvious header artefacts.
    # Known bad values (case-insensitive):
    invalid = {'nan', 'nat', 'none', '', 'pseudonym', 'repseudonym', 'basis', 'basisdaten'}
    # Also drop anything that looks like a plain word (all letters, no digits) or is too short,
    # since real DELCODE pseudonyms are 9-char hex-ish strings containing at least one digit.
    def _looks_like_pseudonym(p: str) -> bool:
        if p.lower() in invalid:
            return False
        if len(p) < 6:
            return False
        if p.isalpha():          # pure letters → header artefact
            return False
        return True
    all_pseudonyms = {p for p in all_pseudonyms if _looks_like_pseudonym(p)}
    
    return all_pseudonyms


def extract_selected_columns() -> pd.DataFrame:
    """Phase 1 (formerly extract_cohorts.ipynb): select + combine baseline/follow-up columns."""
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    documented_variables = read_documented_variables(DOC_DICT_PATH)

    # ── Baseline ──────────────────────────────────────────────────────────────
    print(f"\n{Colors.BOLD}{'─' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}  BASELINE{Colors.RESET}")
    print(f"{Colors.BOLD}{'─' * 60}{Colors.RESET}")
    df_bl_raw = read_excel_with_header(BL_EXCEL)
    df_bl = select_columns(df_bl_raw, BL_COLUMNS)
    report_documentation(df_bl.columns.tolist(), documented_variables, "Baseline selection")
    print(f"  {Colors.GREEN}Baseline selected: {df_bl.shape}{Colors.RESET}")

    bl_groups = {
        "Basisdaten": ["Pseudonym", "visdate", "sex", "brthdat", "prmdiag"],
        **SHARED_GROUPS,
    }
    save_with_multiindex(df_bl, _build_group_map(bl_groups), BL_SELECTED_PATH)

    # ── Follow-up ─────────────────────────────────────────────────────────────
    print(f"\n{Colors.BOLD}{'─' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}  FOLLOW-UP{Colors.RESET}")
    print(f"{Colors.BOLD}{'─' * 60}{Colors.RESET}")
    df_fu_raw = read_excel_with_header(FU_EXCEL)
    df_fu = select_columns(df_fu_raw, FU_COLUMNS)

    # Add empty columns for baseline-only fields so selected outputs stay aligned.
    for column_name in ["ApoE", "AB40-MSD pg/ml", "AB42-MSD pg/ml", "Ratio42_40-MSD"]:
        df_fu[column_name] = np.nan

    report_documentation(df_fu.columns.tolist(), documented_variables, "Follow-up selection")
    print(f"  {Colors.GREEN}Follow-up selected: {df_fu.shape}{Colors.RESET}")
    unique_visits = sorted(df_fu["visnam"].dropna().unique())
    print(f"  Unique visits ({len(unique_visits)}):")
    for v in unique_visits:
        print(f"    • {v}")

    fu_groups = {
        "Basisdaten": ["Pseudonym", "visnam", "visdate", "sex", "brthdat", "prmdiag"],
        **SHARED_GROUPS,
        "NPT_Scores": ["pacc5"],
        "Follow-Up Extra": ["ND_Assay", "Kommentar"],
    }
    save_with_multiindex(df_fu, _build_group_map(fu_groups), FU_SELECTED_PATH)

    # ── Combine ───────────────────────────────────────────────────────────────
    print(f"\n{Colors.BOLD}{'─' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}  COMBINING BASELINE + FOLLOW-UP{Colors.RESET}")
    print(f"{Colors.BOLD}{'─' * 60}{Colors.RESET}")
    comb = combine_baseline_followup(df_bl, df_fu, COMBINED_SELECTED_PATH)

    # ── Missing patients ──────────────────────────────────────────────────────
    print(f"\n{Colors.BOLD}{'─' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}  PATIENT RECONCILIATION{Colors.RESET}")
    print(f"{Colors.BOLD}{'─' * 60}{Colors.RESET}")
    print("  Collecting all documented pseudonyms...")
    all_documented_ids = get_all_documented_pseudonyms()
    existing_ids = set(comb["Pseudonym"].astype(str).str.strip().unique())
    missing_ids = sorted(all_documented_ids - existing_ids)

    if missing_ids:
        print(f"  {Colors.YELLOW}⚠  {len(missing_ids)} patient(s) found in documentation but not in main data:{Colors.RESET}")
        print()
        col_w = 3 + len(str(len(missing_ids)))  # width of the index column
        print(f"  {'#':<{col_w}}  {'Pseudonym'}")
        print(f"  {'─' * col_w}  {'─' * 20}")
        for i, pid in enumerate(missing_ids, 1):
            print(f"  {Colors.YELLOW}{i:<{col_w}}  {pid}{Colors.RESET}")
        print()
        print(f"  Adding {len(missing_ids)} row(s) with NaN clinical data...")
        missing_df = pd.DataFrame({"Pseudonym": missing_ids})
        missing_df["file"] = "BASELINE"
        missing_df["visnam"] = "BASELINE"
        # Since we added new rows with potentially uninitialized columns, concat will fill them with NaN
        comb = pd.concat([comb, missing_df], ignore_index=True)
    else:
        print(f"  {Colors.GREEN}✓ All documented patients are present in the main data.{Colors.RESET}")

    # ── Scan date merge ───────────────────────────────────────────────────────
    print(f"\n{Colors.BOLD}{'─' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}  SCAN DATE MERGE{Colors.RESET}")
    print(f"{Colors.BOLD}{'─' * 60}{Colors.RESET}")
    scans = pd.read_csv(SCAN_DATES_CSV, usecols=["pseudonym", "visit", "run", "scan_date"])
    scans = scans[scans["run"].eq("T1_01")].copy()
    scans["scan_date"] = pd.to_datetime(scans["scan_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    scans["_merge_key"] = scans["visit"].map(VISIT_MAP)
    scans = scans.dropna(subset=["scan_date", "_merge_key"])
    scans = scans.sort_values(["pseudonym", "_merge_key", "run"], kind="mergesort")
    scans = scans.drop_duplicates(subset=["pseudonym", "_merge_key"], keep="first")

    comb["_merge_key"] = comb["visnam"].fillna("BASELINE")
    comb = comb.merge(
        scans[["pseudonym", "_merge_key", "scan_date", "visit"]],
        left_on=["Pseudonym", "_merge_key"],
        right_on=["pseudonym", "_merge_key"],
        how="left",
    )
    comb = comb.drop(columns=["pseudonym", "_merge_key"])

    columns = comb.columns.tolist()
    visdate_index = columns.index("visdate")
    scan_date_values = comb.pop("scan_date")
    visit_values = comb.pop("visit")
    comb.insert(visdate_index + 1, "scan_date", scan_date_values)
    comb.insert(visdate_index + 2, "visit", visit_values)

    # Normalise both date columns to DD-MM-YYYY (safe for repeated runs).
    comb["visdate"] = _to_dd_mm_yyyy(comb["visdate"])
    comb["scan_date"] = _to_dd_mm_yyyy(comb["scan_date"])

    # Fallback missing scan_date to visdate
    comb["scan_date"] = comb["scan_date"].replace("", pd.NA).fillna(comb["visdate"]).fillna("")

    comb.to_csv(COMBINED_SELECTED_PATH, index=False)

    matched = (comb["scan_date"] != "").sum()
    total = len(comb)
    print(f"  {Colors.GREEN}✓ Matched: {matched} / {total} rows ({100 * matched / total:.1f}%){Colors.RESET}")
    if total - matched > 0:
        print(f"  {Colors.YELLOW}⚠  Unmatched rows: {total - matched}{Colors.RESET}")
    else:
        print(f"  Unmatched rows: 0")
    return comb


def build_cohorts(all_visits_df: pd.DataFrame) -> pd.DataFrame:
    """Phase 2 (formerly create_metadata.ipynb): derive `diagnosis` via the converter rule."""
    converter_df = pd.read_excel(CONVERTER_WORKBOOK)

    all_visits_df = all_visits_df.copy()
    all_visits_df[ALL_VISITS_ID_COLUMN] = all_visits_df[ALL_VISITS_ID_COLUMN].astype(str).str.strip()
    all_visits_df["_visdate_dt"] = pd.to_datetime(
        all_visits_df[ALL_VISITS_DATE_COLUMN], format="%d-%m-%Y", errors="coerce"
    )
    all_visits_df["prmdiag_num"] = pd.to_numeric(all_visits_df["prmdiag"], errors="coerce")

    # Exclude relatives
    matched_rows = all_visits_df[all_visits_df["prmdiag_num"] != 100].copy()

    # Converter subject IDs and converter visit keys from workbook
    converter_ids = pd.Index(
        converter_df[CONVERTER_ID_COLUMN].dropna().astype(str).str.strip().unique()
    )
    converter_df = converter_df.copy()
    converter_df["_id_key"] = converter_df[CONVERTER_ID_COLUMN].astype(str).str.strip()
    converter_df["_date_key"] = _to_dd_mm_yyyy(converter_df[CONVERTER_DATE_COLUMN])
    converter_visit_keys = set(
        converter_df.loc[converter_df["_date_key"] != "", ["_id_key", "_date_key"]]
        .itertuples(index=False, name=None)
    )

    matched_rows["_date_key"] = matched_rows[ALL_VISITS_DATE_COLUMN]
    matched_rows["_is_converter_subject"] = matched_rows[ALL_VISITS_ID_COLUMN].isin(converter_ids)
    matched_rows["_is_converter_visit"] = [
        (sid, dkey) in converter_visit_keys
        for sid, dkey in zip(matched_rows[ALL_VISITS_ID_COLUMN], matched_rows["_date_key"], strict=False)
    ]

    # First converter visit date per converter subject
    first_converter_dt = (
        matched_rows.loc[matched_rows["_is_converter_visit"]]
        .groupby(ALL_VISITS_ID_COLUMN)["_visdate_dt"]
        .min()
        .rename("_first_converter_dt")
    )
    matched_rows = matched_rows.merge(
        first_converter_dt, left_on=ALL_VISITS_ID_COLUMN, right_index=True, how="left"
    )
    matched_rows["_is_post_converter_visit"] = (
        matched_rows["_is_converter_subject"]
        & (~matched_rows["_is_converter_visit"])
        & matched_rows["_first_converter_dt"].notna()
        & matched_rows["_visdate_dt"].gt(matched_rows["_first_converter_dt"])
    )

    # Base diagnosis mapping for all non-converter-priority rows
    diagnosis = matched_rows["prmdiag_num"].map({0: "healthy", 1: "scd", 2: "mci", 5: "ad"}).fillna("")

    # Apply converter priority rules
    diagnosis = diagnosis.mask(matched_rows["_is_converter_visit"], "converter")
    diagnosis = diagnosis.mask(matched_rows["_is_post_converter_visit"], "ad")

    if "diagnosis" in matched_rows.columns:
        matched_rows = matched_rows.drop(columns=["diagnosis"])
    insert_idx = matched_rows.columns.get_loc("prmdiag") + 1
    matched_rows.insert(insert_idx, "diagnosis", diagnosis)

    matched_rows = matched_rows.sort_values(
        [ALL_VISITS_ID_COLUMN, "_visdate_dt", "file"], kind="mergesort"
    ).reset_index(drop=True)

    matched_rows = matched_rows.drop(
        columns=[
            "_date_key", "_visdate_dt", "_first_converter_dt",
            "_is_converter_subject", "_is_converter_visit", "_is_post_converter_visit",
            "prmdiag_num",
        ]
    )

    missing_ids = sorted(set(converter_ids) - set(matched_rows[ALL_VISITS_ID_COLUMN].unique()))

    print(f"  {Colors.GREEN}Converter workbook rows : {len(converter_df)}{Colors.RESET}")
    print(f"  {Colors.GREEN}Unique converter IDs    : {len(converter_ids)}{Colors.RESET}")
    print(f"  {Colors.GREEN}Rows (excl. relatives)  : {len(matched_rows)}{Colors.RESET}")
    print(f"  {Colors.GREEN}Unique subjects         : {matched_rows[ALL_VISITS_ID_COLUMN].nunique()}{Colors.RESET}")
    print()
    print(f"  {Colors.BOLD}Diagnosis breakdown:{Colors.RESET}")
    for diag, count in matched_rows["diagnosis"].value_counts(dropna=False).items():
        label = str(diag) if diag else "(unknown)"
        bar = "█" * min(count // 50, 30)
        print(f"    {label:<12} {count:>5}  {Colors.CYAN}{bar}{Colors.RESET}")
    if missing_ids:
        print()
        print(f"  {Colors.YELLOW}⚠  Converter IDs not found in output: {len(missing_ids)}{Colors.RESET}")
        for mid in missing_ids[:20]:
            print(f"    {Colors.YELLOW}• {mid}{Colors.RESET}")
    else:
        print(f"  {Colors.GREEN}✓ All converter IDs found in output table.{Colors.RESET}")

    return matched_rows


def _parse_date_series(series: pd.Series) -> pd.Series:
    series = series.astype(str).str.strip()
    series = series.replace({"": pd.NA, ".": pd.NA, "nan": pd.NA, "NaN": pd.NA})
    return pd.to_datetime(series, dayfirst=True, errors="coerce")


def _missing_visit_series(series: pd.Series) -> pd.Series:
    series = series.astype(str).str.strip()
    return series.eq("") | series.eq(".") | series.str.lower().eq("nan")


def _map_visnam(v) -> str:
    if pd.isna(v) or v == 'BASELINE': 
        return 'M0'
    v_clean = str(v).strip()
    is_tel = 'Tel' in v_clean
    v_clean = v_clean.replace(' Tel', '')
    if v_clean.startswith('Follow-up '):
        try:
            num = int(v_clean.replace('Follow-up ', ''))
            visit_str = f'M{num * 12}'
            if is_tel:
                visit_str += '_Tel'
            return visit_str
        except ValueError:
            pass
    return ''

def fill_missing_visits(df: pd.DataFrame) -> pd.DataFrame:
    """Phase 3 (formerly transfering/fill_missing_visits.py): fill blank `visit`
    values with expected visits mapped from `visnam`; leave scan-matched values
    untouched and only report (not fix) conflicts with that mapping."""
    df = df.copy()

    expected_visit = df["visnam"].fillna("BASELINE").apply(_map_visnam)
    missing = _missing_visit_series(df["visit"])
    
    conflicts = (~missing) & (df["visit"].astype(str).str.strip() != expected_visit)
    conflict_cols = [c for c in [ALL_VISITS_ID_COLUMN, "visdate", "scan_date", "visit", "visnam"] if c in df.columns]
    conflict_rows = df.loc[conflicts, conflict_cols].copy()
    conflict_rows["target_visit"] = expected_visit[conflicts].values

    df.loc[missing, "visit"] = expected_visit[missing].values

    filled = int(missing.sum())
    remaining = int(_missing_visit_series(df["visit"]).sum())
    if len(conflict_rows) > 0:
        print(f"  Filled visits  : {filled} → {remaining} missing remaining")
        print(f"  {Colors.YELLOW}⚠  Conflicts     : {len(conflict_rows)}{Colors.RESET}")
        print()
        print(f"{Colors.YELLOW}{conflict_rows.head(10).to_string(index=False)}{Colors.RESET}")
    else:
        print(f"  {Colors.GREEN}✓ Filled {filled} visits — 0 conflicts, 0 remaining missing{Colors.RESET}")

    return df


def check_scan_exists(row: pd.Series) -> bool:
    pid = str(row["Pseudonym"]).strip()
    visit = str(row["visit"]).strip()
    patient_dir = FMRI_DIR / f"sub-{pid}"
    if not patient_dir.is_dir():
        return False
    # Check if a file containing `_{visit}_` exists in this directory
    for f in patient_dir.iterdir():
        if f.is_file() and f"_{visit}_" in f.name:
            return True
    return False


def main() -> None:
    # ═══════════════════════════════════════════════════════════════
    #  PHASE 1 — Extract & combine selected columns
    # ═══════════════════════════════════════════════════════════════
    all_visits = extract_selected_columns()

    # ═══════════════════════════════════════════════════════════════
    #  PHASE 2 — Build cohorts + derive diagnosis
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{Colors.BOLD}{'─' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}  BUILD COHORTS  (converter rule){Colors.RESET}")
    print(f"{Colors.BOLD}{'─' * 60}{Colors.RESET}")
    cohorts = build_cohorts(all_visits)

    # ═══════════════════════════════════════════════════════════════
    #  PHASE 3 — Fill missing visit labels
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{Colors.BOLD}{'─' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}  FILL MISSING VISITS{Colors.RESET}")
    print(f"{Colors.BOLD}{'─' * 60}{Colors.RESET}")
    cohorts = fill_missing_visits(cohorts)

    # ═══════════════════════════════════════════════════════════════
    #  OUTPUT
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{Colors.BOLD}{'─' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}  OUTPUT{Colors.RESET}")
    print(f"{Colors.BOLD}{'─' * 60}{Colors.RESET}")
    cohorts.to_csv(OUTPUT_ALL_CSV, index=False)
    print(f"  {Colors.GREEN}✓ Saved cohorts_all  → {OUTPUT_ALL_CSV}{Colors.RESET}")

    print("  Checking for physical fMRI scans...")
    has_scan = cohorts.apply(check_scan_exists, axis=1)
    cohorts_with_scans_on_disk = cohorts[has_scan]
    cohorts_with_scans_on_disk.to_csv(OUTPUT_WITH_SCANS_ON_DISK_CSV, index=False)
    print(f"  {Colors.GREEN}✓ Saved cohorts_with_scans_on_disk ({len(cohorts_with_scans_on_disk)} rows) → {OUTPUT_WITH_SCANS_ON_DISK_CSV}{Colors.RESET}")
    print(f"\n{Colors.BOLD}{'═' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}  Done.{Colors.RESET}")
    print(f"{Colors.BOLD}{'═' * 60}{Colors.RESET}")


if __name__ == "__main__":
    main()
