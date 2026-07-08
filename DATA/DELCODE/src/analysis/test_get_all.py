import pandas as pd
from pathlib import Path

REPO_ROOT = Path('/mnt/e/fyassine/ad-early-detection')
METADATA_DIR = REPO_ROOT / "DATA" / "DELCODE" / "__metadata__"
DOCS_DIR = METADATA_DIR / "documentation"

BL_EXCEL = DOCS_DIR / "Antrag_462_Ruat_DysConnectivity Index_20240320_DELCODE_Baseline_repseudonymisiert.xlsx"
FU_EXCEL = DOCS_DIR / "Antrag_462_Ruat_DysConnectivity Index_20240320DELCODE_Follow-up_repseudonymisiert.xlsx"
EXTRA_EXCEL = DOCS_DIR / "Antrag 462_Ruat_MRT-Subjects_nicht-freigegeben.xlsx"

def get_all_documented_pseudonyms() -> set:
    files = [BL_EXCEL, FU_EXCEL, EXTRA_EXCEL]
    all_pseudonyms = set()
    for file in files:
        if not file.exists():
            continue
        print(f"Reading {file.name}")
        xls = pd.ExcelFile(file)
        for sheet in xls.sheet_names:
            df_peek = pd.read_excel(xls, sheet_name=sheet, nrows=5, header=None)
            
            header_to_use = None
            col_indices = []
            for hr in range(len(df_peek)):
                cols = [str(c).lower().strip() for c in df_peek.iloc[hr].values]
                for idx, c in enumerate(cols):
                    if c in ('pseudonym', 'repseudonym'):
                        col_indices.append(idx)
                
                if col_indices:
                    header_to_use = hr
                    break
            
            if header_to_use is not None and col_indices:
                df = pd.read_excel(xls, sheet_name=sheet, header=header_to_use, usecols=col_indices)
                for col in df.columns:
                    all_pseudonyms.update(df[col].dropna().astype(str).str.strip().tolist())
    
    all_pseudonyms = {p for p in all_pseudonyms if p and p.lower() not in ('nan', 'nat', 'none', '')}
    return all_pseudonyms

print(len(get_all_documented_pseudonyms()))
