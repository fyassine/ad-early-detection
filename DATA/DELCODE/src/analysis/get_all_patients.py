import pandas as pd
import glob

excel_files = glob.glob('/mnt/e/fyassine/ad-early-detection/DATA/DELCODE/__metadata__/documentation/*.xlsx')
excel_files.extend(glob.glob('/mnt/e/fyassine/ad-early-detection/DATA/DELCODE/__metadata__/documentation/*/*.xlsx'))

all_pseudonyms = set()

for file in excel_files:
    try:
        xls = pd.ExcelFile(file)
        for sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet, header=None)
            
            # Since header could be on any row, let's just search the whole dataframe for columns that contain 'pseudonym' or 'repseudonym'
            # Or better, just find any cell that matches a pseudonym pattern? No, the patients might not match a strict pattern.
            # Let's assume 'Pseudonym' or 'Repseudonym' is somewhere in the file. 
            
            # If header is row 0 or 1:
            for header_row in [0, 1]:
                df_named = pd.read_excel(xls, sheet_name=sheet, header=header_row)
                cols = [c for c in df_named.columns if isinstance(c, str) and c.lower() in ('pseudonym', 'repseudonym')]
                for c in cols:
                    all_pseudonyms.update(df_named[c].dropna().astype(str).str.strip().tolist())

    except Exception as e:
        print(f"Error reading {file}: {e}")

print(f"Found {len(all_pseudonyms)} unique pseudonyms across all docs.")
