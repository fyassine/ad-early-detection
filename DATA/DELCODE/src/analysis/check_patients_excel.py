import pandas as pd
import glob

excel_files = glob.glob('/mnt/e/fyassine/ad-early-detection/DATA/DELCODE/__metadata__/documentation/*.xlsx')
excel_files.extend(glob.glob('/mnt/e/fyassine/ad-early-detection/DATA/DELCODE/__metadata__/documentation/*/*.xlsx'))

# Patients from inside_fmri_wholebrain_but_not_in_cohorts_csv.csv
patients_not_in_csv = ["fd0a4ba8a", "d7f5479f6", "bc789238a", "d621c4094"]
patients_empty_in_csv = ["cc44a391f", "6bf64d0cc", "b3686755c", "12c8dab7c", "c0dbb0bf3", "08df48201", "661b950b8", "5a9a5f91b", "f3a9124c6", "434050aec", "da07ac132", "74dd3f52d", "475b9bfe2", "073b63746", "dca63a3ab", "b093bca1e", "fcb64766d", "97beef830", "58aedadbc", "4d9cb00ce", "5b0a86468", "6ecf96ab0", "e25c67e62", "bc177df98", "c17579b36", "4e29e3fcf"]

all_patients = set(patients_not_in_csv + patients_empty_in_csv)

print(f"Scanning {len(excel_files)} excel files for {len(all_patients)} patients...")
for file in excel_files:
    try:
        xls = pd.ExcelFile(file)
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            df_str = df.astype(str)
            
            for p in all_patients:
                mask = df_str.apply(lambda col: col.str.contains(p, case=False, na=False))
                if mask.any().any():
                    status = "Not in CSV" if p in patients_not_in_csv else "Empty in CSV"
                    print(f"[{status}] Found patient {p} in file '{file}', sheet '{sheet_name}'")
    except Exception as e:
        print(f"Error reading {file}: {e}")
