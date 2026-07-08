import os
import csv
import re

def main():
    csv_file = '/mnt/e/fyassine/ad-early-detection/DATA/DELCODE/__metadata__/cohorts.csv'
    fmri_dir = '/mnt/e/fyassine/ad-early-detection/DATA/DELCODE/__fmri_wholebrain_sch200_flat__/fmri'
    
    out_file1 = '/mnt/e/fyassine/ad-early-detection/DATA/DELCODE/__metadata__/inside_cohorts_csv_with_scandate_but_not_inside_fmri_wholebrain.csv'
    out_file2 = '/mnt/e/fyassine/ad-early-detection/DATA/DELCODE/__metadata__/inside_fmri_wholebrain_but_not_in_cohorts_csv.csv'

    # 1. Parse CSV
    csv_records = {} # (pid, visit) -> {scan_date, diagnosis}
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row['Pseudonym'].strip()
            visit = row['visit'].strip()
            scan_date = row['scan_date'].strip()
            diagnosis = row['diagnosis'].strip()
            
            csv_records[(pid, visit)] = {
                'scan_date': scan_date,
                'diagnosis': diagnosis
            }

    # 2. Parse fmri directory
    fmri_scans = set() 
    
    for dname in os.listdir(fmri_dir):
        if dname.startswith('sub-'):
            pid = dname[4:]
            patient_dir = os.path.join(fmri_dir, dname)
            if os.path.isdir(patient_dir):
                for fname in os.listdir(patient_dir):
                    match = re.search(r'_(M\d+)_', fname)
                    if match:
                        visit = match.group(1)
                        fmri_scans.add((pid, visit))

    # 3. Find mismatches
    in_csv_no_scan = []
    in_scan_no_csv = []
    
    # Load raw extracted data to identify relatives (prmdiag == 100)
    raw_df_path = '/mnt/e/fyassine/ad-early-detection/DATA/DELCODE/__metadata__/cohorts_csv/extracted/all_visits_selected_columns.csv'
    relatives = set()
    if os.path.exists(raw_df_path):
        with open(raw_df_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('prmdiag', '').strip() == '100':
                    relatives.add(row.get('Pseudonym', '').strip())

    # Check CSV -> fMRI
    for (pid, visit), data in csv_records.items():
        if data['scan_date'] != '':
            if (pid, visit) not in fmri_scans:
                in_csv_no_scan.append([pid, visit, data['scan_date'], data['diagnosis']])
                
    # Check fMRI -> CSV
    for (pid, visit) in fmri_scans:
        if (pid, visit) not in csv_records:
            if pid not in relatives:
                in_scan_no_csv.append([pid, visit, "Not in CSV", "N/A"])
        elif csv_records[(pid, visit)]['scan_date'] == '':
            in_scan_no_csv.append([pid, visit, "Empty in CSV", csv_records[(pid, visit)]['diagnosis']])

    # 4. Write output files
    with open(out_file1, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Patient', 'Visit', 'Scan_Date_in_CSV', 'Diagnosis'])
        for row in in_csv_no_scan:
            writer.writerow(row)
            
    with open(out_file2, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Patient', 'Visit', 'Status', 'Diagnosis_in_CSV'])
        for row in in_scan_no_csv:
            writer.writerow(row)
            
    print(f"File 1 (in CSV with date, no scan): {len(in_csv_no_scan)} records saved to {out_file1}")
    print(f"File 2 (has scan, not in CSV / no date): {len(in_scan_no_csv)} records saved to {out_file2}")

if __name__ == '__main__':
    main()
