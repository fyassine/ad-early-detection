# OASIS-3 Clinical Metadata and Diagnoses

This directory contains the clinical, cognitive, and demographic metadata spreadsheets for the OASIS-3 dataset downloaded from NITRC-IR. These files can be linked with MRI/PET scan sessions using the **"days from entry"** metric.

> 💡 **Tip:** A pre-built, ready-to-use CSV linking every downloaded fMRI scan to its matched clinical visit is available at [visit_level_diagnoses.csv](file:///mnt/e/fyassine/ad-early-detection/DATA/OASIS3/__metadata__/visit_level_diagnoses.csv).
> To regenerate it, run `python3 DATA/OASIS3/__metadata__/build_visit_diagnosis.py` from the project root.

---

## 1. Clinician Diagnostic and Staging Files

### A. [OASIS3_UDSb4_cdr.csv](file:///mnt/e/fyassine/ad-early-detection/DATA/OASIS3/__metadata__/OASIS3_UDSb4_cdr.csv)
Contains the Clinical Dementia Rating (CDR) scores, MMSE scores, and clinician diagnoses.
* **`CDRTOT` (Global CDR Score):** Primary indicator of cognitive status:
  * `0.0` = Cognitively Normal / Healthy
  * `0.5` = Very Mild Dementia / Mild Cognitive Impairment (MCI)
  * `1.0` = Mild Dementia
  * `2.0` = Moderate Dementia
  * `3.0` = Severe Dementia
* **`CDRSUM` (CDR Sum of Boxes):** The sum of the sub-domain scores (memory, orientation, judgment, etc.), offering a more granular scale of impairment (ranging from 0 to 18).
* **`MMSE`:** Mini-Mental State Examination score (range: 0 to 30, lower scores indicate worse impairment).
* **`dx1_code` & `dx1` (Clinician Primary Diagnosis):**
  * `1.0` = Cognitively normal
  * `2.0` = Uncertain dementia
  * `3.0` = AD Dementia (Alzheimer's Disease Dementia)
  * Other codes correspond to vascular dementia (`122.0`), dementia with Lewy bodies (`126.0`), and other specific/mixed etiologies.

### B. [OASIS3_UDSd1_diagnoses.csv](file:///mnt/e/fyassine/ad-early-detection/DATA/OASIS3/__metadata__/OASIS3_UDSd1_diagnoses.csv)
Contains NACC UDS Form D1 (Clinician Diagnosis - Cognitive Status and Dementia) details.
* **`NORMCOG`:** Normal Cognition indicator (`1` = Yes, `0` = No).
* **`DEMENTED`:** Clinically Demented indicator (`1` = Yes, `0` = No).
* **`PROBAD` / `POSSAD`:** Probable / Possible Alzheimer's Disease etiology (`1` = Yes, `0` = No).
* **`MCIAMEM` / `MCINON1` / etc.:** Indicators for amnestic/non-amnestic MCI subtypes.

### C. [OASIS3_demographics.csv](file:///mnt/e/fyassine/ad-early-detection/DATA/OASIS3/__metadata__/OASIS3_demographics.csv)
Provides overall study demographics (gender, years of education, handedness).

---

## ⚠️ **Important:**
## 2. Consensus Diagnoses (Agreed Diagnoses)

In the OASIS-3 dataset (from the Knight ADRC cohort), diagnostic decisions are determined by a **consensus panel** of expert clinicians, neuropsychologists, and researchers. A consensus clinical diagnosis is rendered for each participant at each longitudinal evaluation.
>  The agreed diagnosis is primarily stored in the **`dx1` / `dx1_code`** columns of the `UDSb4_cdr.csv` file, and is backed up by **`NORMCOG`**, **`DEMENTED`**, and **`PROBAD`** columns in the `UDSd1_diagnoses.csv` file.

---

## 3. How to Identify Diagnostic Cohorts Longitudinally

Since OASIS-3 contains longitudinal observations, participants are classified into cohorts based on their progression profiles. Below is a Python code snippet that loads the clinical database and extracts the four key patient groups:

### Python Script to Filter Cohorts
You can run this python script directly in the repository to identify the cohorts:

```python
import pandas as pd

# Load clinical metadata
df = pd.read_csv("DATA/OASIS3/__metadata__/OASIS3_UDSb4_cdr.csv")

# Sort chronologically per subject
df = df.sort_values(by=["OASISID", "days_to_visit"])

# Group by subject and find their diagnostic history
cohorts = {
    "sHC": [],   # Stable Healthy Controls
    "sMCI": [],  # Stable MCI
    "cMCI": [],  # MCI Converters to AD
    "sAD": [],   # Stable/Progressive AD Dementia
}

for subj, group in df.groupby("OASISID"):
    cdrs = group["CDRTOT"].dropna().tolist()
    dxs = group["dx1"].dropna().tolist()
    
    if not cdrs:
        continue
        
    # Check if subject ever had an AD/DAT diagnosis
    ever_ad = any("AD" in str(d) or "DAT" in str(d) for d in dxs)
    ever_demented = any(c >= 1.0 for c in cdrs)
    
    first_cdr = cdrs[0]
    
    # 1. Stable Healthy Control (sHC) - always CDR = 0.0
    if all(c == 0.0 for c in cdrs):
        cohorts["sHC"].append(subj)
        
    # 2. MCI Converter to AD (cMCI) - starts at CDR 0.5, later progresses to dementia with AD diagnosis
    elif first_cdr == 0.5 and ever_demented and ever_ad:
        cohorts["cMCI"].append(subj)
        
    # 3. Stable MCI (sMCI) - has CDR 0.5, but never progresses beyond CDR 0.5 and never gets AD diagnosis
    elif 0.5 in cdrs and not ever_demented and not ever_ad:
        cohorts["sMCI"].append(subj)
        
    # 4. Stable/Progressive AD Dementia (sAD) - enters study with CDR >= 0.5 and AD diagnosis
    elif first_cdr >= 0.5 and ever_ad:
        cohorts["sAD"].append(subj)

# Print Summary Counts
for group_name, subjects in cohorts.items():
    print(f"{group_name}: {len(subjects)} subjects")
```

### Cohort Distributions in the Dataset:
* **Stable Healthy Control (sHC):** `755` subjects
* **Stable/Progressive AD Dementia (sAD):** `191` subjects
* **MCI Converter to AD (cMCI):** `150` subjects
* **Stable MCI (sMCI):** `131` subjects

---

## 4. Reading and Matching Clinical Data to MRI/PET scans

All dates are tracked relative to the participant's study entry using a **"days from entry"** label (`dXXXX`):
* `OAS30001_MR_d0129`: MRI session for subject `OAS30001` occurring `129` days after entry.
* `OAS30001_UDSb4_d0000`: Clinical/CDR assessment for subject `OAS30001` occurring on day `0`.

To map an MRI scan to its corresponding diagnosis, you should link it with the **closest clinical/diagnostic entry in time** (usually within a $\pm365$-day window) using the matchup script:

```bash
python3 DATA/OASIS3/src/oasis-scripts-master/session_matchup/oasis_data_matchup.py \
  --list1 DATA/OASIS3/__metadata__/OASIS3_MR_json.csv \
  --list2 DATA/OASIS3/__metadata__/OASIS3_UDSb4_cdr.csv \
  --lower_bound 365 \
  --upper_bound 365 \
  --output_name DATA/OASIS3/__metadata__/matched_mr_and_clinical_data.csv
```
