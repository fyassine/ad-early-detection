# OASIS-3 Clinical Metadata and Diagnoses

This directory contains the clinical, cognitive, and demographic metadata spreadsheets for the OASIS-3 dataset downloaded from NITRC-IR. These files can be linked with MRI/PET scan sessions using the **"days from entry"** metric.

> 💡 **Tip:** A pre-built, ready-to-use CSV linking every downloaded fMRI scan to its matched clinical visit is available at [visit_level_diagnoses.csv](file:///mnt/e/fyassine/ad-early-detection/DATA/OASIS3/__metadata__/visit_level_diagnoses.csv).
> To regenerate it, run `python3 DATA/OASIS3/__metadata__/build_visit_diagnosis.py` from the project root.

---

## 1. Clinician Diagnostic and Staging Files

### A. [OASIS3_UDSb4_cdr.csv](file:///mnt/e/fyassine/ad-early-detection/DATA/OASIS3/__metadata__/OASIS3_UDSb4_cdr.csv)
Contains the Clinical Dementia Rating (CDR) scores, MMSE scores, and clinician diagnoses.
* **`CDRTOT` (Global CDR Score):** Dementia *staging* scale (not a clinician diagnosis):
  * `0.0` = Cognitively Normal / Healthy
  * `0.5` = Very Mild Dementia
  * `1.0` = Mild Dementia
  * `2.0` = Moderate Dementia
  * `3.0` = Severe Dementia
  * Note: CDR is a severity stage, **not** an MCI/AD diagnosis. The clinician diagnosis lives in `dx1` / `dx1_code`.
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

## 3. Reading and Matching Clinical Data to MRI/PET scans

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
