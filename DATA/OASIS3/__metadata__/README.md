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

---

## 4. Converter / Non-Converter rsfMRI Subject Lists

[build_oasis3_rsfmri_subject_lists.py](file:///mnt/e/fyassine/ad-early-detection/DATA/OASIS3/__metadata__/build_oasis3_rsfmri_subject_lists.py)
builds an OASIS3 analogue of ADNI's `Extended_rsfMRI_MCI_{Converters,NonConverters,Longitudinal}` CSVs
(see `DATA/DELCODE/src/processing/build_adni_rsfmri_subject_lists.py`), with the identical 14-column
schema so both cohorts can be read the same way downstream:

```
subject_id,label,viscode,examdate,diagnosis,has_rsfmri_scan,image_id,fmri_visit,
fmri_date,fmri_description,fmri_group,fmri_tr,fmri_te,date_diff_days
```

Regenerate with:

```bash
source .venv/bin/activate && python DATA/OASIS3/__metadata__/build_oasis3_rsfmri_subject_lists.py
```

### Labels

* `converter` — MCI at some visit, later diagnosed AD.
* `non_converter_stable_mci` — MCI at some visit, never diagnosed AD.

This is built from `OASIS3_UDSd1_diagnoses.csv` (Form D1), **not** from the CDR/`dx1` staging table
used in §1–3 above, because D1 exposes explicit MCI subtypes and AD etiology flags that `dx1` free text
does not cleanly separate.

### MCI definition — all 19 MCI columns, correctly interpreted

MCI = **any** of the 16 UDS Form D1 MCI-subtype columns `== 1`:
`MCIAMEM, MCIAPLUS, MCIAPLAN, MCIAPATT, MCIAPEX, MCIAPVIS, MCINON1, MCIN1LAN, MCIN1ATT, MCIN1EX,
MCIN1VIS, MCINON2, MCIN2LAN, MCIN2ATT, MCIN2EX, MCIN2VIS`.

These columns look sparse (~5.6% of rows populated) but that is **skip-logic, not missing data**: Form
D1 only asks the MCI-subtype questions when `NORMCOG == 0` and `DEMENTED == 0`. Once those gates are
accounted for, D1 yields a complete CN/MCI/AD-eligible diagnosis for ~90% of rows. `IMPNOMCI == 1`
("impaired, not MCI") is deliberately **excluded** — it's a distinct category, not a weaker form of MCI.
In practice the 4 parent flags (`MCIAMEM`, `MCIAPLUS`, `MCINON1`, `MCINON2`) account for nearly all
positives; the 12 domain sub-flags exist mainly to catch the rare row where only a sub-flag fires.

### AD definition — era-aware, do not use `PROBAD` alone

```
AD = DEMENTED == 1 AND (PROBAD == 1 OR (alzdis == 1 AND alzdisif == 1))
```

**OASIS3 spans two UDS form eras that record AD etiology in different fields**, and they are almost
perfectly complementary — of the demented rows, essentially every one answers exactly one of the two:

| Era | `dxmethod` | AD etiology field |
|---|---|---|
| UDS v1/v2 | `NaN` | `PROBAD == 1` |
| UDS v3 | populated | `alzdis == 1 AND alzdisif == 1` (`alzdisif == 1` = primary cause; `alzdisif == 2` = contributing but not primary) |

**`DEMENTED == 1 AND PROBAD == 1` alone silently drops every UDS-v3 AD diagnosis** (v3 never populates
`PROBAD`), cutting the AD-diagnosed subject count roughly in half and understating the converter cohort
by ~3×. Always OR in the `alzdis`/`alzdisif` v3 path.

### No calendar dates

OASIS3 is de-identified: there are **no calendar dates anywhere**, only "days from entry"
(`days_to_visit`, and the `d####` suffix in session labels). The `examdate` and `fmri_date` columns in
the output CSVs therefore hold **integers** (days from entry), not dates — unlike the same-named columns
in the ADNI CSVs. `date_diff_days` is the one column directly comparable across both cohorts.

### Scan matching window: ±365 days

OASIS3 clinical visits are roughly annual (unlike ADNI's denser visit schedule), so a ±90-day window
(as used for ADNI) matches almost nothing. This script uses **±365 days**, consistent with
`build_visit_diagnosis.py` and the recommendation in §3 above, and records `date_diff_days` so the
window can be tightened downstream without regenerating.

Only `scan category == "bold-rest"` scans are used, excluding `SeriesDescription == "rsfmri_ref"`
(a reference/calibration scan, not usable resting-state data). `RepetitionTime`/`EchoTime` are stored in
**seconds** in `OASIS3_MR_json.csv` (BIDS convention) and are converted ×1000 to milliseconds to match
ADNI's units before assigning `fmri_group` (A/B/C by TR, same thresholds as the ADNI script).

### Longitudinal CSV criterion

The ADNI `Extended_rsfMRI_MCI_Longitudinal` CSV has no reproducible generator in this repo — it was
built ad hoc. Rather than guess at its exact rule, the OASIS3 Longitudinal CSV uses an explicit,
documented criterion: **subjects with ≥2 scan-matched visits** (`has_rsfmri_scan == True`) across the
converter/non-converter union.

### Unanchored CSV — scans that never get a row anywhere else

`attach_fmri()` only emits a row for an rsfMRI session when it is the *closest* in-window
(≤365-day) session to some labeled visit. A downloaded session that never wins that comparison
for any visit of its subject — either every visit is more than 365 days away, or a different
session was closer for all of them — gets no row in the Converters, NonConverters, or
Longitudinal CSVs and otherwise looks like missing data rather than an intentional exclusion.

`Extended_rsfMRI_MCI_Unanchored_<date>.csv` makes those sessions visible instead of silently
dropping them: one row per orphaned session, subject/scan identifiers, plus the closest labeled
visit *regardless of distance* and the true `date_diff_days` to it.

```
subject_id,label,image_id,fmri_visit,fmri_date,fmri_description,
fmri_group,fmri_tr,fmri_te,nearest_viscode,nearest_examdate,nearest_diagnosis,date_diff_days
```

As of the 18Jul2026 run: 170 sessions across 101 converter/non-converter subjects are unanchored
(60 of those subjects have *zero* matched visits anywhere — every session they have is
unanchored). The gap to the nearest labeled visit is large, not borderline: median 1,376 days,
mean 1,635 days, min 369 days, max 4,781 days. Widening `DATE_WINDOW` would not meaningfully
recover these — even doubling it to 730 days only pulls in 30 of the 170 sessions, at the cost of
attaching a diagnosis label that's up to two years stale. These scans are legitimate rsfMRI data;
they're just too far from any clinical visit to trust a diagnosis label. They're better suited to
label-free uses (e.g. GAAE pretraining) than to the labeled classifier CSVs.
