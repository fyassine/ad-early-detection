# Mapping: Table 1 variables → DELCODE filtered sheets

This file maps variables shown in the Table from https://link.springer.com/article/10.1186/s13195-017-0314-2 DZNE DELCODE Paper of cutoff values of biomarkers to candidate columns present in the three Excel sheets used by the conversion pipeline.

Sheets inspected:
- `BaselineMCI_filtered` (sheet in `MCI_Baseline.xlsm`)
- `MCIfollowupMCIdate_filtered` (sheet in `MCI_followup.xlsm`)
- `allConverters_MCIvisits` (sheet in `allConverters.xlsm`)

| Table 1 variable | Baseline sheet candidates | Followup sheet candidates | allConverters_MCIvisits candidates | Notes |
|---|---:|---:|---:|---|
| MMSE, mean (SD) | `mmstot` | `mmse` | `mmstot` | `mmstot` appears to be MMSE total in baseline/converters; followup uses `mmse`. |
| CDR, mean (SD) | `cdrglobal` | `cdrglobal` | `cdrglobal` | Global CDR available in all three. |
| CDR-SOB, mean (SD) | `cdrtot` | `cdrtot` | `cdrtot` | CDR sum-of-boxes present. |
| Aβ42 (pg/ml) | `Abeta42` | `Aß42 (pg/ml)` | `Abeta42` | Same biomarker, naming differs. |
| Aβ42/Aβ40, mean (SD) | `ratio_Abeta42_40` | `Aß42/40` | `ratio_Abeta42_40` | Direct ratio column exists. |
| tTau (pg/ml), mean (SD) | `totaltau` | `t-tau (pg/ml)` | `totaltau` | Same biomarker, different naming. |
| pTau181 (pg/ml), mean (SD) | `phosphotau181` | `p-tau-181 (pg/ml)` | `phosphotau181` | Same biomarker, different naming. |
| Aβ42/Tau ratio (Hulstaert), mean (SD) | `ratio_Abeta42_phosphotau181` | `Aß42/p-tau-181` | `ratio_Abeta42_phosphotau181` | Present as Aβ42/pTau181 in sheets; Hulstaert-specific formula not explicitly present. |
