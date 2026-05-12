#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import nibabel as nib
import warnings
import sys
import pandas as pd
import shutil
import subprocess
from scipy import stats
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure
from nilearn import input_data


# Suppress warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Update base paths
path_base = '/dss/dssfs03/pn72zi/pn72zi-dss-0001/di38jor/Projects/Glioma_TMS'
path_corr = os.path.join(path_base, 'outputs', 'old_DCI')
path_bold = os.path.join(path_base, 'data', 'postprocessed_v1')  # Path to BOLD files
ref_path = '/dss/dssfs03/pn72zi/pn72zi-dss-0001/di38jor/Projects/GSP/1000_subjects/DCI/outputs/corr_stats'

# Define constants
folder_lh = 'SEED_LH'
folder_rh = 'SEED_RH'
folder_lh_AI = 'SEED_LH_AI'
folder_rh_AI = 'SEED_RH_AI'
thr = 4.0
lthr = -4.0

# Load reference data
L_mean_total = np.load(os.path.join(ref_path, f'L_mean_total.npy'))
L_std_total = np.load(os.path.join(ref_path, f'L_std_total.npy'))
R_mean_total = np.load(os.path.join(ref_path, f'R_mean_total.npy'))
R_std_total = np.load(os.path.join(ref_path, f'R_std_total.npy'))

print(f'L_mean_total shape: {L_mean_total.shape}')
print(f'L_mean_total squeezed shape: {np.squeeze(L_mean_total).shape}')
print(f'L_std_total shape: {L_std_total.shape}')
print(f'R_mean_total shape: {R_mean_total.shape}')
print(f'R_std_total shape: {R_std_total.shape}')

# Replace both inf and extremely large values with 0
threshold = 1e100  # Set a reasonable threshold
L_mean_total[np.abs(L_mean_total) > threshold] = 0
R_mean_total[np.abs(R_mean_total) > threshold] = 0
L_std_total[np.abs(L_std_total) > threshold] = 0
R_std_total[np.abs(R_std_total) > threshold] = 0

# Replace remaining inf/nan
L_mean_total[~np.isfinite(L_mean_total)] = 0
R_mean_total[~np.isfinite(R_mean_total)] = 0
L_std_total[~np.isfinite(L_std_total)] = 0
R_std_total[~np.isfinite(R_std_total)] = 0

# Initialize arrays to store AI values
LH_AI_total = []
RH_AI_total = []

bold_img = nib.load('/dss/dssfs03/pn72zi/pn72zi-dss-0001/di38jor/Projects/Glioma_TMS/data/postprocessed_v1/sub-001S/ses-01/sub-001S_ses-01_task-rest_space-MNI152NLin2009cAsym_res-2_desc-ICAAROMA2Phys1GS_bold_reoriented.nii.gz')


# Main processing loop
for subject in sorted(os.listdir(path_corr)):

    if not subject.startswith('sub-'):
        continue

    with open(f'/dss/dssfs03/pn72zi/pn72zi-dss-0001/di38jor/Projects/Glioma_TMS/codes/seeds_order/{subject}_right_seeds.txt', 'r') as file:
        right_seeds = [line.strip() for line in file]
    with open(f'/dss/dssfs03/pn72zi/pn72zi-dss-0001/di38jor/Projects/Glioma_TMS/codes/seeds_order/{subject}_left_seeds.txt', 'r') as file:
        left_seeds = [line.strip() for line in file]

    # Extract subject label from the subject name
    subject_label = str(subject.split('-')[1])  # Ensure it's a string

    subject_path = os.path.join(path_corr, subject)

    for session in sorted(os.listdir(subject_path)):
        if not session.startswith('ses-'):
            continue

        session_path = os.path.join(subject_path, session)
        print(f'Processing {subject} {session}')

        try:
            # Load correlation matrices
            Corr_RR = np.load(os.path.join(session_path, f'{subject}_{session}_Corr_RR.npy'))
            Corr_RL = np.load(os.path.join(session_path, f'{subject}_{session}_Corr_RL.npy'))
            Corr_LL = np.load(os.path.join(session_path, f'{subject}_{session}_Corr_LL.npy'))
            Corr_LR = np.load(os.path.join(session_path, f'{subject}_{session}_Corr_LR.npy'))

            # LH processing
            L_tumor = np.concatenate((Corr_LL, Corr_LR), axis=1)
            print(f'L_tumor mean: {np.mean(L_tumor)}')
            print(f'L_mean_total mean: {np.mean(np.nan_to_num(L_mean_total, copy=True, nan=0, posinf=0, neginf=0))}')
            print(f'L_std_total mean: {np.mean(np.nan_to_num(L_std_total, copy=True, nan=0, posinf=0, neginf=0))}')
            Abnormality_score_LH = np.divide((L_tumor - np.squeeze(L_mean_total)), np.squeeze(L_std_total))
            Abnormality_score_LH = np.nan_to_num(Abnormality_score_LH, copy=True, nan=0, posinf=0, neginf=0)
            print(f'Abnormality_score_LH mean: {np.mean(Abnormality_score_LH)}')
            abn = np.abs(Abnormality_score_LH)
            abn[abn <= thr] = 0
            abn[abn > thr] = 1
            print(f'shape of the LH is {abn.shape}')
            ALT_Abnormality_Score_LH_all_z4 = np.sum(abn, axis=1)
            print(f'ALT_Abnormality_Score_LH_all_z4 shape: {ALT_Abnormality_Score_LH_all_z4.shape}')
            LH_AI = np.sum(ALT_Abnormality_Score_LH_all_z4) / 2218

            LH_AI_total.append([subject, session, thr, subject_label, LH_AI])

            # RH processing
            R_tumor = np.concatenate((Corr_RR, Corr_RL), axis=1)
            print(f'R_tumor mean: {np.mean(R_tumor)}')
            print(f'R_mean_total mean: {np.mean(np.nan_to_num(R_mean_total, copy=True, nan=0, posinf=0, neginf=0))}')
            print(f'R_std_total mean: {np.mean(np.nan_to_num(R_std_total, copy=True, nan=0, posinf=0, neginf=0))}')
            Abnormality_score_RH = np.divide((R_tumor - np.squeeze(R_mean_total)), np.squeeze(R_std_total))
            Abnormality_score_RH = np.nan_to_num(Abnormality_score_RH, copy=True, nan=0, posinf=0, neginf=0)
            print(f'Abnormality_score_RH mean: {np.mean(Abnormality_score_RH)}')
            abn = np.abs(Abnormality_score_RH)
            abn[abn <= thr] = 0
            abn[abn > thr] = 1
            print(f'shape of the RH is {abn.shape}')
            ALT_Abnormality_Score_RH_all_z4 = np.sum(abn, axis=1)
            RH_AI = np.sum(ALT_Abnormality_Score_RH_all_z4) / 2038

            RH_AI_total.append([subject, session, thr, subject_label, RH_AI])

            # Add after calculating abnormality scores:
            print(f"Number of values above threshold in LH: {np.sum(np.abs(Abnormality_score_LH) > thr)}")
            print(f"Number of values above threshold in RH: {np.sum(np.abs(Abnormality_score_RH) > thr)}")
            print(f"Percentage above threshold LH: {np.sum(np.abs(Abnormality_score_LH) > thr) / (2218 * 4256) * 100:.2f}%")
            print(f"Percentage above threshold RH: {np.sum(np.abs(Abnormality_score_RH) > thr) / (2038 * 4256) * 100:.2f}%")

            # Distribution statistics
            print(f"LH abnormality score percentiles: {np.percentile(np.abs(Abnormality_score_LH), [25, 50, 75, 90, 95, 99])}")
            print(f"RH abnormality score percentiles: {np.percentile(np.abs(Abnormality_score_RH), [25, 50, 75, 90, 95, 99])}")

            # Save individual AI results
            pd.DataFrame([LH_AI_total[-1]], columns=['Subject', 'Session', 'Threshold', 'Subject_Label', 'LH_AI']).to_csv(
                os.path.join(session_path, f'{subject}_{session}_corrLH_th_{thr}_Delcode.csv'),
                index=False
            )
            pd.DataFrame([RH_AI_total[-1]], columns=['Subject', 'Session', 'Threshold', 'Subject_Label', 'RH_AI']).to_csv(
                os.path.join(session_path, f'{subject}_{session}_corrRH_th_{thr}_Delcode.csv'),
                index=False
            )

            # Create and prepare AI folders
            path_lh_ai = os.path.join(session_path, f'{folder_lh_AI}_thr_{thr}')
            path_rh_ai = os.path.join(session_path, f'{folder_rh_AI}_thr_{thr}')

            for path in [path_lh_ai, path_rh_ai]:
                if os.path.exists(path):
                    shutil.rmtree(path)
                os.mkdir(path)

            # Process LH seeds
            AI_counter_LH = 0
            for filename in sorted(file for file in os.listdir(os.path.join(session_path, f'{subject}_{folder_lh}')) if not file.startswith('.')):
                LH_Seed = nib.load(os.path.join(session_path, f'{subject}_{folder_lh}', filename))
                LH_Seed_vol = LH_Seed.get_fdata()

                indices = np.where(LH_Seed_vol == 1)
                LH_Seed_vol[indices] = ALT_Abnormality_Score_LH_all_z4[left_seeds.index(filename)]

                hdr_final_LH = nib.Nifti1Image(LH_Seed_vol, affine=bold_img.affine)
                nib.save(hdr_final_LH, os.path.join(path_lh_ai, filename))

                AI_counter_LH += 1

            # Sum LH seeds
            LH_Seed_AI_sum_vol_ges = np.zeros_like(LH_Seed_vol)
            for filename in sorted(os.listdir(path_lh_ai)):
                if not filename.startswith('.'):
                    LH_Seed_AI_sum = nib.load(os.path.join(path_lh_ai, filename))
                    LH_Seed_AI_sum_vol = LH_Seed_AI_sum.get_fdata()
                    LH_Seed_AI_sum_vol_ges += LH_Seed_AI_sum_vol

            hdr_final_LH_ges = nib.Nifti1Image(LH_Seed_AI_sum_vol_ges, affine=bold_img.affine)
            print('******************************************')
            print(LH_Seed_AI_sum)
            print(LH_Seed_AI_sum.affine)
            nib.save(hdr_final_LH_ges, os.path.join(path_lh_ai, f'LH_Seed_AI_sum_vol_ges_orig_{thr}.nii.gz'))

            # Process RH seeds
            AI_counter_RH = 0
            for filename in sorted(file for file in os.listdir(os.path.join(session_path, f'{subject}_{folder_rh}')) if not file.startswith('.')):
                RH_Seed = nib.load(os.path.join(session_path, f'{subject}_{folder_rh}', filename))
                RH_Seed_vol = RH_Seed.get_fdata()

                indices = np.where(RH_Seed_vol == 1)
                RH_Seed_vol[indices] = ALT_Abnormality_Score_RH_all_z4[right_seeds.index(filename)]

                hdr_final_RH = nib.Nifti1Image(RH_Seed_vol, affine=bold_img.affine)
                nib.save(hdr_final_RH, os.path.join(path_rh_ai, filename))

                AI_counter_RH += 1

            # Sum RH seeds
            RH_Seed_AI_sum_vol_ges = np.zeros_like(RH_Seed_vol)
            for filename in sorted(os.listdir(path_rh_ai)):
                if not filename.startswith('.'):
                    RH_Seed_AI_sum = nib.load(os.path.join(path_rh_ai, filename))
                    RH_Seed_AI_sum_vol = RH_Seed_AI_sum.get_fdata()
                    RH_Seed_AI_sum_vol_ges += RH_Seed_AI_sum_vol

            hdr_final_RH_ges = nib.Nifti1Image(RH_Seed_AI_sum_vol_ges, affine=bold_img.affine)
            nib.save(hdr_final_RH_ges, os.path.join(path_rh_ai, f'RH_Seed_AI_sum_vol_ges_orig_{thr}.nii.gz'))

        except Exception as e:
            print(f"Error processing {subject} {session}: {str(e)}")

# Save combined DCI values for all subjects and sessions
pd.DataFrame(LH_AI_total, columns=['Subject', 'Session', 'Threshold', 'Subject_Label', 'LH_AI']).to_csv(
    os.path.join(path_corr, f'LH_DCI_all_subjects_th_{thr}_Delcode.csv'),
    index=False
)

pd.DataFrame(RH_AI_total, columns=['Subject', 'Session', 'Threshold', 'Subject_Label', 'RH_AI']).to_csv(
    os.path.join(path_corr, f'RH_DCI_all_subjects_th_{thr}_Delcode.csv'),
    index=False
)

# Smoothing process
print("Starting smoothing process...")
for subject in sorted(os.listdir(path_corr)):
    if not subject.startswith('sub-'):
        continue

    subject_path = os.path.join(path_corr, subject)

    for session in sorted(os.listdir(subject_path)):
        if not session.startswith('ses-'):
            continue

        session_path = os.path.join(subject_path, session)

        for hemisphere in ['LH', 'RH']:
            ai_path = os.path.join(session_path, f'SEED_{hemisphere}_AI_thr_{thr}')
            if not os.path.exists(ai_path):
                continue

            input_file = os.path.join(ai_path, f'{hemisphere}_Seed_AI_sum_vol_ges_orig_{thr}.nii.gz')
            output_file = os.path.join(ai_path, f'{hemisphere}_Seed_AI_sum_vol_ges_orig{thr}_smoothed_6_{subject}_{session}.nii.gz')

            try:
                subprocess.run(['fslmaths', input_file, '-kernel', 'gauss', '3', '-fmean', output_file, '-odt', 'float'], check=True)
                print(f"Smoothed {input_file}")
            except subprocess.CalledProcessError as e:
                print(f"Error smoothing {input_file}: {str(e)}")

print("Processing complete.")
