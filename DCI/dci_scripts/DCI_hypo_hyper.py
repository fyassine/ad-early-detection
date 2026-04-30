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
from scipy.ndimage import binary_fill_holes, binary_closing
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure
from nilearn import input_data

def create_brain_mask_from_bold(subject, session, bold_base_path, output_session_path):
    """
    Create a brain mask from BOLD file by identifying non-zero voxels
    """
    
    # Construct BOLD file path
    bold_filename = f"{subject}_{session}_task-rest_space-MNI152NLin2009cAsym_res-2_desc-ICAAROMA2Phys1GS_bold_reoriented.nii.gz"
    bold_file_path = os.path.join(bold_base_path, subject, session, bold_filename)
    
    if not os.path.exists(bold_file_path):
        print(f"BOLD file not found: {bold_file_path}")
        return None
    
    try:
        # Load BOLD data
        print(f"Creating brain mask from: {bold_file_path}")
        bold_img = nib.load(bold_file_path)
        bold_data = bold_img.get_fdata()
        
        # Create brain mask
        if len(bold_data.shape) == 4:  # 4D data (x, y, z, time)
            # Calculate mean and std across time
            mean_signal = np.mean(bold_data, axis=3)
            std_signal = np.std(bold_data, axis=3)
            
            # Create mask where there's meaningful signal
            brain_mask = np.logical_or(
                np.abs(mean_signal) > 1e-6,
                std_signal > 1e-6
            ).astype(np.uint8)
            
        else:  # 3D data
            brain_mask = (np.abs(bold_data) > 1e-6).astype(np.uint8)
        
        # Clean up the mask
        from scipy.ndimage import binary_fill_holes, binary_closing
        brain_mask = binary_fill_holes(brain_mask).astype(np.uint8)
        brain_mask = binary_closing(brain_mask, iterations=2).astype(np.uint8)
        
        # Save brain mask
        mask_img = nib.Nifti1Image(brain_mask, affine=bold_img.affine)
        mask_output_path = os.path.join(output_session_path, f'{subject}_{session}_brain_mask_from_bold.nii.gz')
        
        nib.save(mask_img, mask_output_path)
        print(f"Brain mask saved: {mask_output_path} ({np.sum(brain_mask)} voxels)")
        
        return mask_output_path
        
    except Exception as e:
        print(f"Error creating brain mask: {str(e)}")
        return None

def apply_brain_mask(input_file, mask_file, output_file):
    """
    Apply brain mask to DCI map using Python/nibabel instead of fslmaths
    Check if voxel is in brain mask: keep value if yes, set to zero if no
    """
    try:
        # Load input DCI map and brain mask
        input_img = nib.load(input_file)
        mask_img = nib.load(mask_file)
        
        # Get the data arrays
        input_data = input_img.get_fdata()
        mask_data = mask_img.get_fdata()
        
        # Ensure mask is binary (0 or 1)
        mask_binary = (mask_data > 0).astype(np.uint8)
        
        # Apply mask: keep original value if in mask (mask=1), set to 0 if outside (mask=0)
        masked_data = np.where(mask_binary == 1, input_data, 0)
        
        # Create output image with same header/affine as input
        masked_img = nib.Nifti1Image(masked_data, affine=mask_img.affine, header=input_img.header)
        
        # Save the result
        nib.save(masked_img, output_file)
        
        # Print some statistics for verification
        total_voxels = np.prod(input_data.shape)
        brain_voxels = np.sum(mask_binary)
        nonzero_input = np.sum(input_data != 0)
        nonzero_masked = np.sum(masked_data != 0)
        
        print(f"Applied brain mask to {input_file}")
        print(f"  Total voxels: {total_voxels}")
        print(f"  Brain mask voxels: {brain_voxels} ({brain_voxels/total_voxels*100:.1f}%)")
        print(f"  Non-zero input voxels: {nonzero_input}")
        print(f"  Non-zero masked voxels: {nonzero_masked}")
        print(f"  Saved to: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"Error applying brain mask to {input_file}: {str(e)}")
        return False

# Suppress warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Update base paths
path_base = '/dss/dssfs03/pn72zi/pn72zi-dss-0001/di38jor/Projects/Glioma_TMS'
path_corr = os.path.join(path_base, 'outputs', 'test_DCI')
path_bold = os.path.join(path_base, 'data', 'postprocessed_v1')  # Path to BOLD files
ref_path = '/dss/dssfs03/pn72zi/pn72zi-dss-0001/di38jor/Projects/GSP/1000_subjects/DCI/outputs/corr_stats'

# Define constants
folder_lh = 'SEED_LH'
folder_rh = 'SEED_RH'
folder_lh_AI = 'SEED_LH_AI'
folder_rh_AI = 'SEED_RH_AI'
# New folders for hypo/hyperconnectivity
folder_lh_AI_hypo = 'SEED_LH_AI_HYPO'
folder_rh_AI_hypo = 'SEED_RH_AI_HYPO'
folder_lh_AI_hyper = 'SEED_LH_AI_HYPER'
folder_rh_AI_hyper = 'SEED_RH_AI_HYPER'

thr = 4.0
lthr = -4.0

# Load reference data
L_mean_total = np.load(os.path.join(ref_path, f'L_mean_total.npy'))
L_std_total = np.load(os.path.join(ref_path, f'L_std_total.npy'))
R_mean_total = np.load(os.path.join(ref_path, f'R_mean_total.npy'))
R_std_total = np.load(os.path.join(ref_path, f'R_std_total.npy'))

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
# New arrays for hypo/hyperconnectivity
LH_AI_hypo_total = []
RH_AI_hypo_total = []
LH_AI_hyper_total = []
RH_AI_hyper_total = []

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

        # Create brain mask from BOLD file
        brain_mask_path = os.path.join(session_path, f'{subject}_{session}_brain_mask_from_bold.nii.gz')
        
        # Create brain mask if it doesn't exist
        if not os.path.exists(brain_mask_path):
            brain_mask_path = create_brain_mask_from_bold(subject, session, path_bold, session_path)
            if brain_mask_path is None:
                print(f"Could not create brain mask for {subject} {session}, skipping...")
                continue

        try:
            # Load correlation matrices
            Corr_RR = np.load(os.path.join(session_path, f'{subject}_{session}_Corr_RR.npy'))
            Corr_RL = np.load(os.path.join(session_path, f'{subject}_{session}_Corr_RL.npy'))
            Corr_LL = np.load(os.path.join(session_path, f'{subject}_{session}_Corr_LL.npy'))
            Corr_LR = np.load(os.path.join(session_path, f'{subject}_{session}_Corr_LR.npy'))

            print("L_mean_total range:", np.min(L_mean_total), np.max(L_mean_total))
            print("R_mean_total range:", np.min(R_mean_total), np.max(R_mean_total))

            # LH processing with hypo/hyperconnectivity
            L_tumor = np.concatenate((Corr_LL, Corr_LR), axis=1)
            print(f'L_tumor mean: {np.mean(L_tumor)}')
            print(f'L_mean_total mean: {np.mean(np.nan_to_num(L_mean_total, copy=True, nan=0, posinf=0, neginf=0))}')
            print(f'L_std_total mean: {np.mean(np.nan_to_num(L_std_total, copy=True, nan=0, posinf=0, neginf=0))}')
            
            Abnormality_score_LH = np.divide((L_tumor - np.squeeze(L_mean_total)), np.squeeze(L_std_total))
            Abnormality_score_LH = np.nan_to_num(Abnormality_score_LH, copy=True, nan=0, posinf=0, neginf=0)
            print(f'Abnormality_score_LH mean: {np.mean(Abnormality_score_LH)}')
            
            # Original thresholding for total AI
            abn = np.abs(Abnormality_score_LH)
            abn[abn <= thr] = 0
            abn[abn > thr] = 1
            print(f'shape of the LH is {abn.shape}')
            ALT_Abnormality_Score_LH_all_z4 = np.sum(abn, axis=1)
            LH_AI = np.sum(ALT_Abnormality_Score_LH_all_z4) / 2218

            # NEW: Hypo/hyperconnectivity maps for LH
            connectivity_map_LH = np.zeros_like(Abnormality_score_LH)
            hypoconnective_map_LH = np.zeros_like(Abnormality_score_LH)
            hyperconnective_map_LH = np.zeros_like(Abnormality_score_LH)

            # Identify hypoconnective regions (negative abnormality)
            hypoconnective_map_LH[Abnormality_score_LH <= -thr] = -1
            connectivity_map_LH[Abnormality_score_LH <= -thr] = -1
            
            # Identify hyperconnective regions (positive abnormality)
            hyperconnective_map_LH[Abnormality_score_LH >= thr] = 1
            connectivity_map_LH[Abnormality_score_LH >= thr] = 1
            
            # Calculate AI values for hypo/hyperconnectivity
            ALT_Abnormality_Score_LH_hypo = np.sum(hypoconnective_map_LH, axis=1)
            ALT_Abnormality_Score_LH_hyper = np.sum(hyperconnective_map_LH, axis=1)
            
            LH_AI_hypo = np.sum(ALT_Abnormality_Score_LH_hypo) / 2218
            LH_AI_hyper = np.sum(ALT_Abnormality_Score_LH_hyper) / 2218

            # Store results
            LH_AI_total.append([subject, session, thr, subject_label, LH_AI])
            LH_AI_hypo_total.append([subject, session, thr, subject_label, LH_AI_hypo])
            LH_AI_hyper_total.append([subject, session, thr, subject_label, LH_AI_hyper])

            # RH processing with hypo/hyperconnectivity
            R_tumor = np.concatenate((Corr_RR, Corr_RL), axis=1)
            print(f'R_tumor mean: {np.mean(R_tumor)}')
            print(f'R_mean_total mean: {np.mean(np.nan_to_num(R_mean_total, copy=True, nan=0, posinf=0, neginf=0))}')
            print(f'R_std_total mean: {np.mean(np.nan_to_num(R_std_total, copy=True, nan=0, posinf=0, neginf=0))}')
            
            Abnormality_score_RH = np.divide((R_tumor - np.squeeze(R_mean_total)), np.squeeze(R_std_total))
            Abnormality_score_RH = np.nan_to_num(Abnormality_score_RH, copy=True, nan=0, posinf=0, neginf=0)
            print(f'Abnormality_score_RH mean: {np.mean(Abnormality_score_RH)}')
            
            # Original thresholding for total AI
            abn = np.abs(Abnormality_score_RH)
            abn[abn <= thr] = 0
            abn[abn > thr] = 1
            print(f'shape of the RH is {abn.shape}')
            ALT_Abnormality_Score_RH_all_z4 = np.sum(abn, axis=1)
            RH_AI = np.sum(ALT_Abnormality_Score_RH_all_z4) / 2038

            # NEW: Hypo/hyperconnectivity maps for RH
            connectivity_map_RH = np.zeros_like(Abnormality_score_RH)
            hypoconnective_map_RH = np.zeros_like(Abnormality_score_RH)
            hyperconnective_map_RH = np.zeros_like(Abnormality_score_RH)

            # Identify hypoconnective regions (negative abnormality)
            hypoconnective_map_RH[Abnormality_score_RH <= -thr] = -1
            connectivity_map_RH[Abnormality_score_RH <= -thr] = -1
            
            # Identify hyperconnective regions (positive abnormality)
            hyperconnective_map_RH[Abnormality_score_RH >= thr] = 1
            connectivity_map_RH[Abnormality_score_RH >= thr] = 1
            
            # Calculate AI values for hypo/hyperconnectivity
            ALT_Abnormality_Score_RH_hypo = np.sum(hypoconnective_map_RH, axis=1)
            ALT_Abnormality_Score_RH_hyper = np.sum(hyperconnective_map_RH, axis=1)
            
            RH_AI_hypo = np.sum(ALT_Abnormality_Score_RH_hypo) / 2038
            RH_AI_hyper = np.sum(ALT_Abnormality_Score_RH_hyper) / 2038

            # Store results
            RH_AI_total.append([subject, session, thr, subject_label, RH_AI])
            RH_AI_hypo_total.append([subject, session, thr, subject_label, RH_AI_hypo])
            RH_AI_hyper_total.append([subject, session, thr, subject_label, RH_AI_hyper])

            # Print enhanced statistics
            print(f"Number of values above threshold in LH: {np.sum(np.abs(Abnormality_score_LH) > thr)}")
            print(f"Number of values above threshold in RH: {np.sum(np.abs(Abnormality_score_RH) > thr)}")
            print(f"LH Hypoconnective voxels: {np.sum(hypoconnective_map_LH == -1)}")
            print(f"LH Hyperconnective voxels: {np.sum(hyperconnective_map_LH == 1)}")
            print(f"RH Hypoconnective voxels: {np.sum(hypoconnective_map_RH == -1)}")
            print(f"RH Hyperconnective voxels: {np.sum(hyperconnective_map_RH == 1)}")
            print(f"LH AI Total: {LH_AI:.6f}, Hypo: {LH_AI_hypo:.6f}, Hyper: {LH_AI_hyper:.6f}")
            print(f"RH AI Total: {RH_AI:.6f}, Hypo: {RH_AI_hypo:.6f}, Hyper: {RH_AI_hyper:.6f}")

            # Save individual AI results (enhanced with hypo/hyper)
            pd.DataFrame([LH_AI_total[-1]], columns=['Subject', 'Session', 'Threshold', 'Subject_Label', 'LH_AI']).to_csv(
                os.path.join(session_path, f'{subject}_{session}_corrLH_th_{thr}_Delcode.csv'),
                index=False
            )
            pd.DataFrame([RH_AI_total[-1]], columns=['Subject', 'Session', 'Threshold', 'Subject_Label', 'RH_AI']).to_csv(
                os.path.join(session_path, f'{subject}_{session}_corrRH_th_{thr}_Delcode.csv'),
                index=False
            )
            
            # NEW: Save hypo/hyperconnectivity results
            pd.DataFrame([LH_AI_hypo_total[-1]], columns=['Subject', 'Session', 'Threshold', 'Subject_Label', 'LH_AI_Hypo']).to_csv(
                os.path.join(session_path, f'{subject}_{session}_corrLH_hypo_th_{thr}_Delcode.csv'),
                index=False
            )
            pd.DataFrame([LH_AI_hyper_total[-1]], columns=['Subject', 'Session', 'Threshold', 'Subject_Label', 'LH_AI_Hyper']).to_csv(
                os.path.join(session_path, f'{subject}_{session}_corrLH_hyper_th_{thr}_Delcode.csv'),
                index=False
            )
            pd.DataFrame([RH_AI_hypo_total[-1]], columns=['Subject', 'Session', 'Threshold', 'Subject_Label', 'RH_AI_Hypo']).to_csv(
                os.path.join(session_path, f'{subject}_{session}_corrRH_hypo_th_{thr}_Delcode.csv'),
                index=False
            )
            pd.DataFrame([RH_AI_hyper_total[-1]], columns=['Subject', 'Session', 'Threshold', 'Subject_Label', 'RH_AI_Hyper']).to_csv(
                os.path.join(session_path, f'{subject}_{session}_corrRH_hyper_th_{thr}_Delcode.csv'),
                index=False
            )

            # Create and prepare AI folders (original + new hypo/hyper folders)
            path_lh_ai = os.path.join(session_path, f'{folder_lh_AI}_thr_{thr}')
            path_rh_ai = os.path.join(session_path, f'{folder_rh_AI}_thr_{thr}')
            path_lh_ai_hypo = os.path.join(session_path, f'{folder_lh_AI_hypo}_thr_{thr}')
            path_rh_ai_hypo = os.path.join(session_path, f'{folder_rh_AI_hypo}_thr_{thr}')
            path_lh_ai_hyper = os.path.join(session_path, f'{folder_lh_AI_hyper}_thr_{thr}')
            path_rh_ai_hyper = os.path.join(session_path, f'{folder_rh_AI_hyper}_thr_{thr}')

            for path in [path_lh_ai, path_rh_ai, path_lh_ai_hypo, path_rh_ai_hypo, path_lh_ai_hyper, path_rh_ai_hyper]:
                if os.path.exists(path):
                    shutil.rmtree(path)
                os.mkdir(path)

            # Process LH seeds (original + hypo/hyper)
            AI_counter_LH = 0
            for filename in sorted(file for file in os.listdir(os.path.join(session_path, f'{subject}_{folder_lh}')) if not file.startswith('.')):
                LH_Seed = nib.load(os.path.join(session_path, f'{subject}_{folder_lh}', filename))
                LH_Seed_vol = LH_Seed.get_fdata()

                indices = np.where(LH_Seed_vol == 1)
                seed_idx = left_seeds.index(filename)
                
                # Original AI
                LH_Seed_vol_orig = LH_Seed_vol.copy()
                LH_Seed_vol_orig[indices] = ALT_Abnormality_Score_LH_all_z4[seed_idx]
                hdr_final_LH = nib.Nifti1Image(LH_Seed_vol_orig, affine=LH_Seed.affine)
                nib.save(hdr_final_LH, os.path.join(path_lh_ai, filename))

                # Hypoconnectivity AI
                LH_Seed_vol_hypo = LH_Seed_vol.copy()
                LH_Seed_vol_hypo[indices] = ALT_Abnormality_Score_LH_hypo[seed_idx]
                hdr_final_LH_hypo = nib.Nifti1Image(LH_Seed_vol_hypo, affine=LH_Seed.affine)
                nib.save(hdr_final_LH_hypo, os.path.join(path_lh_ai_hypo, filename))

                # Hyperconnectivity AI
                LH_Seed_vol_hyper = LH_Seed_vol.copy()
                LH_Seed_vol_hyper[indices] = ALT_Abnormality_Score_LH_hyper[seed_idx]
                hdr_final_LH_hyper = nib.Nifti1Image(LH_Seed_vol_hyper, affine=LH_Seed.affine)
                nib.save(hdr_final_LH_hyper, os.path.join(path_lh_ai_hyper, filename))

                AI_counter_LH += 1

            # Sum LH seeds (original + hypo/hyper)
            LH_Seed_AI_sum_vol_ges = np.zeros_like(LH_Seed_vol)
            LH_Seed_AI_hypo_sum_vol_ges = np.zeros_like(LH_Seed_vol)
            LH_Seed_AI_hyper_sum_vol_ges = np.zeros_like(LH_Seed_vol)
            
            for filename in sorted(os.listdir(path_lh_ai)):
                if not filename.startswith('.'):
                    # Original
                    LH_Seed_AI_sum = nib.load(os.path.join(path_lh_ai, filename))
                    LH_Seed_AI_sum_vol = LH_Seed_AI_sum.get_fdata()
                    LH_Seed_AI_sum_vol_ges += LH_Seed_AI_sum_vol
                    
                    # Hypo
                    LH_Seed_AI_hypo_sum = nib.load(os.path.join(path_lh_ai_hypo, filename))
                    LH_Seed_AI_hypo_sum_vol = LH_Seed_AI_hypo_sum.get_fdata()
                    LH_Seed_AI_hypo_sum_vol_ges += LH_Seed_AI_hypo_sum_vol
                    
                    # Hyper
                    LH_Seed_AI_hyper_sum = nib.load(os.path.join(path_lh_ai_hyper, filename))
                    LH_Seed_AI_hyper_sum_vol = LH_Seed_AI_hyper_sum.get_fdata()
                    LH_Seed_AI_hyper_sum_vol_ges += LH_Seed_AI_hyper_sum_vol

            # Save summed volumes
            hdr_final_LH_ges = nib.Nifti1Image(LH_Seed_AI_sum_vol_ges, affine=LH_Seed_AI_sum.affine)
            nib.save(hdr_final_LH_ges, os.path.join(path_lh_ai, f'LH_Seed_AI_sum_vol_ges_orig_{thr}.nii.gz'))
            
            hdr_final_LH_hypo_ges = nib.Nifti1Image(LH_Seed_AI_hypo_sum_vol_ges, affine=LH_Seed_AI_hypo_sum.affine)
            nib.save(hdr_final_LH_hypo_ges, os.path.join(path_lh_ai_hypo, f'LH_Seed_AI_hypo_sum_vol_ges_orig_{thr}.nii.gz'))
            
            hdr_final_LH_hyper_ges = nib.Nifti1Image(LH_Seed_AI_hyper_sum_vol_ges, affine=LH_Seed_AI_hyper_sum.affine)
            nib.save(hdr_final_LH_hyper_ges, os.path.join(path_lh_ai_hyper, f'LH_Seed_AI_hyper_sum_vol_ges_orig_{thr}.nii.gz'))

            # Process RH seeds (original + hypo/hyper)
            AI_counter_RH = 0
            for filename in sorted(file for file in os.listdir(os.path.join(session_path, f'{subject}_{folder_rh}')) if not file.startswith('.')):
                RH_Seed = nib.load(os.path.join(session_path, f'{subject}_{folder_rh}', filename))
                RH_Seed_vol = RH_Seed.get_fdata()

                indices = np.where(RH_Seed_vol == 1)
                seed_idx = right_seeds.index(filename)
                
                # Original AI
                RH_Seed_vol_orig = RH_Seed_vol.copy()
                RH_Seed_vol_orig[indices] = ALT_Abnormality_Score_RH_all_z4[seed_idx]
                hdr_final_RH = nib.Nifti1Image(RH_Seed_vol_orig, affine=RH_Seed.affine)
                nib.save(hdr_final_RH, os.path.join(path_rh_ai, filename))

                # Hypoconnectivity AI
                RH_Seed_vol_hypo = RH_Seed_vol.copy()
                RH_Seed_vol_hypo[indices] = ALT_Abnormality_Score_RH_hypo[seed_idx]
                hdr_final_RH_hypo = nib.Nifti1Image(RH_Seed_vol_hypo, affine=RH_Seed.affine)
                nib.save(hdr_final_RH_hypo, os.path.join(path_rh_ai_hypo, filename))

                # Hyperconnectivity AI
                RH_Seed_vol_hyper = RH_Seed_vol.copy()
                RH_Seed_vol_hyper[indices] = ALT_Abnormality_Score_RH_hyper[seed_idx]
                hdr_final_RH_hyper = nib.Nifti1Image(RH_Seed_vol_hyper, affine=RH_Seed.affine)
                nib.save(hdr_final_RH_hyper, os.path.join(path_rh_ai_hyper, filename))

                AI_counter_RH += 1

            # Sum RH seeds (original + hypo/hyper)
            RH_Seed_AI_sum_vol_ges = np.zeros_like(RH_Seed_vol)
            RH_Seed_AI_hypo_sum_vol_ges = np.zeros_like(RH_Seed_vol)
            RH_Seed_AI_hyper_sum_vol_ges = np.zeros_like(RH_Seed_vol)
            
            for filename in sorted(os.listdir(path_rh_ai)):
                if not filename.startswith('.'):
                    # Original
                    RH_Seed_AI_sum = nib.load(os.path.join(path_rh_ai, filename))
                    RH_Seed_AI_sum_vol = RH_Seed_AI_sum.get_fdata()
                    RH_Seed_AI_sum_vol_ges += RH_Seed_AI_sum_vol
                    
                    # Hypo
                    RH_Seed_AI_hypo_sum = nib.load(os.path.join(path_rh_ai_hypo, filename))
                    RH_Seed_AI_hypo_sum_vol = RH_Seed_AI_hypo_sum.get_fdata()
                    RH_Seed_AI_hypo_sum_vol_ges += RH_Seed_AI_hypo_sum_vol
                    
                    # Hyper
                    RH_Seed_AI_hyper_sum = nib.load(os.path.join(path_rh_ai_hyper, filename))
                    RH_Seed_AI_hyper_sum_vol = RH_Seed_AI_hyper_sum.get_fdata()
                    RH_Seed_AI_hyper_sum_vol_ges += RH_Seed_AI_hyper_sum_vol

            # Save summed volumes
            hdr_final_RH_ges = nib.Nifti1Image(RH_Seed_AI_sum_vol_ges, affine=RH_Seed_AI_sum.affine)
            nib.save(hdr_final_RH_ges, os.path.join(path_rh_ai, f'RH_Seed_AI_sum_vol_ges_orig_{thr}.nii.gz'))
            
            hdr_final_RH_hypo_ges = nib.Nifti1Image(RH_Seed_AI_hypo_sum_vol_ges, affine=RH_Seed_AI_hypo_sum.affine)
            nib.save(hdr_final_RH_hypo_ges, os.path.join(path_rh_ai_hypo, f'RH_Seed_AI_hypo_sum_vol_ges_orig_{thr}.nii.gz'))
            
            hdr_final_RH_hyper_ges = nib.Nifti1Image(RH_Seed_AI_hyper_sum_vol_ges, affine=RH_Seed_AI_hyper_sum.affine)
            nib.save(hdr_final_RH_hyper_ges, os.path.join(path_rh_ai_hyper, f'RH_Seed_AI_hyper_sum_vol_ges_orig_{thr}.nii.gz'))

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

# NEW: Save combined hypo/hyperconnectivity results
pd.DataFrame(LH_AI_hypo_total, columns=['Subject', 'Session', 'Threshold', 'Subject_Label', 'LH_AI_Hypo']).to_csv(
    os.path.join(path_corr, f'LH_DCI_hypo_all_subjects_th_{thr}_Delcode.csv'),
    index=False
)

pd.DataFrame(LH_AI_hyper_total, columns=['Subject', 'Session', 'Threshold', 'Subject_Label', 'LH_AI_Hyper']).to_csv(
    os.path.join(path_corr, f'LH_DCI_hyper_all_subjects_th_{thr}_Delcode.csv'),
    index=False
)

pd.DataFrame(RH_AI_hypo_total, columns=['Subject', 'Session', 'Threshold', 'Subject_Label', 'RH_AI_Hypo']).to_csv(
    os.path.join(path_corr, f'RH_DCI_hypo_all_subjects_th_{thr}_Delcode.csv'),
    index=False
)

pd.DataFrame(RH_AI_hyper_total, columns=['Subject', 'Session', 'Threshold', 'Subject_Label', 'RH_AI_Hyper']).to_csv(
    os.path.join(path_corr, f'RH_DCI_hyper_all_subjects_th_{thr}_Delcode.csv'),
    index=False
)

# Enhanced smoothing process with hypo/hyperconnectivity maps
print("Starting enhanced smoothing process with hypo/hyperconnectivity maps...")
for subject in sorted(os.listdir(path_corr)):
    if not subject.startswith('sub-'):
        continue

    subject_path = os.path.join(path_corr, subject)

    for session in sorted(os.listdir(subject_path)):
        if not session.startswith('ses-'):
            continue

        session_path = os.path.join(subject_path, session)
        brain_mask_path = os.path.join(session_path, f'{subject}_{session}_brain_mask_from_bold.nii.gz')
        
        # Skip if no brain mask
        if not os.path.exists(brain_mask_path):
            print(f"No brain mask found for {subject} {session}, skipping smoothing...")
            continue

        # Check if all AI paths exist
        lh_ai_path = os.path.join(session_path, f'SEED_LH_AI_thr_{thr}')
        rh_ai_path = os.path.join(session_path, f'SEED_RH_AI_thr_{thr}')
        lh_ai_hypo_path = os.path.join(session_path, f'SEED_LH_AI_HYPO_thr_{thr}')
        rh_ai_hypo_path = os.path.join(session_path, f'SEED_RH_AI_HYPO_thr_{thr}')
        lh_ai_hyper_path = os.path.join(session_path, f'SEED_LH_AI_HYPER_thr_{thr}')
        rh_ai_hyper_path = os.path.join(session_path, f'SEED_RH_AI_HYPER_thr_{thr}')
        
        all_paths = [lh_ai_path, rh_ai_path, lh_ai_hypo_path, rh_ai_hypo_path, lh_ai_hyper_path, rh_ai_hyper_path]
        
        if not all(os.path.exists(path) for path in all_paths):
            print(f"Missing AI paths for {subject} {session}, skipping...")
            continue

        # Define input files for each type
        input_files = {
            'combined': [
                os.path.join(lh_ai_path, f'LH_Seed_AI_sum_vol_ges_orig_{thr}.nii.gz'),
                os.path.join(rh_ai_path, f'RH_Seed_AI_sum_vol_ges_orig_{thr}.nii.gz')
            ],
            'hypo': [
                os.path.join(lh_ai_hypo_path, f'LH_Seed_AI_hypo_sum_vol_ges_orig_{thr}.nii.gz'),
                os.path.join(rh_ai_hypo_path, f'RH_Seed_AI_hypo_sum_vol_ges_orig_{thr}.nii.gz')
            ],
            'hyper': [
                os.path.join(lh_ai_hyper_path, f'LH_Seed_AI_hyper_sum_vol_ges_orig_{thr}.nii.gz'),
                os.path.join(rh_ai_hyper_path, f'RH_Seed_AI_hyper_sum_vol_ges_orig_{thr}.nii.gz')
            ]
        }
        
        # Check if all input files exist
        all_files_exist = True
        for file_type, files in input_files.items():
            for file_path in files:
                if not os.path.exists(file_path):
                    print(f"Missing {file_type} input file: {file_path}")
                    all_files_exist = False
        
        if not all_files_exist:
            print(f"Missing input files for {subject} {session}, skipping...")
            continue

        try:
            print(f"Processing enhanced maps for {subject} {session}")
            
            # Process each type of connectivity map
            for map_type, files in input_files.items():
                lh_file, rh_file = files
                
                # Load LH and RH maps
                lh_img = nib.load(lh_file)
                rh_img = nib.load(rh_file)
                
                lh_data = lh_img.get_fdata()
                rh_data = rh_img.get_fdata()
                
                # Combine LH and RH maps
                combined_data = lh_data + rh_data
                
                # Create combined image
                combined_img = nib.Nifti1Image(combined_data, affine=lh_img.affine)
                
                # Create output directory for this map type
                output_dir = os.path.join(session_path, f'SEED_COMBINED_AI_{map_type.upper()}_thr_{thr}')
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                # Save combined map
                combined_file = os.path.join(output_dir, f'Combined_Seed_AI_{map_type}_sum_vol_ges_orig_{thr}.nii.gz')
                nib.save(combined_img, combined_file)
                print(f"Saved combined {map_type} map: {combined_file}")
                
                # Apply brain mask before smoothing
                masked_unsmoothed_file = os.path.join(output_dir, f'Combined_Seed_AI_{map_type}_sum_vol_ges_orig{thr}_masked_unsmoothed_{subject}_{session}.nii.gz')
                success = apply_brain_mask(combined_file, brain_mask_path, masked_unsmoothed_file)
                
                if not success:
                    print(f"Failed to apply brain mask for {map_type} map {subject} {session}")
                    continue
                    
                print(f"Applied brain mask to {map_type} map before smoothing: {masked_unsmoothed_file}")
                
                # Smooth the masked map
                smoothed_masked_file = os.path.join(output_dir, f'Combined_Seed_AI_{map_type}_sum_vol_ges_orig{thr}_masked_smoothed_{subject}_{session}.nii.gz')
                subprocess.run(['fslmaths', masked_unsmoothed_file, '-kernel', 'gauss', '12', '-fmean', smoothed_masked_file], check=True)
                print(f"Smoothed {map_type} masked map: {smoothed_masked_file}")
                
                # Apply mask again to clean up edge artifacts
                final_masked_file = os.path.join(output_dir, f'Combined_Seed_AI_{map_type}_sum_vol_ges_orig{thr}_final_masked_{subject}_{session}.nii.gz')
                apply_brain_mask(smoothed_masked_file, brain_mask_path, final_masked_file)
                print(f"Final {map_type} masked map: {final_masked_file}")
            
        except Exception as e:
            print(f"Error processing enhanced maps for {subject} {session}: {str(e)}")

print("Enhanced processing complete with hypo/hyperconnectivity analysis.")