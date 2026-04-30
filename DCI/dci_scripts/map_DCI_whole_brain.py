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

def zscore_dci_map(dci_data, brain_mask_data):
    """
    Z-score normalize DCI values within brain mask
    """
    # Get brain voxels only (where mask > 0)
    brain_voxels = dci_data[brain_mask_data > 0]
    
    # Remove zeros from brain voxels for z-score calculation
    non_zero_brain_voxels = brain_voxels[brain_voxels != 0]
    
    if len(non_zero_brain_voxels) == 0:
        print("Warning: No non-zero voxels found in brain mask")
        return dci_data
    
    # Calculate mean and std of non-zero brain voxels
    mean_val = np.mean(non_zero_brain_voxels)
    std_val = np.std(non_zero_brain_voxels)
    
    if std_val == 0:
        print("Warning: Standard deviation is zero, cannot z-score")
        return dci_data
    
    # Z-score the entire DCI map
    zscore_data = np.zeros_like(dci_data)
    
    # Only z-score voxels within brain mask
    mask_indices = brain_mask_data > 0
    zscore_data[mask_indices] = (dci_data[mask_indices] - mean_val) / std_val
    
    print(f"Z-score normalization: mean={mean_val:.4f}, std={std_val:.4f}")
    print(f"Z-scored range: {np.min(zscore_data[mask_indices]):.4f} to {np.max(zscore_data[mask_indices]):.4f}")
    
    return zscore_data

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

def clean_reference_data(mean_data, std_data, min_std=0.01):
    """
    Clean corrupted reference statistics
    """
    # Make copies to avoid modifying originals
    clean_mean = mean_data.copy()
    clean_std = std_data.copy()
    
    # Step 1: Replace inf with NaN first
    clean_mean[np.isinf(clean_mean)] = np.nan
    clean_std[np.isinf(clean_std)] = np.nan
    
    # Step 2: Set very small std to minimum threshold
    clean_std[clean_std < min_std] = min_std
    
    # Step 3: For connections with NaN mean/std, use global statistics
    valid_mean = clean_mean[np.isfinite(clean_mean)]
    valid_std = clean_std[np.isfinite(clean_std)]
    
    if len(valid_mean) > 0:
        global_mean = np.median(valid_mean)  # Use median (robust to outliers)
        global_std = np.median(valid_std)
    else:
        global_mean = 0.0
        global_std = 0.5
    
    # Replace NaN values
    clean_mean[np.isnan(clean_mean)] = global_mean
    clean_std[np.isnan(clean_std)] = global_std
    
    print(f"Replaced {np.sum(np.isinf(mean_data))} inf means with {global_mean:.4f}")
    print(f"Replaced {np.sum(np.isinf(std_data))} inf stds with {global_std:.4f}")
    print(f"Set {np.sum(std_data < min_std)} small stds to minimum {min_std}")
    
    return clean_mean, clean_std

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

############################################################################
L_mean_clean, L_std_clean = clean_reference_data(L_mean_total, L_std_total)
R_mean_clean, R_std_clean = clean_reference_data(R_mean_total, R_std_total)

# Verify cleaning worked
print(f"After cleaning - L_mean inf: {np.sum(np.isinf(L_mean_clean))}")
print(f"After cleaning - L_std inf: {np.sum(np.isinf(L_std_clean))}")
print(f"After cleaning - L_std zeros: {np.sum(L_std_clean == 0)}")
print(f"After cleaning - R_mean inf: {np.sum(np.isinf(R_mean_clean))}")
print(f"After cleaning - R_std inf: {np.sum(np.isinf(R_std_clean))}")
print(f"After cleaning - R_std zeros: {np.sum(R_std_clean == 0)}")

# Use cleaned data for z-score calculations
L_mean_total = L_mean_clean
L_std_total = L_std_clean
R_mean_total = R_mean_clean
R_std_total = R_std_clean

def fix_diagonal_corruption(mean_matrix, std_matrix, expected_diagonal_mean=12.0, expected_diagonal_std=1.0):
    """
    Fix corrupted diagonal elements in reference connectivity matrices
    """
    fixed_mean = mean_matrix.copy()
    fixed_std = std_matrix.copy()
    
    # Identify which part is the intrahemispheric block (diagonal submatrix)
    if mean_matrix.shape[0] == 2218:  # LH matrix
        intra_block = fixed_mean[:, :2218]  # LH-to-LH block
        intra_std_block = fixed_std[:, :2218]
    else:  # RH matrix (2038)
        intra_block = fixed_mean[:, :2038]  # RH-to-RH block  
        intra_std_block = fixed_std[:, :2038]
    
    # Fix diagonal elements
    np.fill_diagonal(intra_block, expected_diagonal_mean)
    np.fill_diagonal(intra_std_block, expected_diagonal_std)
    
    return fixed_mean, fixed_std

# Fix both hemisphere reference data
L_mean_fixed, L_std_fixed = fix_diagonal_corruption(L_mean_total, L_std_total, 
                                                   expected_diagonal_mean=12.0, 
                                                   expected_diagonal_std=1.0)

R_mean_fixed, R_std_fixed = fix_diagonal_corruption(R_mean_total, R_std_total,
                                                   expected_diagonal_mean=12.9, 
                                                   expected_diagonal_std=1.0)

# Replace your reference data
L_mean_total = L_mean_fixed
L_std_total = L_std_fixed  
R_mean_total = R_mean_fixed
R_std_total = R_std_fixed
#############################################################################

# Initialize arrays to store AI values
LH_AI_total = []
RH_AI_total = []

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

            assert np.allclose(Corr_RL, Corr_LR.T), "RL and LR matrices should be transposes"

            print("L_mean_total range:", np.min(L_mean_total), np.max(L_mean_total))
            print("R_mean_total range:", np.min(R_mean_total), np.max(R_mean_total))

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

                hdr_final_LH = nib.Nifti1Image(LH_Seed_vol, affine=LH_Seed.affine)
                nib.save(hdr_final_LH, os.path.join(path_lh_ai, filename))

                AI_counter_LH += 1

            # Sum LH seeds
            LH_Seed_AI_sum_vol_ges = np.zeros_like(LH_Seed_vol)
            for filename in sorted(os.listdir(path_lh_ai)):
                if not filename.startswith('.'):
                    LH_Seed_AI_sum = nib.load(os.path.join(path_lh_ai, filename))
                    LH_Seed_AI_sum_vol = LH_Seed_AI_sum.get_fdata()
                    LH_Seed_AI_sum_vol_ges += LH_Seed_AI_sum_vol

            hdr_final_LH_ges = nib.Nifti1Image(LH_Seed_AI_sum_vol_ges, affine=LH_Seed_AI_sum.affine)
            nib.save(hdr_final_LH_ges, os.path.join(path_lh_ai, f'LH_Seed_AI_sum_vol_ges_orig_{thr}.nii.gz'))

            # Process RH seeds
            AI_counter_RH = 0
            for filename in sorted(file for file in os.listdir(os.path.join(session_path, f'{subject}_{folder_rh}')) if not file.startswith('.')):
                RH_Seed = nib.load(os.path.join(session_path, f'{subject}_{folder_rh}', filename))
                RH_Seed_vol = RH_Seed.get_fdata()

                indices = np.where(RH_Seed_vol == 1)
                RH_Seed_vol[indices] = ALT_Abnormality_Score_RH_all_z4[right_seeds.index(filename)]

                hdr_final_RH = nib.Nifti1Image(RH_Seed_vol, affine=RH_Seed.affine)
                nib.save(hdr_final_RH, os.path.join(path_rh_ai, filename))

                AI_counter_RH += 1

            # Sum RH seeds
            RH_Seed_AI_sum_vol_ges = np.zeros_like(RH_Seed_vol)
            for filename in sorted(os.listdir(path_rh_ai)):
                if not filename.startswith('.'):
                    RH_Seed_AI_sum = nib.load(os.path.join(path_rh_ai, filename))
                    RH_Seed_AI_sum_vol = RH_Seed_AI_sum.get_fdata()
                    RH_Seed_AI_sum_vol_ges += RH_Seed_AI_sum_vol

            hdr_final_RH_ges = nib.Nifti1Image(RH_Seed_AI_sum_vol_ges, affine=RH_Seed_AI_sum.affine)
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


print("Starting smoothing process with corrected mask-then-smooth approach...")
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

        # Check if both LH and RH AI paths exist
        lh_ai_path = os.path.join(session_path, f'SEED_LH_AI_thr_{thr}')
        rh_ai_path = os.path.join(session_path, f'SEED_RH_AI_thr_{thr}')
        
        if not (os.path.exists(lh_ai_path) and os.path.exists(rh_ai_path)):
            print(f"Missing LH or RH AI paths for {subject} {session}, skipping...")
            continue

        # Define input files for LH and RH
        lh_input_file = os.path.join(lh_ai_path, f'LH_Seed_AI_sum_vol_ges_orig_{thr}.nii.gz')
        rh_input_file = os.path.join(rh_ai_path, f'RH_Seed_AI_sum_vol_ges_orig_{thr}.nii.gz')
        
        # Check if both input files exist
        if not (os.path.exists(lh_input_file) and os.path.exists(rh_input_file)):
            print(f"Missing LH or RH input files for {subject} {session}, skipping...")
            continue

        try:
            print(f"Processing combined LH+RH map for {subject} {session}")
            
            lh_img = nib.load(lh_input_file)
            rh_img = nib.load(rh_input_file)
            
            lh_data = lh_img.get_fdata()
            rh_data = rh_img.get_fdata()
            
            combined_data = lh_data + rh_data
            
            # Create combined image
            combined_img = nib.Nifti1Image(combined_data, affine=lh_img.affine)
            
            # Create a combined output directory
            combined_ai_path = os.path.join(session_path, f'SEED_COMBINED_AI_thr_{thr}')
            if not os.path.exists(combined_ai_path):
                os.makedirs(combined_ai_path)
            
            # Save combined map
            combined_file = os.path.join(combined_ai_path, f'Combined_Seed_AI_sum_vol_ges_orig_{thr}.nii.gz')
            nib.save(combined_img, combined_file)
            print(f"Saved combined map: {combined_file}")
            
            
            masked_unsmoothed_file = os.path.join(combined_ai_path, f'Combined_Seed_AI_sum_vol_ges_orig{thr}_masked_unsmoothed_{subject}_{session}.nii.gz')
            success = apply_brain_mask(combined_file, brain_mask_path, masked_unsmoothed_file)
            
            if not success:
                print(f"Failed to apply brain mask for {subject} {session}")
                continue
                
            print(f"Applied brain mask before smoothing: {masked_unsmoothed_file}")
            
            smoothed_masked_file = os.path.join(combined_ai_path, f'Combined_Seed_AI_sum_vol_ges_orig{thr}_masked_smoothed_{subject}_{session}.nii.gz')
            # subprocess.run(['fslmaths', masked_unsmoothed_file, '-kernel', 'gauss', '12', '-fmean', smoothed_masked_file], check=True)
            subprocess.run(['fslmaths', masked_unsmoothed_file, '-kernel', 'gauss', '12', '-fmean', smoothed_masked_file], check=True)
            print(f"Smoothed masked map: {smoothed_masked_file}")
            
            final_masked_file = os.path.join(combined_ai_path, f'Combined_Seed_AI_sum_vol_ges_orig{thr}_final_masked_{subject}_{session}.nii.gz')
            apply_brain_mask(smoothed_masked_file, brain_mask_path, final_masked_file)
            print(f"Final masked map: {final_masked_file}")

            
            brain_mask_img = nib.load(brain_mask_path)
            brain_mask_data = brain_mask_img.get_fdata()
            
            final_smoothed_img = nib.load(final_masked_file)
            final_smoothed_data = final_smoothed_img.get_fdata()
            
            print(f"Applying z-score normalization to smoothed map for {subject} {session}")
            zscore_smoothed_data = zscore_dci_map(final_smoothed_data, brain_mask_data)
            
            zscore_smoothed_file = os.path.join(combined_ai_path, f'Combined_Seed_AI_smoothed_zscore_{subject}_{session}.nii.gz')
            zscore_smoothed_img = nib.Nifti1Image(zscore_smoothed_data, affine=final_smoothed_img.affine)
            nib.save(zscore_smoothed_img, zscore_smoothed_file)
            print(f"Saved z-scored smoothed map: {zscore_smoothed_file}")

            zscore_smoothed_masked_file = os.path.join(combined_ai_path, f'Combined_Seed_AI_smoothed_zscore_masked_{subject}_{session}.nii.gz')
            apply_brain_mask(zscore_smoothed_file, brain_mask_path, zscore_smoothed_masked_file)
            print(f"Final z-scored smoothed masked map: {zscore_smoothed_masked_file}")
            
        except Exception as e:
            print(f"Error processing combined map for {subject} {session}: {str(e)}")

print("Processing complete with corrected mask-then-smooth approach.")