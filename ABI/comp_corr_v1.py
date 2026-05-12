#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 12:46:42 2020

@author: neurolab
"""

# suppress warnings
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Essential modules for data organization and manipulation
import numpy as np
import pandas as pd
import os
import shutil

# Nilearn modules for fMRI analysis
import nibabel as nib
from nilearn.input_data import NiftiMasker, NiftiLabelsMasker
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure
from nilearn import input_data

# Other modules for computation and plotting
from scipy import stats
from scipy.ndimage.measurements import center_of_mass
import seaborn as sns
#import matplotlib.pyplot as plt
import itertools
import time




path_onc = '/dss/dssfs03/pn72zi/pn72zi-dss-0001/di38jor/Projects/TMS_v2/outputs/postprocessed'
output_path = '/dss/dssfs03/pn72zi/pn72zi-dss-0001/di38jor/Projects/TMS_v2/outputs/correlations_test_v1'

folder_lh = 'SEED_LH'
folder_rh = 'SEED_RH'
template = 'final_mask_wThalamus_wglobusPall_2.82_2.82_2.82_reorient_1.88_18_18_GSP.nii.gz'

# Get list of all subjects
subjects = [d for d in os.listdir(path_onc) if d.startswith('sub-')]

for subject in subjects:
    print(f"Processing subject: {subject}")

    # Get all sessions for this subject
    sessions = [d for d in os.listdir(os.path.join(path_onc, subject)) if d.startswith('ses-')]

    for session in sessions:
        print(f"Processing session: {session}")

        sub_number = subject.split('-')[1]

        # Create output directories
        path1 = os.path.join(output_path, subject, session, f'{subject}_{folder_rh}')
        path2 = os.path.join(output_path, subject, session, f'{subject}_{folder_lh}')

        os.makedirs(path1, exist_ok=True)
        os.makedirs(path2, exist_ok=True)

        try:
            file_path = os.path.join(path_onc, subject, session, f'{subject}_{session}_task-rest_space-MNI152NLin2009cAsym_res-2_desc-ICAAROMA2Phys1GS_bold_reoriented.nii.gz')
            print(file_path)
            initial_data = nib.load(file_path)
            initial_data2 = initial_data.get_fdata()

            timepoints = len(initial_data2[0][0][0])

            if timepoints != 238:
                print(f"Warning: timepoints of data is {timepoints}")

            img = nib.load('/dss/dssfs02/lwp-dss-0001/pn72zi/pn72zi-dss-0000/di76gez/mask/'+str(template))
            vol = img.get_fdata()
            vol = vol/100


            ###RH####

            initial_data_dummy = initial_data2.copy()

            initial_data_dummy[49:97, :, :] = 0
            initial_data2_RH = initial_data_dummy###only one hemisphere

            # mask_final_RH =mask2[0:38, :, :]

            # n_voxels = np.prod(initial_data2_RH.shape[:-1])

            # n_voxels_mask = np.prod(mask_final_RH.shape)

            # #- Reshape 4D array to 2D array n_voxels by n_volumes
            # initial_data_2d_RH = np.reshape(initial_data2_RH, (n_voxels, initial_data2_RH.shape[-1]))

            # #reshape mask

            # mask_1d_RH = np.reshape(mask_final_RH,(n_voxels_mask))



            # mask_2d_RH = np.zeros((124,1),dtype=mask_1d_RH.dtype) + mask_1d_RH

            # mask_2d_transp_RH = np.transpose(mask_2d_RH)

            # ###if I need the corr value related to the voxel do not multiply here, set later all NaN to 999 or so (NaN because Nenner ist 0 für corr Berechnung)
            # initial_data_2d_masked_RH = initial_data_2d_RH*mask_2d_transp_RH

            # ###extract only non zero values
            # #initial_data_2d_masked_sqeezed_RH = initial_data_2d_masked_RH[~np.all(initial_data_2d_masked_RH == 0, axis=1)]

            # ################################

            # initial_data_2d_masked_sqeezed_RH = initial_data_2d_masked_RH


                # RH seeds
            tvol_RH=vol.copy()
            tvol_RH[49:97,:,:]=0
            hdr_RH=tvol_RH.copy()

            hdr_mask_RH = 0*vol

            tvol_RH[tvol_RH>0.22]=1
            tvol_RH[tvol_RH<1]=0
            mask_RH=tvol_RH.copy()

            ####masking before####
            n_voxels_mask_RH = np.prod(hdr_RH.shape)

            mask_1d_RH = np.reshape(hdr_RH,(n_voxels_mask_RH))


            hdr_mask_2D_RH = np.zeros((timepoints,1),dtype=mask_1d_RH.dtype) + mask_1d_RH
            hdr_mask_2D_trans_RH = np.transpose(hdr_mask_2D_RH)


            n_voxels_RH = np.prod(initial_data2_RH.shape[:-1])
            initial_data_2d_RH = np.reshape(initial_data2_RH, (n_voxels_RH, initial_data2_RH.shape[-1]))


            initial_data_2d_masked_RH = initial_data_2d_RH*hdr_mask_2D_trans_RH

            initial_data_2d_masked_RH = np.reshape(initial_data_2d_masked_RH, (97, 115, 97, timepoints))




            #niftiwrite(hdr,'/Users/neurolab/Desktop/Stephan/Oncology/Templates/final_mask/LH_Sym_final_mask_wThalamus_2.82_2.82_2.82.nii.gz');

            #vox_sum = np.zeros((2,4))

            #initial_data2_RH_dummy = 0*initial_data2_RH
            #initial_data2_RH_dummy[:,:,:,:] = 1000

            initial_data2_RH_corr = np.empty((0, timepoints))

            cnt = 0
            right_seeds = []
            for i in range(1, 49, 4):
                print(i)
                for j in range(3, 115, 4):
                    for k in range(8, 97, 4):
                        basevol_RH = 0 * vol
                        basevol_RH[i - 2:i + 2, j - 2:j + 2, k - 2:k + 2] = 1
                        maskvol_RH = mask_RH * basevol_RH
                        vox_RH = np.argwhere(maskvol_RH == 1)
                        if len(vox_RH) > 4:
                            cnt += 1
                            hdr_RH = maskvol_RH
                            a_RH = (1/(len(vox_RH)))*sum(sum(sum(initial_data_2d_masked_RH[i-2:i+2,j-2:j+2,k-2:k+2,:])))

                            #initial_data2_RH_corr = a
                            initial_data2_RH_corr = np.vstack([initial_data2_RH_corr, a_RH])#, axis=0)

                            seed_name = f'Seed_0{i}_{j}_{k}.nii.gz' if i < 10 else f'Seed_{i}_{j}_{k}.nii.gz'
                            right_seeds.append(seed_name)
                            seed_path = os.path.join(output_path, subject, session, f'sub-{sub_number}_{folder_rh}', seed_name)
                            hdr_final_RH = nib.Nifti1Image(hdr_RH, affine=img.affine)
                            nib.save(hdr_final_RH, seed_path)
                            hdr_mask_RH = hdr_mask_RH + hdr_RH

    #    np.save(os.path.join(path_onc, f'sub-{sub_number}', 'ses-01',
    #                      f'sub-{sub_number}_initial_data2_RH_BOLD'), initial_data2_RH_corr)
            np.save(os.path.join(output_path, subject, session, f'{subject}_{session}_initial_data2_RH_BOLD'), initial_data2_RH_corr)



            #initial_data_2d_masked_RH[34*77*65+23*65+37*1,:]

            # initial_data_2d_masked_RH2 = 0*initial_data_2d_masked_RH

            #initial_data_2d_masked_RH[~np.isnan(initial_data_2d_masked_RH).any(axis=1)]


            #mask = np.all(np.isnan(initial_data_2d_masked_RH), axis=1)


            #a = initial_data_2d_masked_RH[((initial_data_2d_masked_RH < 999) & (initial_data_2d_masked_RH > -1)).all(axis=1)]

            #b = a[(a > 0).all(axis=1)]


            #a[~mask]

            #new_array = [tuple(row) for row in initial_data_2d_masked_RH]

            #initial_data_2d_masked_RH_dropdup= np.unique(new_array, axis = 0)


            #initial_data_2d_masked_RH2,idx=np.unique(initial_data_2d_masked_RH, axis=0,return_index=True)

            #initial_data_2d_masked_RH2 = initial_data_2d_masked_RH[np.sort(idx)]



            #indexes = np.unique(new_array, return_index=True)[1]

            #[initial_data_2d_masked_RH_dropdup[index] for (index) in sorted(indexes)]



            #initial_data_2d_masked_RH = np.reshape(initial_data_2d_masked_RH, (65, 77, 65, 119))
            #initial_data_2d_masked_RH_nii = nib.Nifti1Image(initial_data_2d_masked_RH, affine = initial_data.affine)
            #nib.save(initial_data_2d_masked_RH_nii, '/Users/neurolab/Desktop/Drmed/GSP/derivatives/participant/fmriprep/sub-'+str(counter)+'/ses-1/func/sub-'+str(counter)+'_ses-1_task-rest_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold_skipped_5_smoothed_8_cleaned_seeded.nii.gz')

            ##############################################################
            #########################


            hdr_RH[:,:,:] = 0
            whole_hemisphere_rh_start = hdr_RH


            import os
            os.chdir(path1)
            for filename in os.listdir(os.getcwd()):
                #print(filename)
                if filename !='.DS_Store':
                    whole_hemisphere_rh = nib.load(filename)
                    vol = whole_hemisphere_rh.get_fdata()

                    whole_hemisphere_rh_start = whole_hemisphere_rh_start + vol

            whole_hemisphere_rh_start_final= nib.Nifti1Image(whole_hemisphere_rh_start, affine=img.affine)
            nib.save(whole_hemisphere_rh_start_final,(path1 + '_whole_hemisphere_rh.nii.gz'))




            #LH

            file_path = os.path.join(path_onc, f'sub-{sub_number}', session, f'sub-{sub_number}_{session}_task-rest_space-MNI152NLin2009cAsym_res-2_desc-ICAAROMA2Phys1GS_bold_reoriented.nii.gz')

            initial_data = nib.load(file_path)
            initial_data2 = initial_data.get_fdata()


            img = nib.load('/dss/dssfs02/lwp-dss-0001/pn72zi/pn72zi-dss-0000/di76gez/mask/'+str(template))

            #img = nib.load('/Users/Wu/Desktop/Research/GSP/'+str(template))
            vol = img.get_fdata()

            vol = vol / 100

            ###LH####

            initial_data_dummy = initial_data2.copy()

            initial_data_dummy[1:48, :, :] = 0
            initial_data2_LH = initial_data_dummy  # only one hemisphere

            # LH seeds
            tvol_LH = vol.copy()
            tvol_LH[1:48, :, :] = 0
            hdr_LH = tvol_LH.copy()

            hdr_mask_LH = 0 * vol

            tvol_LH[tvol_LH > 0.22] = 1
            tvol_LH[tvol_LH < 1] = 0
            mask_LH = tvol_LH.copy()

            n_voxels_mask_LH = np.prod(mask_LH.shape)

            mask_1d_LH = np.reshape(mask_LH, (n_voxels_mask_LH))

            hdr_mask_2D_LH = np.zeros((timepoints, 1), dtype=mask_1d_LH.dtype) + mask_1d_LH
            hdr_mask_2D_trans_LH = np.transpose(hdr_mask_2D_LH)

            n_voxels_LH = np.prod(initial_data2_LH.shape[:-1])
            initial_data_2d_LH = np.reshape(initial_data2_LH, (n_voxels_LH, initial_data2_LH.shape[-1]))

            initial_data_2d_masked_LH = initial_data_2d_LH * hdr_mask_2D_trans_LH

            initial_data_2d_masked_LH = np.reshape(initial_data_2d_masked_LH, (97, 115, 97, timepoints))

            initial_data2_LH_corr = np.empty((0, timepoints))

            cnt = 0
            left_seeds = []
            for i in range(49, 97, 4):
                print(i)
                for j in range(3, 115, 4):
                    for k in range(3, 97, 4):
                        basevol_LH = 0 * vol
                        basevol_LH[i - 2:i + 2, j - 2:j + 2, k - 2:k + 2] = 1
                        maskvol_LH = mask_LH * basevol_LH
                        vox_LH = np.argwhere(maskvol_LH == 1)
                        if len(vox_LH) > 4:
                            cnt = cnt + 1
                            hdr_LH = maskvol_LH
                            a_LH = (1/(len(vox_LH)))*sum(sum(sum(initial_data_2d_masked_LH[i-2:i+2,j-2:j+2,k-2:k+2,:])))

                            initial_data2_LH_corr = np.vstack([initial_data2_LH_corr, a_LH])

                            seed_name = f'Seed_{i}_{j}_{k}.nii.gz'
                            left_seeds.append(seed_name)
                            seed_path = os.path.join(output_path, subject, session, f'sub-{sub_number}_{folder_lh}', seed_name)
                            hdr_final_LH = nib.Nifti1Image(hdr_LH, affine=img.affine)
                            nib.save(hdr_final_LH, seed_path)
                            hdr_mask_LH = hdr_mask_LH + hdr_LH

            np.save(os.path.join(output_path, subject, session, f'{subject}_{session}_initial_data2_LH_BOLD'), initial_data2_LH_corr)



            hdr_LH[:,:,:] = 0
            whole_hemisphere_lh_start = hdr_LH



            import os
            os.chdir(path2)
            for filename in os.listdir(os.getcwd()):
                if filename !='.DS_Store':

                    whole_hemisphere_lh = nib.load(filename)
                    vol = whole_hemisphere_lh.get_fdata()
                    whole_hemisphere_lh_start = whole_hemisphere_lh_start + vol

            whole_hemisphere_lh_start_final= nib.Nifti1Image(whole_hemisphere_lh_start, affine=img.affine)
            nib.save(whole_hemisphere_lh_start_final,(path2 + '_whole_hemisphere_lh.nii.gz'))



            print('saving seeds!!!!!!!!!')
            with open(f'/dss/dssfs03/pn72zi/pn72zi-dss-0001/di38jor/Projects/TMS_v2/codes/seeds_order/{subject}_left_seeds.txt', 'w') as file:
                for item in left_seeds:
                    file.write(f"{item}\n")
            with open(f'/dss/dssfs03/pn72zi/pn72zi-dss-0001/di38jor/Projects/TMS_v2/codes/seeds_order/{subject}_right_seeds.txt', 'w') as file:
                for item in right_seeds:
                    file.write(f"{item}\n")

            #initial_data_2d_masked_LH = np.reshape(initial_data_2d_masked_LH, (65, 77, 65, 124))
            #initial_data_2d_masked_LH_nii = nib.Nifti1Image(initial_data_2d_masked_LH, affine = initial_data.affine)
            #nib.save(initial_data_2d_masked_LH_nii, '/Users/neurolab/Desktop/Stephan/Oncology/sub_21_initial_data_2d_masked_LH_seeds_2mm.nii.gz')




            def corr2_coeff(A, B):
                # Rowwise mean of input arrays & subtract from input arrays themeselves
                A_mA = A - A.mean(1)[:, None]
                B_mB = B - B.mean(1)[:, None]

                # Sum of squares across rows
                ssA = (A_mA**2).sum(1)
                ssB = (B_mB**2).sum(1)

                # Finally get corr coeff
                r = np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))
                ###hier NaNs zu 0 schon setzen? Sonst inf bei z?


                r = np.clip(r, -0.999, 0.999)

                if np.any(r > 0.999):
                    print("r has values greater than 0.999")
                if np.any(r < -0.999):
                    print("r has values less than -0.999")
                exceeding_values = r[(r > 0.999) | (r < -0.999)]
                if exceeding_values.size > 0:
                    print("Values exceeding the threshold:", exceeding_values)
                    print(exceeding_values.size)

                z =  (1/2) * np.log((1+r)/(1-r))

                return z



            # Correlation calculations
            Corr_RR = corr2_coeff(initial_data2_RH_corr, initial_data2_RH_corr)
            Corr_RL = corr2_coeff(initial_data2_RH_corr, initial_data2_LH_corr)
            Corr_LL = corr2_coeff(initial_data2_LH_corr, initial_data2_LH_corr)
            Corr_LR = corr2_coeff(initial_data2_LH_corr, initial_data2_RH_corr)

            # Set nan to 0
            Corr_RR_final = np.nan_to_num(Corr_RR, copy=True, nan=0, posinf=None, neginf=None)
            Corr_RL_final = np.nan_to_num(Corr_RL, copy=True, nan=0, posinf=None, neginf=None)
            Corr_LL_final = np.nan_to_num(Corr_LL, copy=True, nan=0, posinf=None, neginf=None)
            Corr_LR_final = np.nan_to_num(Corr_LR, copy=True, nan=0, posinf=None, neginf=None)

            # Save correlation results
            np.save(os.path.join(output_path, subject, session, f'{subject}_{session}_Corr_RR'), Corr_RR_final)
            np.save(os.path.join(output_path, subject, session, f'{subject}_{session}_Corr_RL'), Corr_RL_final)
            np.save(os.path.join(output_path, subject, session, f'{subject}_{session}_Corr_LL'), Corr_LL_final)
            np.save(os.path.join(output_path, subject, session, f'{subject}_{session}_Corr_LR'), Corr_LR_final)

        except Exception as e:
            print(f"Error processing {subject} {session}: {str(e)}")

print("Processing complete.")
