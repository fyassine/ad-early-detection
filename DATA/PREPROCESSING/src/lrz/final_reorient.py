import os
import numpy as np
import nibabel as nib



sub_dir='/dss/dssfs02/lwp-dss-0001/pn72zi/pn72zi-dss-0000/di76gez/Glioma/Glioma_random/final_data'
ses=1
for sub in range(1,200):
    try:
        #if sub < 10: no need to change!!! sub.04 macht für 0001 und 0010 und 0900
        print (sub)
        input_file = os.path.join(sub_dir, f"sub-{sub:01}", f"ses-{ses:01}", f"sub-{sub:01}_ses-1_task-rest_run-3_space-MNI152NLin2009cAsym_res-2_desc-ICAAROMA2Phys1GS_bold.nii.gz")
    
        #input_file = os.path.join(sub_dir, f"sub-000{sub}", "ses-1", f"sub-000{sub}_ses-1_task-rest_space-MNI152NLin2009cAsym_res-2_desc-ICAAROMA2Phys1GS_bold.nii.gz")

        #input_file = os.path.join(sub_dir, f"sub-{sub:03d}", "ses-1", f"sub-{sub:04d}_ses-1_task-rest_space-MNI152NLin2009cAsym_res-2_desc-ICAAROMA2Phys1GS_bold.nii.gz")
        output_file = os.path.join(sub_dir, f"sub-{sub:01}", f"ses-{ses:01}", f"sub-{sub:01}_ses-1_task-rest_run-3_space-MNI152NLin2009cAsym_res-2_desc-ICAAROMA2Phys1GS_bold_reoriented.nii.gz")
        #if sub < 100:
        #input_file = os.path.join(sub_dir, f"sub-00{sub}", "ses-1", f"sub-00{sub}_ses-1_task-rest_space-MNI152NLin2009cAsym_res-2_desc-ICAAROMA2Phys1GS_bold.nii.gz")
        #output_file = os.path.join(sub_dir, f"sub-{sub}", "ses-1", f"sub-{sub}_ses-1_task-rest_space-MNI152NLin2009cAsym_res-2_desc-ICAAROMA2Phys1GS_bold_reoriented.nii.gz")
        #if sub < 1000:
        #input_file = os.path.join(sub_dir, f"sub-0{sub}", "ses-1", f"sub-0{sub}_ses-1_task-rest_space-MNI152NLin2009cAsym_res-2_desc-ICAAROMA2Phys1GS_bold.nii.gz")
        #output_file = os.path.join(sub_dir, f"sub-{sub}", "ses-1", f"sub-{sub}_ses-1_task-rest_space-MNI152NLin2009cAsym_res-2_desc-ICAAROMA2Phys1GS_bold_reoriented.nii.gz")





        infor = {
            'L': (1, 'x'),
            'A': (2, 'y'),
            'S': (3, 'z'),
            'R': (-1, '-x'),
            'P': (-2, '-y'),
            'I': (-3, '-z')
        }

        # Load the fMRI
        print(input_file)
        img = nib.load(input_file)

        # Determine the current orientation
        input_orient = nib.aff2axcodes(img.header.get_best_affine())

        # Change the orientation into num
        orient_num = [infor[orient][0] for orient in input_orient]

        # Determine how to change
        right_orient_num = [1, 2, 3]
        abs_orient_num = [abs(num) for num in orient_num]

        direct_fsl = []
        for m in right_orient_num:
                indx_from_raw_num = abs_orient_num.index(m)
                sign_indx = 1 if orient_num[indx_from_raw_num] > 0 else -1
                final_indx = sign_indx * indx_from_raw_num
                direct_fsl.append(infor[input_orient[final_indx]][1])



        direct_final = ' '.join(direct_fsl)
        warning = os.system(f'fslswapdim {input_file} {direct_final} {output_file}')

        #if len(warning) > 0:
        os.system(f'fslorient -forceradiological {output_file}')
        output_orient = os.system(f'mri_info --orientation {output_file}')
    except OSError:
                    print ("Does not exist")
        

