#!/bin/bash

# Input and output directories
input_dir="/data2/core-rad-fni/Delcode_faschmit/data/Converter_newcriteria_nifti/M60"
bids_dir="/data2/core-rad-fni/Delcode_faschmit/data/Converter_newcriteria_BIDS/M60"

mkdir -p "$bids_dir"

# Loop through each subject in the input directory
for subject_dir in "$input_dir"/*; do
    if [ -d "$subject_dir" ]; then
        # Extract the subject ID (everything before the first hyphen)
        subject_name=$(basename "$subject_dir")
        subject_id=$(echo "$subject_name" | cut -d'-' -f1)

        # Define paths for the target folders
        mprage_nd_folder=$(find "$subject_dir" -type d -iname "*MPRAGE*" )
        restingstate_folder=$(find "$subject_dir" -type d -iname "*RestingState*")

        # Skip subjects that don't have both required folders
        if [ -n "$mprage_nd_folder" ] && [ -n "$restingstate_folder" ]; then
            # Create BIDS directories for the subject
            anat_dir="$bids_dir/sub-${subject_id}/ses-01/anat"
            func_dir="$bids_dir/sub-${subject_id}/ses-01/func"
            mkdir -p "$anat_dir" "$func_dir"

            # Handle T1w anatomical file and its JSON
            t1_file=$(find "$mprage_nd_folder" -type f -iname "*.nii.gz" | head -n 1)
            if [ -n "$t1_file" ]; then
                # Copy NIFTI file
                cp "$t1_file" "$anat_dir/sub-${subject_id}_ses-01_T1w.nii.gz"

                # Find and copy corresponding JSON file
                t1_json="${t1_file%.nii.gz}.json"
                if [ -f "$t1_json" ]; then
                    cp "$t1_json" "$anat_dir/sub-${subject_id}_ses-01_T1w.json"
                fi
            fi

            # Handle BOLD functional file and its JSON
            bold_file=$(find "$restingstate_folder" -type f -iname "*.nii.gz" | head -n 1)
            if [ -n "$bold_file" ]; then
                # Copy NIFTI file
                cp "$bold_file" "$func_dir/sub-${subject_id}_ses-01_task-rest_bold.nii.gz"

                # Find and copy corresponding JSON file
                bold_json="${bold_file%.nii.gz}.json"
                if [ -f "$bold_json" ]; then
                    cp "$bold_json" "$func_dir/sub-${subject_id}_ses-01_task-rest_bold.json"
                fi
            fi

            # Log progress
            echo "Converted $subject_name to BIDS format."
        else
            echo "Skipping $subject_name: Required folders not found."
        fi
    fi
done