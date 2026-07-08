#!/bin/bash

# Input and output directories
input_dir="/data2/core-rad-fni/Delcode_faschmit/data/Converter_newcriteria/M60"
output_dir="/data2/core-rad-fni/Delcode_faschmit/data/Converter_newcriteria_nifti/M60"

mkdir -p "$output_dir"

# Target subdirectory patterns
target_patterns=("MPRAGE" "RestingState")

# Loop through each main folder in the input directory
for main_folder in "$input_dir"/*; do
    if [ -d "$main_folder" ]; then
        # Check for "scans" directory case-insensitively
        scans_dir=$(find "$main_folder" -maxdepth 1 -type d -iname "scans")
        if [ -n "$scans_dir" ]; then
            # Loop through the target patterns
            for pattern in "${target_patterns[@]}"; do
                # Find subdirectories matching the pattern, excluding ones ending with "ND"
                matching_dirs=$(find "$scans_dir" -maxdepth 1 -type d -iname "*$pattern*" ! -iname "*ND")
                for dicom_folder in $matching_dirs; do
                    if [ -d "$dicom_folder" ]; then
                        # Extract the main folder name
                        folder_name=$(basename "$main_folder")

                        # Extract the subdirectory name
                        subfolder_name=$(basename "$dicom_folder")

                        # Create corresponding output folder
                        output_subdir="$output_dir/$folder_name/$subfolder_name"
                        mkdir -p "$output_subdir"

                        # Run dcm2niix
                        dcm2niix -f "%f_%p_%t_%s" -p y -z y -ba n -o "$output_subdir" "$dicom_folder"

                        echo "Converted $dicom_folder to $output_subdir"
                    fi
                done
            done
        fi
    fi
done