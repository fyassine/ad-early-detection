#!/bin/bash

# Directory containing the data
DATA_DIR=/data2/core-rad/swunderl/Glioma_Sophia/data_dummy

# Directory to store the changed data
CHANGED_DIR=/data2/core-rad/swunderl/Glioma_Sophia/data_dummy_changed

# Counter for new subject numbers
SUBJECT_COUNTER=1

# Log file to store mappings between original and new subject names
LOG_FILE="$CHANGED_DIR/subject_mapping.log"

# Function to increment subject counter
increment_counter() {
    SUBJECT_COUNTER=$((SUBJECT_COUNTER + 1))
}

# Iterate over folders in the data directory
for folder in $DATA_DIR/sub-HGG_* $DATA_DIR/sub-LGG_* $DATA_DIR/sub-NG_*; do
    echo "Processing folder: $folder"
    if [ -d "$folder" ]; then
        # Extract the original subject number
        OLD_SUBJECT=$(basename "$folder")

        # Construct the new subject name
        NEW_SUBJECT="sub-$SUBJECT_COUNTER"

        # Create the new subject directory if it doesn't exist
        mkdir -p "$CHANGED_DIR/$NEW_SUBJECT"

        # Rename files in anat directory
        if [ -d "$folder/anat" ]; then
            mkdir -p "$CHANGED_DIR/$NEW_SUBJECT/anat"
            cp "$folder/anat/${OLD_SUBJECT}_T1w.gz" "$CHANGED_DIR/$NEW_SUBJECT/anat/${NEW_SUBJECT}_T1w.gz"
            cp "$folder/anat/${OLD_SUBJECT}_T1w.json" "$CHANGED_DIR/$NEW_SUBJECT/anat/${NEW_SUBJECT}_T1w.json"
        fi

        # Rename files in func directory
        if [ -d "$folder/func" ]; then
            mkdir -p "$CHANGED_DIR/$NEW_SUBJECT/func"
            cp "$folder/func/${OLD_SUBJECT}_task-rest_bold.json" "$CHANGED_DIR/$NEW_SUBJECT/func/${NEW_SUBJECT}_task-rest_bold.json"
            cp "$folder/func/${OLD_SUBJECT}_task-rest_bold.nii.gz" "$CHANGED_DIR/$NEW_SUBJECT/func/${NEW_SUBJECT}_task-rest_bold.nii.gz"
        fi

        # Log the mapping between old and new subject names
        echo "$OLD_SUBJECT -> $NEW_SUBJECT" >> "$LOG_FILE"

        # Increment subject counter
        increment_counter
    fi
done

