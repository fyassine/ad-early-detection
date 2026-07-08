import os
import shutil
import json
from pathlib import Path

def convert_to_nii_gz(source_dir):
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".nii"):
                nii_file = os.path.join(root, file)
                nii_gz_file = os.path.join(root, file + ".nii.gz")
                print(f"Compressing {nii_file} to {nii_gz_file}")
                #os.system(f"pigz -n {nii_file}")  # You need pigz installed
                #os.rename(nii_gz_file, nii_gz_file[:-3])  # Remove the extra .gz extension

def move_files_to_bids(data_directory, bids_directory):
    for subject_name in os.listdir(data_directory):
        subject_dir = os.path.join(data_directory, subject_name)
        if os.path.isdir(subject_dir):
            bids_subject_dir = os.path.join(bids_directory, 'sub-' + subject_name)
            # Create BIDS subject directory if not exists
            Path(bids_subject_dir).mkdir(parents=True, exist_ok=True)

            # Move anatomical files
            anat_source_dir = os.path.join(subject_dir, "anat")
            anat_destination_dir = os.path.join(bids_subject_dir, 'anat')
            move_anat_files(anat_source_dir, anat_destination_dir, subject_name)

            # Move functional files
            bold_source_dir = os.path.join(subject_dir, "bold")
            bold_destination_dir = os.path.join(bids_subject_dir, 'func')
            move_bold_files(bold_source_dir, bold_destination_dir, subject_name)

def move_anat_files(source_dir, destination_dir, subject_name):
    Path(destination_dir).mkdir(parents=True, exist_ok=True)
    for root, _, files in os.walk(source_dir):
        for file in files:
            source_file = os.path.join(root, file)
            destination_file = os.path.join(destination_dir, "sub-" + subject_name + "_T1w.nii.gz")
            shutil.copy(source_file, destination_file)
            print(source_file)
            print(destination_file)
            print("*"*50)

            # Create JSON sidecar for anatomical data
            json_content = {
                "Modality": "anatomy"
            }
            json_file_path = os.path.join(destination_dir, "sub-" + subject_name + "_T1w.json")
            with open(json_file_path, 'w') as json_file:
                json.dump(json_content, json_file, indent=4)

def move_bold_files(source_dir, destination_dir, subject_name):
    Path(destination_dir).mkdir(parents=True, exist_ok=True)
    merged_file = os.path.join(source_dir, "merged_" + subject_name + ".nii.gz")
    destination_file = os.path.join(destination_dir, "sub-" + subject_name + "_task-rest_bold.nii.gz")
    if os.path.exists(merged_file):
        shutil.copy(merged_file, destination_file)
        print(f"Moved {merged_file} to {destination_dir}")

        # Create JSON sidecar for functional data
        json_content = {
            "RepetitionTime": 3.0,
            "TaskName": "resting_state"
        }
        json_file_path = os.path.join(destination_dir, "sub-" + subject_name + "_task-rest_bold.json")
        with open(json_file_path, 'w') as json_file:
            json.dump(json_content, json_file, indent=4)

# Specify the directory paths
data_directory = "/dss/dssfs02/lwp-dss-0001/pn72zi/pn72zi-dss-0000/di76gez/Glioma/Glioma_Sophia_Publication/RawData/BIDS3"
bids_directory = "/dss/dssfs02/lwp-dss-0001/pn72zi/pn72zi-dss-0000/di76gez/Glioma/Glioma_Sophia_Publication/RawData/BIDS4"

# Convert .nii files to .nii.gz
convert_to_nii_gz(data_directory)

# Move files to BIDS format
move_files_to_bids(data_directory, bids_directory)
print("Done!") 
