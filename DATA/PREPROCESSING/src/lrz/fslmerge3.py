import os
import subprocess

def find_files(folder_path):
    subjects = []
    for subject_name in os.listdir(folder_path):
        subject_folder = os.path.join(folder_path, subject_name)
        bold_folders = [folder for folder in os.listdir(subject_folder) if folder.startswith("bold")]
        for bold_folder in bold_folders:
            bold_path = os.path.join(subject_folder, bold_folder)
            file_005 = find_file(bold_path, '005')
            file_006 = find_file(bold_path, '006')
            subject = {
                "subject_name": subject_name,
                "bold_folder": bold_folder,
                "file_005": file_005,
                "file_006": file_006
            }
            subjects.append(subject)
    return subjects

def find_file(bold_path, folder):
    folder_path = os.path.join(bold_path, folder)
    if os.path.exists(folder_path):
        files = [f for f in os.listdir(folder_path) if not f.startswith('.')]
        reorient_files = [f for f in files if "reorient" in f]
        if reorient_files:
            return reorient_files[0]
        elif files:
            return files[0]
    return None

def merge_files(subjects):
    for subject in subjects:
        file_005 = subject["file_005"]
        file_006 = subject["file_006"]
        
        if not file_005:
            file_006_007 = [file_006]
            file_007 = find_file(os.path.join(folder_path, subject["subject_name"], subject["bold_folder"]), '007')
            if file_007:
                file_006_007.append(file_007)
            input_files = [os.path.join(folder_path, subject["subject_name"], subject["bold_folder"], folder, file) for folder, file in zip(['006', '007'], file_006_007) if file]
        else:
            input_files = [os.path.join(folder_path, subject["subject_name"], subject["bold_folder"], folder, file) for folder, file in zip(['005', '006'], [file_005, file_006]) if file]
        
        output_file = os.path.join(folder_path, subject["subject_name"], subject["bold_folder"], f"merged_{subject['subject_name']}.nii.gz")

        try:
            subprocess.run(["fslmerge", "-t", output_file] + input_files, check=True)
            print(f"Files for subject {subject['subject_name']} merged successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            continue

# Specify the folder path containing subject directories
folder_path = "/dss/dssfs02/lwp-dss-0001/pn72zi/pn72zi-dss-0000/di76gez/Glioma/Glioma_Sophia_Publication/RawData/data"

# Find files in the bold folders
subjects = find_files(folder_path)

# Merge files for each subject
merge_files(subjects)

