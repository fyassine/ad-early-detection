import os
import numpy as np
import nibabel as nib
import pandas as pd 


# Constants
thr = 4.0

# Paths
folder_paths = ["/dss/dssfs02/lwp-dss-0001/pn72zi/pn72zi-dss-0000/di76gez/NAPKON/final_data_22.1.1",
               "/dss/dssfs02/lwp-dss-0001/pn72zi/pn72zi-dss-0000/di76gez/Delcode/delcode_converter/final_data",
               "/dss/dssfs02/lwp-dss-0001/pn72zi/pn72zi-dss-0000/di76gez/Delcode/hcdelcode/final_data"
                #"/home/swunderlich/Desktop/lmu-rad/resources/NAPKON/final_data_22.1.1"
               ]
path_healthy = {
    "hcp": "/dss/dssfs02/lwp-dss-0001/pn72zi/pn72zi-dss-0000/di76gez/HCP_Aging/final_data/HCP_corr_matrices",
    "gsp": "/dss/dssfs02/lwp-dss-0001/pn72zi/pn72zi-dss-0000/di76gez/GSP/GSP_all_data_together/corr_matrices"
}
path_mask = "/dss/dssfs02/lwp-dss-0001/pn72zi/pn72zi-dss-0000/di35fuv/masks"

# Load healthy stats
h_stats_hcp = {
    "LH": {
        "mean": np.load(os.path.join(path_healthy["hcp"], "L_mean_total.npy"), mmap_mode="r"),
        "std": np.load(os.path.join(path_healthy["hcp"], "L_std_total.npy"), mmap_mode="r")
    },
    "RH": {
        "mean": np.load(os.path.join(path_healthy["hcp"], "R_mean_total.npy"), mmap_mode="r"),
        "std": np.load(os.path.join(path_healthy["hcp"], "R_std_total.npy"), mmap_mode="r")
    }
}
# Load healthy stats
h_stats_gsp = {
    "LH": {
        "mean": np.load(os.path.join(path_healthy["gsp"], "L_mean_total.npy"), mmap_mode="r"),
        "std": np.load(os.path.join(path_healthy["gsp"], "L_std_total.npy"), mmap_mode="r")
    },
    "RH": {
        "mean": np.load(os.path.join(path_healthy["gsp"], "R_mean_total.npy"), mmap_mode="r"),
        "std": np.load(os.path.join(path_healthy["gsp"], "R_std_total.npy"), mmap_mode="r")
    }
}

# Split mask into hemispheres
def split_mask(mask, hem):
    # Function to get half of the mask
    def get_mask_half(template_mask, range_tuples):
        mask_ = np.zeros(template_mask.shape)
        mask_[
            range_tuples[0][0] : range_tuples[0][1],
            range_tuples[1][0] : range_tuples[1][1],
            range_tuples[2][0] : range_tuples[2][1],
        ] = template_mask[
            range_tuples[0][0] : range_tuples[0][1],
            range_tuples[1][0] : range_tuples[1][1],
            range_tuples[2][0] : range_tuples[2][1],
        ]
        return np.squeeze(mask_)
    
    range_tuple_RH = ((0, 48), (0, 115), (0, 97))
    range_tuple_LH = ((48, 97), (0, 115), (0, 97))

    if hem == "RH":
        return get_mask_half(mask, range_tuple_RH)
    elif hem == "LH": 
        return get_mask_half(mask, range_tuple_LH)
    else:
        return
    
# Load masks
mask_list = [os.path.join(path_mask, f) for f in os.listdir(path_mask)]

data = []
for folder_path in folder_paths:
    dataset_name = folder_path.split("/")[-2]
    # Loop through subjects
    for HGG in os.listdir(folder_path):
        if not os.path.isdir(os.path.join(folder_path, HGG)):
            continue
        # Loop through sessions
        for ses in os.listdir(os.path.join(folder_path, HGG)):
            item_path = os.path.join(folder_path, HGG, ses)

            # Loop through hemispheres
            for hem in ["LH", "RH"]:
                try:
                    print(f"Processing {dataset_name} {HGG} {ses} {hem}")
                    path_seed = os.path.join(item_path, f"{HGG}_SEED_{hem}")
                    
                    # Load tumor
                    L = np.load(os.path.join(item_path, f"{HGG}_{ses}_Corr_{hem[0]}L.npy"))
                    R = np.load(os.path.join(item_path, f"{HGG}_{ses}_Corr_{hem[0]}R.npy"))

                    tumor = np.concatenate((L, R), axis=1) if hem == "LH" else np.concatenate((R, L), axis=1)

                    # Calculate Abnormality Score
                    if "NAPKON" == dataset_name:
                        print("NAPKON")
                        abn_score = np.divide((tumor - np.squeeze(h_stats_hcp[hem]["mean"])), np.squeeze(h_stats_hcp[hem]["std"]))
                    else:
                        abn_score = np.divide((tumor - np.squeeze(h_stats_gsp[hem]["mean"])), np.squeeze(h_stats_gsp[hem]["std"]))
                    # Replace NaN values with 0
                    abn_score = np.nan_to_num(abn_score, copy=True, nan=0, posinf=0, neginf=0)

                    # Thresholding
                    abn = np.abs(abn_score)
                    abn[abn <= thr] = 0
                    abn[abn > thr] = 1

                    # Sum of abnormality scores
                    ALT_abn_score_all_z4 = np.sum(abn, axis=1)

                    # Calculate AI
                    n_seed = len(os.listdir(path_seed))
                    AI = np.sum(ALT_abn_score_all_z4) / n_seed
                    print(f"{dataset_name} {HGG} {ses} {hem} AI: {AI}")

                    # Create AI_total array
                    AI_total = np.array([thr, HGG, ses, AI])

                    # Seed
                    AI_counter = 0
                    Seed_sum = np.zeros((97,115,97))
                    dcc_vals = []
                    for filename in sorted(os.listdir(path_seed)):
                        Seed = nib.load(os.path.join(path_seed, filename))
                        Seed_vol = Seed.get_fdata()
                        indices = np.where(Seed_vol == 1)
                        a = np.sum(Seed_vol)
                        Seed_vol[indices] = ALT_abn_score_all_z4[AI_counter]/np.count_nonzero(Seed_vol)
                        dcc_vals.append(int(ALT_abn_score_all_z4[AI_counter]))
                        Seed_sum += Seed_vol
                        AI_counter += 1

                    row = [
                        dataset_name,
                        int(HGG.split("-")[1]),
                        int(ses.split("-")[1]),
                        hem,
                        np.sum(Seed_sum)/n_seed,
                    ]
                    print(row)

                    masked_total_dci = 0
                    for mask_path in sorted(mask_list, key=lambda x: (int(x.split("_")[1]), int(x.split("_")[2]))):

                        mask_img = nib.load(mask_path)
                        mask_data = mask_img.get_fdata()
                        mask = split_mask(mask_data, hem)
                        masked_data = np.multiply(mask, Seed_sum)
                        row.append(np.sum(masked_data)/n_seed)
                        if "thick_7_" in mask_path:
                            masked_total_dci += np.sum(masked_data)/n_seed

                    row.insert(5, masked_total_dci)
                    row.append(dcc_vals)
                    data.append(row)
                    print(f"Processed {dataset_name} {HGG} {ses} {hem}")
                    # seed_sum: dci_map

                         
                except Exception as e:
                    print(e)
                    continue

# Create dataframe and save to csv
print("Saving to csv")
cols = ["dataset", "sub", "ses", "hem", "whole_dci", "sum_masked_dci"]+ \
        [f"{i}_{j}" for i in [7, 17] for j in range(1, 8 if i == 7 else 18)]+ \
        ["dcc_vals"]
df = pd.DataFrame(data, columns=cols)
df = df.sort_values(by=["dataset", "sub", "ses", "hem"])
df.to_csv("/dss/dssfs02/lwp-dss-0001/pn72zi/pn72zi-dss-0000/di35fuv/masked_dci_norm_seed_v3.csv", index=False)
print("Done")
