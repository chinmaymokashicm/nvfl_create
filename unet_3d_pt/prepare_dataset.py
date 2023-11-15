"""
Prepares dataset for NVFlare from original folder structure
Original dataset does not have a labeled test set
"""
import os, shutil, random, json

random.seed(2023)

folderpath_original = "/data/cmokashi/msd/brain_tumor"
folderpath_new = "unet_3d_pt/dataset"

list_IDs = list(set([f.split(".")[0].split("_")[1] for f in os.listdir(os.path.join(folderpath_original, "imagesTr")) if not f.startswith(".")]))
prefix_filename = "BRATS"
n_IDs = len(list_IDs)
list_train_IDs = random.sample(list_IDs, int(0.8 * n_IDs))
list_test_IDs = [id for id in list_IDs if id not in list_train_IDs]

dict_mapping_acronym = {"train": "Tr", "test": "Ts"}

folderpath_train_images = os.path.join(folderpath_original, "imagesTr")
folderpath_train_labels = os.path.join(folderpath_original, "labelsTr")

for train_or_test in ["train", "test"]:
    for type in ["images", "labels"]:
        folderpath = os.path.join(folderpath_new, train_or_test, type)
        if not os.path.exists(folderpath):
            os.makedirs(folderpath)
            
# Prepare dataset.json    
with open(os.path.join(folderpath_original, "dataset.json"), "r") as f:
    dict_meta = json.load(f)
    
dict_meta_new = {}
for key, value in dict_meta.items():
    if key not in ["training", "test"]:
        dict_meta_new[key] = dict_meta[key]
    else:
        dict_meta_new[key] = []

for train_or_test, list_ids in {"train": list_train_IDs, "test": list_test_IDs}.items():
    for id in list_ids:
        filename = f"{prefix_filename}_{id}.nii.gz"
        filepath_image_original = os.path.join(folderpath_train_images, filename)
        filepath_label_original = os.path.join(folderpath_train_labels, filename)
        
        filepath_image_new = os.path.join(folderpath_new, train_or_test, "images")
        filepath_label_new = os.path.join(folderpath_new, train_or_test, "labels")
        
        shutil.copy2(filepath_image_original, filepath_image_new)
        shutil.copy2(filepath_label_original, filepath_label_new)
        
        # Append to dataset.json
        dict_rel_filepaths = {item[:-1]: os.path.join(".", f"{item}{dict_mapping_acronym[train_or_test]}", filename) for item in ["images", "labels"]}
        if train_or_test == "train":
            dict_meta_new["training"].append(dict_rel_filepaths)
        else:
            dict_meta_new["test"].append(dict_rel_filepaths)
            
with open(os.path.join(folderpath_new, "dataset.json"), "w") as f:
    json.dump(dict_meta_new, f)
    
# for train_or_test in os.listdir(folderpath_new):
#     if os.path.isdir(train_or_test):
#         for image_or_label in ["images", "labels"]:
#             print(train_or_test, len(os.listdir(os.path.join(folderpath_new, train_or_test, image_or_label))))