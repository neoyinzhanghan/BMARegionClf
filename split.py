# split the into train, val, test by 0.8, 0.1, 0.1
# the data is structures like imagenet dataset
# e.g.
# train
#   - adequate
#       - 0_TL.png
#       - 0_TR.png
#       - 0_BL.png
#       - 0_BR.png
#       - 1_TL.png
#       - 1_TR.png
#       - 1_BL.png
#       - 1_BR.png
#       - ...
#   - blood
#       - 0_TL.png
#       - 0_TR.png
#       - 0_BL.png
#       - 0_BR.png
#       - 1_TL.png
#       - 1_TR.png
#       - 1_BL.png
#       - 1_BR.png
#       - ...
#   - clot
#       - 0_TL.png
#       - 0_TR.png
#       - 0_BL.png
#       - 0_BR.png
#       - 1_TL.png
#       - 1_TR.png
#       - 1_BL.png
#       - 1_BR.png
#       - ...
# val
#   - adequate
#       - 0_TL.png
#       - 0_TR.png
#       - 0_BL.png ...

import os
import numpy as np
from PIL import Image
from tqdm import tqdm

input_dir = "/Users/neo/Documents/Research/DeepHeme/LLData/bma_region_clf_data_cropped"
save_dir = "/Users/neo/Documents/Research/DeepHeme/LLData/bma_region_clf_data_cropped_split"

# create the save_dir if it doesn't exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    os.makedirs(os.path.join(save_dir, "train"))
    os.makedirs(os.path.join(save_dir, "val"))
    os.makedirs(os.path.join(save_dir, "test"))
    os.makedirs(os.path.join(save_dir, "train", "adequate"))
    os.makedirs(os.path.join(save_dir, "train", "blood"))
    os.makedirs(os.path.join(save_dir, "train", "clot"))
    os.makedirs(os.path.join(save_dir, "val", "adequate"))
    os.makedirs(os.path.join(save_dir, "val", "blood"))
    os.makedirs(os.path.join(save_dir, "val", "clot"))
    os.makedirs(os.path.join(save_dir, "test", "adequate"))
    os.makedirs(os.path.join(save_dir, "test", "blood"))
    os.makedirs(os.path.join(save_dir, "test", "clot"))

# split the data
dirs = os.listdir(input_dir)
all_subdirs = [subdir for subdir in dirs if os.path.isdir(os.path.join(input_dir, subdir))]
all_adequate_files = []
all_blood_files = []
all_clot_files = []

for subdir in tqdm(all_subdirs, desc="Processing subdirs"):
    print(f"Processing {subdir}...")
    subdir_path = os.path.join(input_dir, subdir)
    subdirs = [os.path.join(subdir_path, file) for file in os.listdir(subdir_path) if os.path.isdir(os.path.join(subdir_path, file))]

    for subsubdir in subdirs:
        adequate_dirs = [os.path.join(subsubdir, file) for file in os.listdir(subsubdir) if file.startswith("adequate")]
        blood_dirs = [os.path.join(subsubdir, file) for file in os.listdir(subsubdir) if file.startswith("blood")]
        clot_dirs = [os.path.join(subsubdir, file) for file in os.listdir(subsubdir) if file.startswith("clot")]

        all_adequate_files.extend([os.path.join(adequate_dir, file) for adequate_dir in adequate_dirs for file in os.listdir(adequate_dir)])
        all_blood_files.extend([os.path.join(blood_dir, file) for blood_dir in blood_dirs for file in os.listdir(blood_dir)])
        all_clot_files.extend([os.path.join(clot_dir, file) for clot_dir in clot_dirs for file in os.listdir(clot_dir)])

# split the data
train_adequate, val_adequate, test_adequate = np.split(all_adequate_files, [int(.8*len(all_adequate_files)), int(.9*len(all_adequate_files))])
train_blood, val_blood, test_blood = np.split(all_blood_files, [int(.8*len(all_blood_files)), int(.9*len(all_blood_files))])
train_clot, val_clot, test_clot = np.split(all_clot_files, [int(.8*len(all_clot_files)), int(.9*len(all_clot_files))])

# save the data
for img_path in tqdm(train_adequate, desc="Saving train adequate"):
    img = Image.open(img_path)
    img.save(os.path.join(save_dir, "train", "adequate", os.path.basename(img_path)))

for img_path in tqdm(val_adequate, desc="Saving val adequate"):
    img = Image.open(img_path)
    img.save(os.path.join(save_dir, "val", "adequate", os.path.basename(img_path)))

for img_path in tqdm(test_adequate, desc="Saving test adequate"): 
    img = Image.open(img_path)
    img.save(os.path.join(save_dir, "test", "adequate", os.path.basename(img_path)))

for img_path in tqdm(train_blood, desc="Saving train blood"):   
    img = Image.open(img_path)
    img.save(os.path.join(save_dir, "train", "blood", os.path.basename(img_path)))

for img_path in tqdm(val_blood, desc="Saving val blood"):
    img = Image.open(img_path)
    img.save(os.path.join(save_dir, "val", "blood", os.path.basename(img_path)))

for img_path in tqdm(test_blood, desc="Saving test blood"):
    img = Image.open(img_path)
    img.save(os.path.join(save_dir, "test", "blood", os.path.basename(img_path)))

for img_path in tqdm(train_clot, desc="Saving train clot"):
    img = Image.open(img_path)
    img.save(os.path.join(save_dir, "train", "clot", os.path.basename(img_path)))

for img_path in tqdm(val_clot, desc="Saving val clot"):
    img = Image.open(img_path)
    img.save(os.path.join(save_dir, "val", "clot", os.path.basename(img_path)))

for img_path in tqdm(test_clot, desc="Saving test clot"):
    img = Image.open(img_path)
    img.save(os.path.join(save_dir, "test", "clot", os.path.basename(img_path)))

print("Done!")