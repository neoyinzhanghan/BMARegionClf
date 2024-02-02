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
import random
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

all_adequate_files = [os.path.join(input_dir, "adequate", file) for file in os.listdir(os.path.join(input_dir, "adequate"))]
all_blood_files = [os.path.join(input_dir, "blood", file) for file in os.listdir(os.path.join(input_dir, "blood"))]
all_clot_files = [os.path.join(input_dir, "clot", file) for file in os.listdir(os.path.join(input_dir, "clot"))]

# split the data
random.shuffle(all_adequate_files)
random.shuffle(all_blood_files)
random.shuffle(all_clot_files)

print(len(all_adequate_files), len(all_blood_files), len(all_clot_files))

train_adequate = all_adequate_files[:int(0.8 * len(all_adequate_files))]
val_adequate = all_adequate_files[int(0.8 * len(all_adequate_files)):int(0.9 * len(all_adequate_files))]
test_adequate = all_adequate_files[int(0.9 * len(all_adequate_files)):]

train_blood = all_blood_files[:int(0.8 * len(all_blood_files))]
val_blood = all_blood_files[int(0.8 * len(all_blood_files)):int(0.9 * len(all_blood_files))]
test_blood = all_blood_files[int(0.9 * len(all_blood_files)):]

train_clot = all_clot_files[:int(0.8 * len(all_clot_files))]
val_clot = all_clot_files[int(0.8 * len(all_clot_files)):int(0.9 * len(all_clot_files))]
test_clot = all_clot_files[int(0.9 * len(all_clot_files)):]

# save the data
print("Saving data...")
print(len(train_adequate), len(val_adequate), len(test_adequate))
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