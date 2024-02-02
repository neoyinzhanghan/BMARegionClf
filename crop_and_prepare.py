import os
from PIL import Image
from tqdm import tqdm

input_dir = "/Users/neo/Documents/Research/DeepHeme/LLData/bma_region_clf_data"
output_dir = "/Users/neo/Documents/Research/DeepHeme/LLData/bma_region_clf_data_cropped"

# for each folder in input_dir, each further subfolder contains three folders: "adequateX", "bloodX", "clotX"
# where X can be anything

# for each image in each of the three folders, crop the image into 4 pieces of 512x512 (there might be some overlaps between the 4 crops but its okay)
# save the cropped images into the output_dir under "adequate", "blood", "clot" folders depending on the original folder it was from


# create the output_dir if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    os.makedirs(os.path.join(output_dir, "adequate"))
    os.makedirs(os.path.join(output_dir, "blood"))
    os.makedirs(os.path.join(output_dir, "clot"))

dirs = os.listdir(input_dir)
all_subdirs = [subdir for subdir in dirs if os.path.isdir(os.path.join(input_dir, subdir))]

crop_size = 512

def crop(img_path):
    """ return 4 images, TL, TR, BL, BR"""

    img = Image.open(img_path)
    width, height = img.size

    TL = img.crop((0, 0, crop_size, crop_size))
    TR = img.crop((width - crop_size, 0, width, crop_size))
    BL = img.crop((0, height - crop_size, crop_size, height))
    BR = img.crop((width - crop_size, height - crop_size, width, height))

    return TL, TR, BL, BR

current_index = 0

all_adequate_files = []
all_blood_files = []
all_clot_files = []

# first crop and save all the adequate images
for subdir in all_subdirs:
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

# crop and save the images
for img_path in tqdm(all_adequate_files, total=len(all_adequate_files)):
    TL, TR, BL, BR = crop(img_path)
    TL.save(os.path.join(output_dir, "adequate", f"{current_index}_TL.png"))
    TR.save(os.path.join(output_dir, "adequate", f"{current_index}_TR.png"))
    BL.save(os.path.join(output_dir, "adequate", f"{current_index}_BL.png"))
    BR.save(os.path.join(output_dir, "adequate", f"{current_index}_BR.png"))
    current_index += 1


for img_path in tqdm(all_blood_files, total=len(all_blood_files)):
    TL, TR, BL, BR = crop(img_path)
    TL.save(os.path.join(output_dir, "blood", f"{current_index}_TL.png"))
    TR.save(os.path.join(output_dir, "blood", f"{current_index}_TR.png"))
    BL.save(os.path.join(output_dir, "blood", f"{current_index}_BL.png"))
    BR.save(os.path.join(output_dir, "blood", f"{current_index}_BR.png"))
    current_index += 1

for img_path in tqdm(all_clot_files, total=len(all_clot_files)):
    TL, TR, BL, BR = crop(img_path)
    TL.save(os.path.join(output_dir, "clot", f"{current_index}_TL.png"))
    TR.save(os.path.join(output_dir, "clot", f"{current_index}_TR.png"))
    BL.save(os.path.join(output_dir, "clot", f"{current_index}_BL.png"))
    BR.save(os.path.join(output_dir, "clot", f"{current_index}_BR.png"))
    current_index += 1

