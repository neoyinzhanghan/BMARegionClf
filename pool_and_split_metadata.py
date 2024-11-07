import os
import random
import pandas as pd

adequate_folders = []
inadequate_folders = []

adequate_images = []
inadequate_images = []

train_probability = 0.8
val_probability = 0.1
test_probability = 0.1

pooled_and_split_metadata = {
    "image_path": [],
    "label": [],
}

for adequate_folder in adequate_folders:
    # get all the png and jpg images in the folder
    images = [
        f
        for f in os.listdir(adequate_folder)
        if f.endswith(".png") or f.endswith(".jpg")
    ]

    image_paths = [os.path.join(adequate_folder, image) for image in images]
    labels = ["adequate" for _ in images]

    pooled_and_split_metadata["image_path"].extend(image_paths)
    pooled_and_split_metadata["label"].extend(labels)

for inadequate_folder in inadequate_folders:
    # get all the png and jpg images in the folder
    images = [
        f
        for f in os.listdir(inadequate_folder)
        if f.endswith(".png") or f.endswith(".jpg")
    ]

    image_paths = [os.path.join(inadequate_folder, image) for image in images]
    labels = ["inadequate" for _ in images]

    pooled_and_split_metadata["image_path"].extend(image_paths)
    pooled_and_split_metadata["label"].extend(labels)

# now create pandas dataframe
pooled_and_split_metadata_df = pd.DataFrame(pooled_and_split_metadata)


# now add a new column named split
split = []
for i in range(len(pooled_and_split_metadata_df)):
    rand = random.random()
    if rand < train_probability:
        split.append("train")
    elif rand < train_probability + val_probability:
        split.append("val")
    else:
        split.append("test")

pooled_and_split_metadata_df["split"] = split

# save the dataframe to a csv file named split_region_clf_v3_metadata.csv in the working directory
pooled_and_split_metadata_df.to_csv("split_region_clf_v3_metadata.csv", index=False)
