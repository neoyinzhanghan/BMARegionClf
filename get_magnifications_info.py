bma_dir = "/media/hdd1/BMAs"
save_dir = "/media/hdd1"

import os

# traverse through all the ndpi files in the BMA directory
# save the magnification of every single levels of each BMA file in a pandas dataframe and save it as a csv file named magnification.csv
# use the pyvips library to read the ndpi files
import pyvips

import pandas as pd

# first figure out the maximum number of levels in all the BMA files
max_levels = 0

for root, dirs, files in os.walk(bma_dir):
    for file in files:
        if file.endswith(".ndpi"):
            filename = os.path.join(root, file)
            bma = pyvips.Image.new_from_file(filename, level=0)
            levels = bma.get("openslide.level-count")
            if levels > max_levels:
                max_levels = levels

# create the columns for the dataframe
columns = ["filename"]
for i in range(max_levels):
    columns.append(f"level_{i}_magnification")

# create the dataframe
df = pd.DataFrame(columns=columns)

# fill the dataframe
for root, dirs, files in os.walk(bma_dir):
    for file in files:
        if file.endswith(".ndpi"):
            filename = os.path.join(root, file)
            bma = pyvips.Image.new_from_file(filename, level=0)
            levels = bma.get("openslide.level-count")
            magnifications = [bma.get("openslide.level-mag", i) for i in range(levels)]
            magnifications.insert(0, filename)
            df = df.append(pd.Series(magnifications, index=df.columns), ignore_index=True)

# save the dataframe as a csv file
df.to_csv(os.path.join(save_dir, "magnification.csv"), index=False)