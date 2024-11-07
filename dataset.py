import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

# import transform to tensor
from torchvision.transforms import ToTensor
from PIL import Image


class RegionClassificationDataset(Dataset):
    def __init__(self, metadata_csv_path, split, transform=None):
        """
        Args:
            metadata_csv_path (str): Path to the CSV file containing metadata.
            split (str): One of 'train', 'val', or 'test' to specify the dataset split.
        """
        self.metadata = pd.read_csv(metadata_csv_path)

        # Filter the data based on the split
        self.metadata = self.metadata[self.metadata["split"] == split].reset_index(
            drop=True
        )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the image path and label
        img_path = self.metadata.loc[idx, "image_path"]
        label = self.metadata.loc[idx, "label"]

        # Open the image
        image = Image.open(img_path).convert("RGB")

        # convert the pil image to a tensor
        image = ToTensor()(image)

        # Map label to an integer class (adequate -> 0, inadequate -> 1)
        label = 0 if label == "adequate" else 1

        return image, label
