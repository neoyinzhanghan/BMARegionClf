import pandas as pd
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


# class RegionClassificationDataset(Dataset):
#     def __init__(
#         self, metadata_csv_path, split, transform=None, balanced_sampling=True
#     ):
#         """
#         Args:
#             metadata_csv_path (str): Path to the CSV file containing metadata.
#             split (str): One of 'train', 'val', or 'test' to specify the dataset split.
#             balanced_sampling (bool): If True, enables balanced sampling for each class.
#         """
#         self.metadata = pd.read_csv(metadata_csv_path)

#         # Filter the data based on the split
#         self.metadata = self.metadata[self.metadata["split"] == split].reset_index(
#             drop=True
#         )

#         # Split indices for each class
#         self.adequate_indices = self.metadata[
#             self.metadata["label"] == "adequate"
#         ].index.tolist()
#         self.inadequate_indices = self.metadata[
#             self.metadata["label"] == "inadequate"
#         ].index.tolist()

#         # Check if balanced sampling is enabled
#         self.balanced_sampling = balanced_sampling

#     def __len__(self):
#         # Return the total length as the maximum number of samples available for balanced sampling
#         if self.balanced_sampling:
#             return min(len(self.adequate_indices), len(self.inadequate_indices)) * 2
#         else:
#             return len(self.metadata)

#     def __getitem__(self, idx):
#         if self.balanced_sampling:
#             # Alternate between classes for balanced sampling
#             if idx % 2 == 0:
#                 sample_idx = random.choice(self.adequate_indices)
#             else:
#                 sample_idx = random.choice(self.inadequate_indices)
#         else:
#             sample_idx = idx

#         # Get the image path and label from the sampled index
#         img_path = self.metadata.loc[sample_idx, "image_path"]
#         label = self.metadata.loc[sample_idx, "label"]

#         # Open the image
#         image = Image.open(img_path).convert("RGB")

#         # Convert the PIL image to a tensor
#         image = ToTensor()(image)

#         # Map label to an integer class (adequate -> 0, inadequate -> 1)
#         label = 0 if label == "adequate" else 1

#         return image, label


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
