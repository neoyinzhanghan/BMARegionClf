import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision import transforms, datasets, models
import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from torchmetrics import Accuracy, AUROC

# [Previous code remains the same...]

# Define a custom dataset that applies downsampling
class DownsampledDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, downsample_factor):
        self.dataset = dataset
        self.downsample_factor = downsample_factor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.downsample_factor > 1:
            size = (512 // self.downsample_factor, 512 // self.downsample_factor)
            image = transforms.functional.resize(image, size)
        return image, label

# Data Module
class ImageDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, downsample_factor):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.downsample_factor = downsample_factor
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.61070228, 0.54225375, 0.65411311), std=(0.1485182, 0.1786308, 0.12817113))
        ])

    def setup(self, stage=None):
        # Load train, validation and test datasets
        train_dataset = datasets.ImageFolder(root=os.path.join(self.data_dir, 'train'), transform=self.transform)
        val_dataset = datasets.ImageFolder(root=os.path.join(self.data_dir, 'val'), transform=self.transform)
        test_dataset = datasets.ImageFolder(root=os.path.join(self.data_dir, 'test'), transform=self.transform)

        self.train_dataset = DownsampledDataset(train_dataset, self.downsample_factor)
        self.val_dataset = DownsampledDataset(val_dataset, self.downsample_factor)
        self.test_dataset = DownsampledDataset(test_dataset, self.downsample_factor)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=20)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=20)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=20)


# Model Module
class ResNetModel(pl.LightningModule):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        self.train_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.train_auroc = AUROC(num_classes=num_classes, task='multiclass')
        self.val_auroc = AUROC(num_classes=num_classes, task='multiclass')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        self.train_accuracy(y_hat, y)
        self.train_auroc(y_hat, y)
        self.log('train_acc', self.train_accuracy, on_step=True, on_epoch=True)
        self.log('train_auroc', self.train_auroc, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        return optimizer
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.val_accuracy(y_hat, y)
        self.val_auroc(y_hat, y)
        return loss

    def on_validation_epoch_end(self):
        self.log('val_acc_epoch', self.val_accuracy.compute())
        self.log('val_auroc_epoch', self.val_auroc.compute())
        # Handle or reset saved outputs as needed

# Main training loop
def train_model(downsample_factor):
    data_module = ImageDataModule(data_dir='/media/hdd2/neo/bma_region_clf_data_full_v2_split', batch_size=32, downsample_factor=downsample_factor)
    model = ResNetModel(num_classes=2)
    
    # Logger
    logger = TensorBoardLogger('lightning_logs', name=str(downsample_factor))

    # Trainer configuration for distributed training
    trainer = pl.Trainer(
        max_epochs=100, 
        logger=logger, 
        devices=3, 
        accelerator='gpu'  # 'ddp' for DistributedDataParallel
    )
    trainer.fit(model, data_module)

# Run training for each downsampling factor
for factor in [1,2,4,8,16]:
    train_model(factor)
