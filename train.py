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
            size = (image.size[1] // self.downsample_factor, image.size[0] // self.downsample_factor)
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
            transforms.Resize(512//downsample_factor),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def setup(self, stage=None):
        # Modify to point to your dataset
        full_dataset = datasets.ImageFolder(root=self.data_dir, transform=self.transform)
        self.dataset = DownsampledDataset(full_dataset, self.downsample_factor)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)


# Model Module
class ResNetModel(pl.LightningModule):
    def __init__(self, num_classes=3):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.train_auroc = AUROC(num_classes=num_classes)
        self.val_auroc = AUROC(num_classes=num_classes)

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

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.val_accuracy(y_hat, y)
        self.val_auroc(y_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        self.log('val_acc', self.val_accuracy, on_step=True, on_epoch=True)
        self.log('val_auroc', self.val_auroc, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        return optimizer

    def validation_epoch_end(self, outputs):
        self.log('val_acc_epoch', self.val_accuracy.compute())
        self.log('val_auroc_epoch', self.val_auroc.compute())

# Main training loop
def train_model(downsample_factor):
    data_module = ImageDataModule(data_dir='path/to/your/data', batch_size=32, downsample_factor=downsample_factor)
    model = ResNetModel(num_classes=3)
    
    # Logger
    logger = TensorBoardLogger('lightning_logs', name=str(downsample_factor))

    # Trainer configuration for distributed training
    trainer = pl.Trainer(
        max_epochs=10, 
        logger=logger, 
        devices=3, 
        num_workers=20, 
        accelerator='ddp'  # 'ddp' for DistributedDataParallel
    )
    trainer.fit(model, data_module)

# Run training for each downsampling factor
for factor in [1, 2, 4, 8, 16]:
    train_model(factor)
