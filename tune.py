####################################################################################################
# Standard imports
####################################################################################################
import os
import tempfile
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import albumentations as A
import numpy as np
from ray.train import Checkpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from torch import nn
from torchmetrics import Accuracy, AUROC


####################################################################################################
# RAY related imports
####################################################################################################
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer

####################################################################################################
# Constants
####################################################################################################

default_config = {"lr": 0.001}


def get_feat_extract_augmentation_pipeline(image_size):
    """Returns a randomly chosen augmentation pipeline for SSL."""

    ## Simple augumentation to improtve the data generalibility
    transform_shape = A.Compose(
        [
            A.ShiftScaleRotate(p=0.8),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(shear=(-10, 10), p=0.3),
            A.ISONoise(
                color_shift=(0.01, 0.02),
                intensity=(0.05, 0.01),
                always_apply=False,
                p=0.2,
            ),
        ]
    )
    transform_color = A.Compose(
        [
            A.RandomBrightnessContrast(
                contrast_limit=0.4, brightness_by_max=0.4, p=0.5
            ),
            A.CLAHE(p=0.3),
            A.ColorJitter(p=0.2),
            A.RandomGamma(p=0.2),
        ]
    )

    # compose the two augmentation pipelines
    return A.Compose(
        [A.Resize(image_size, image_size), A.OneOf([transform_shape, transform_color])]
    )


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

            # Apply augmentation
            image = get_feat_extract_augmentation_pipeline(
                image_size=512 // self.downsample_factor
            )(image=np.array(image))["image"]

        return image, label


# Data Module
class ImageDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, downsample_factor):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.downsample_factor = downsample_factor
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize(mean=(0.61070228, 0.54225375, 0.65411311), std=(0.1485182, 0.1786308, 0.12817113))
            ]
        )

    def setup(self, stage=None):
        # Load train, validation and test datasets
        train_dataset = datasets.ImageFolder(
            root=os.path.join(self.data_dir, "train"), transform=self.transform
        )
        val_dataset = datasets.ImageFolder(
            root=os.path.join(self.data_dir, "val"), transform=self.transform
        )
        test_dataset = datasets.ImageFolder(
            root=os.path.join(self.data_dir, "test"), transform=self.transform
        )

        self.train_dataset = DownsampledDataset(train_dataset, self.downsample_factor)
        self.val_dataset = DownsampledDataset(val_dataset, self.downsample_factor)
        self.test_dataset = DownsampledDataset(test_dataset, self.downsample_factor)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=20
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=20
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=20
        )


# Model Module
class ResNetModel(pl.LightningModule):
    def __init__(self, num_classes=2, config=default_config):
        super().__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        assert num_classes >= 2

        if num_classes == 2:
            task = "binary"
        elif num_classes > 2:
            task = "multiclass"

        task = "multiclass"

        self.train_accuracy = Accuracy(task=task, num_classes=num_classes)
        self.val_accuracy = Accuracy(task=task, num_classes=num_classes)
        self.train_auroc = AUROC(num_classes=num_classes, task=task)
        self.val_auroc = AUROC(num_classes=num_classes, task=task)

        self.config = config

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        self.train_accuracy(y_hat, y)
        self.train_auroc(y_hat, y)
        self.log("train_acc", self.train_accuracy, on_step=True, on_epoch=True)
        self.log("train_auroc", self.train_auroc, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])
        return optimizer

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.val_accuracy(y_hat, y)
        self.val_auroc(y_hat, y)
        return loss

    def on_validation_epoch_end(self):
        self.log("val_acc_epoch", self.val_accuracy.compute())
        self.log("val_auroc_epoch", self.val_auroc.compute())
        # Handle or reset saved outputs as needed


####################################################################################################
# RAY Tune
####################################################################################################

num_epochs = 5
num_samples = 10
scheduler = ASHAScheduler(
    max_t=num_epochs,
    grace_period=1,
    reduction_factor=2,
)

fixed_config = {"downsample_factor": 1, "num_classes": 2, "batch_size": 32}
search_space = {
    "lr": tune.sample_from(lambda spec: 10 ** (-10 * np.random.rand())),
}
scaling_config = ScalingConfig(
    num_workers=1,
    use_gpu=True,
    resources_per_worker={"CPU": 27, "GPU": 3},
)
run_config = RunConfig(
    checkpoint_config=CheckpointConfig(
        num_to_keep=2,
        checkpoint_score_attribute="ptl/val_accuarcy",
        checkpoint_score_order="max",
    )
)


def train_func(config):
    dm = ImageDataModule(
        data_dir="/media/hdd2/neo/bma_region_clf_data_full_v2_split",
        batch_size=fixed_config["batch_size"],
        downsample_factor=fixed_config['downsample_factor'],
    )
    model = ResNetModel(num_classes=fixed_config["num_classes"])

    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        strategy=RayDDPStrategy(),
        callbacks=[RayTrainReportCallback()],
        plugins=[RayLightningEnvironment()],
        enable_progress_bar=False,
    )

    trainer = prepare_trainer(trainer)
    trainer.fit(model, datamodule=dm)


ray_trainer = TorchTrainer(
    train_func,
    scaling_config=scaling_config,
    run_config=run_config,
)


def tune_clf_asha(num_samples=10):
    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2,
    )

    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": search_space},
        tune_config=tune.TuneConfig(
            metric="val_acc_epoch",
            mode="max",
            num_samples=num_samples,
            scheduler=scheduler,
        ),
    )
    return tuner.fit()


if __name__ == "__main__":

    results = tune_clf_asha(num_samples=num_samples)

    results.get_best_result(metric="val_acc_epoch", mode="max")
