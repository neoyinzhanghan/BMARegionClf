import torch
from torch import nn
from torchvision import models
from dataclasses import dataclass

@dataclass # This is a dataclass that represents the model
class ResNetModelConfig:
    """ Get a config for a ResNet model.

    Parameters:
    -------------
    num_classes: int
        The number of classes in the dataset.
    model_name: str
        The name of the model to use.
    pretrained: bool
        Whether to use a pretrained model.
    
    """
    num_classes: int = 2
    model_name: str = "resnet50"
    pretrained: bool = True
    
    def get_model(self):
        return models.resnet50(pretrained=self.pretrained, num_classes=self.num_classes)