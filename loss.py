from typing import Optional, Literal, Sequence
from dataclasses import dataclass

import torch.nn as nn
# from dlt.CtsCIndexLoss import CtsCIndexLoss


# TODO: should class weights go in get loss?
@dataclass
class ClassificationLossGConfig():
    """
    Configures and gets a classification loss.

    Parameters
    ----------
    name:

    class_weights:

    reduction:
    """

    name: str = 'cross_entropy'

    class_weights: Optional[Sequence[float]] = None

    reduction: Optional[Literal['mean', 'sum', 'none']] = 'mean'

    def get_loss(self) -> nn.Module:
        """
        Returns the instantiated loss function.
        """

        # Note both CE and BCE expect logists
        # TODO: binary vs. multiclass?
        # TODO: @Iain get better understanding of Focal/Jaccard params
        # TODO: for segmentaiton do we want to allow ignore index?
        # https://github.com/microsoft/torchgeo/blob/main/torchgeo/trainers/segmentation.py
        # also whats up with the -1000

        if self.name in ['focal', 'jaccard']:
            try:
                import segmentation_models_pytorch as smp
            except ImportError as e:
                print("segmentation_models_pytorch has not been installed; please install it!")
                raise e

        if self.name == 'cross_entropy':
            return nn.CrossEntropyLoss(weight=self.class_weights,
                                       reduction=self.reduction)

        elif self.name == 'binary_cross_entropy':
            return nn.BCEWithLogitsLoss(weight=self.class_weights,
                                        reduction=self.reduction)

        elif self.name == 'focal':
            return smp.FocalLoss(mode="multiclass",
                                 alpha=None,
                                 gamma=2,
                                 normalized=True,
                                 reduction=self.reduction)

        elif self.name == 'jaccard':
            if self.reduction == 'sum':
                raise NotImplementedError(
                    "segmentation_models_pytorch JaccardLoss "
                    "does not allow sum. Perhaps this is an oversight?")

            return smp.JaccardLoss(mode='multiclass',
                                   classes=None,
                                   log_loss=False,
                                   from_logits=True,
                                   smooth=0,
                                   eps=1e-7)