from typing import Optional, Literal, Union
from dataclasses import dataclass

from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC, \
    MulticlassF1Score, MulticlassJaccardIndex, \
    BinaryAccuracy, BinaryAUROC, BinaryF1Score, JaccardIndex, \
    BinaryConfusionMatrix, MulticlassConfusionMatrix

from torchmetrics import MeanSquaredError, MeanAbsoluteError, SpearmanCorrCoef
from torchmetrics import Metric, MetricCollection

# BEWARE there is a bug!!
# https://github.com/Lightning-AI/torchmetrics/issues/1604
# https://github.com/Lightning-AI/torchmetrics/pull/1676#issuecomment-1925364782
# TODO: how do we handle logging things like BinaryConfusionMatrix that output a tensor, not a dict

@dataclass
class BinaryClfMetricsGConfig():

    def get_metrics(self, datainfo) -> Union[Metric, MetricCollection]:

        threshold = 0.5
        return MetricCollection({
            'accuracy': BinaryAccuracy(threshold=threshold),
            'auc': BinaryAUROC(),
            'f1': BinaryF1Score(threshold=threshold),
            # 'confusion_mat': BinaryConfusionMatrix(threshold=threshold,
            #                                        normalize=None)
        })
