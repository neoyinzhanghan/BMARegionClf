from typing import Tuple, Literal
from dataclasses import dataclass

from torch.optim import Adam, SGD, AdamW
from torch.optim import Optimizer

@dataclass
class OptimizerGConfig():
    """
    Configures and gets an optimizer.

    Parameters
    ----------
    name:

    lr:

    momentum:

    dampening:

    nesterov:

    betas:

    weight_decay:
    """

    name: Literal['sgd', 'adam', 'adamw'] = 'sgd'
    lr: float = 0.001
    momentum: float = 0
    dampening: float = 0
    nesterov: bool = False
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0

    def get_optimizer(self, params) -> Optimizer:
        """
        Setups up the optimizer.

        Parameters
        ----------
        params

        Ouput
        -----
        optimizer
        """

        name = self.name.lower()
        assert name in ['sgd', 'adam', 'adamw']

        if name == 'adam':
            return Adam(params=params,
                        lr=self.lr,
                        betas=self.betas,
                        weight_decay=self.weight_decay)

        if name == 'adamw':
            return AdamW(params=params,
                         lr=self.lr,
                         betas=self.betas,
                         weight_decay=self.weight_decay)

        elif name == 'sgd':
            return SGD(params=params,
                       lr=self.lr,
                       momentum=self.momentum,
                       dampening=self.dampening,
                       nesterov=self.nesterov,
                       weight_decay=self.weight_decay)