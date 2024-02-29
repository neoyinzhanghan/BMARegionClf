from typing import Optional, Union, Literal
from dataclasses import dataclass, field

from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, \
    SequentialLR, StepLR, ReduceLROnPlateau, ExponentialLR
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


@dataclass
class LrSchedDetailsConfig:
    interval: Literal['epoch', 'step'] = 'epoch'
    frequency: int = 1
    monitor: Optional[str] = None   # 'val/loss'


@dataclass
class LrSchedulerGConfig():
    """
    Configures and gets the learning rate scheduler, possibly with a linear warmup period.

    Parameters
    ----------
    name:

    gamma:

    step_size:

    mode:

    factor:

    patience:

    threshold:

    threshold_mode:

    cooldown:

    min_lr:

    max_epochs:

    use_warmup:

    warmup_epochs:

    details
    """
    name: Optional[str] = None

    # step
    gamma: float = 0.1  # also exponential
    step_size: int = 1

    # reduce
    mode: str = 'min'
    factor: float = 0.1
    patience: int = 10
    threshold: float = 0.0001
    threshold_mode: str = 'rel'
    cooldown: int = 0
    min_lr: float = 0

    # cosine
    max_epochs: Optional[int] = None

    # warmup
    use_warmup: bool = False
    warmup_epochs: int = 10

    # details
    # TODO: factory
    details: LrSchedDetailsConfig = field(default_factory=LrSchedDetailsConfig)

    def get_lr_scheduler(self,
                         optimizer: Optimizer
                         ) -> Union[None, _LRScheduler]:
        """
        Sets up the learning rate scheduler.

        Parameters
        ----------
        optimizer:

        Ouput
        -----
        ls_scheduler or None
        """
        assert self.name is None or self.name in \
            ['step', 'reduce_on_plateau', 'cosine_decay']

        if self.name is None or self.name == 'flat':

            if self.use_warmup:
                # constant learning rate
                # TODO: probably just write our own constant LR since this
                # is a bit weird
                scheduler = StepLR(optimizer=optimizer, step_size=1, gamma=1)
            else:
                scheduler = None

        elif self.name == 'step':
            scheduler = StepLR(optimizer=optimizer,
                               step_size=self.step_size,
                               gamma=self.gamma)

        elif self.name == 'exponential':
            scheduler = ExponentialLR(optimizer=optimizer,
                                      gamma=self.gamm)

        elif self.name == 'reduce_on_plateau':
            scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                          mode=self.mode,
                                          factor=self.factor,
                                          patience=self.patience,
                                          threshold=self.threshold,
                                          threshold_mode=self.threshold_mode,
                                          cooldown=self.cooldown,
                                          min_lr=self.min_lr,
                                          eps=1e-08)

        elif self.name == 'cosine_decay':
            assert self.max_epochs is not None

            if self.use_warmup:
                T_max = self.max_epochs - self.warmup_epochs
            else:
                T_max = self.max_epochs
            scheduler = CosineAnnealingLR(T_max=T_max)

        ##########
        # Warmup #
        ##########
        if self.use_warmup:
            scheduler = add_linear_warmup(scheduler=scheduler,
                                          warmup_epochs=self.warmup_epochs)

        return scheduler


def add_linear_warmup(scheduler,
                      warmup_epochs: int
                      ) -> _LRScheduler:
    """
    Adds an initial linear warmup period to a learning rate scheduler.

    Parameters
    ----------
    scheduler:

    warmup_epochs:

    Output
    ------
    scheduler_with_warmup
    """
    # TODO: start factor = 0?
    warmup_scheduler = LinearLR(optimizer=scheduler.optimizer,
                                start_factor=1e-8,
                                end_factor=1,
                                total_iters=warmup_epochs)

    return SequentialLR(optimizer=scheduler.optimizer,
                        schedulers=[warmup_scheduler, scheduler],
                        milestones=[warmup_epochs]
                        )