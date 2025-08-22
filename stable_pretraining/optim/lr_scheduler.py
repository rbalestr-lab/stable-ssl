import math

import numpy as np
import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    LinearLR,
    MultiStepLR,
    SequentialLR,
    _LRScheduler,
)

# Default parameter factories for common schedulers (both torch and custom)
# These callables receive the calling module (for trainer context) and optimizer
DEFAULT_SCHEDULER_FACTORIES = {
    # torch schedulers
    "CosineAnnealingLR": lambda module, opt: {
        "T_max": getattr(module.trainer, "estimated_stepping_batches", None),
    },
    "OneCycleLR": lambda module, opt: {
        "max_lr": opt.param_groups[0]["lr"],
        "total_steps": getattr(module.trainer, "estimated_stepping_batches", None),
        "pct_start": min(10 / getattr(module.trainer, "max_epochs", 1), 0.01),
    },
    "StepLR": lambda module, opt: {"step_size": 30, "gamma": 0.1},
    "ExponentialLR": lambda module, opt: {"gamma": 0.9},
    "ReduceLROnPlateau": lambda module, opt: {
        "mode": "min",
        "patience": 10,
        "factor": 0.1,
    },
    "LinearLR": lambda module, opt: {},
    "ConstantLR": lambda module, opt: {},
    # custom schedulers (defined below)
    "LinearWarmup": lambda module, opt: {
        "total_steps": getattr(module.trainer, "estimated_stepping_batches", None),
        "start_factor": 0.01,
        "peak_step": max(
            1, int(0.01 * getattr(module.trainer, "estimated_stepping_batches", 1))
        ),
    },
    "LinearWarmupCosineAnnealing": lambda module, opt: {
        "total_steps": getattr(module.trainer, "estimated_stepping_batches", None),
        "start_factor": 0.01,
        "end_lr": 0.0,
        "peak_step": max(
            1, int(0.01 * getattr(module.trainer, "estimated_stepping_batches", 1))
        ),
    },
    "LinearWarmupCyclicAnnealing": lambda module, opt: {
        "total_steps": getattr(module.trainer, "estimated_stepping_batches", None),
        "start_factor": 0.01,
        "peak_step": max(
            1, int(0.1 * getattr(module.trainer, "estimated_stepping_batches", 1))
        ),
    },
    "LinearWarmupThreeStepsAnnealing": lambda module, opt: {
        "total_steps": getattr(module.trainer, "estimated_stepping_batches", None),
        "start_factor": 0.001,
        "gamma": 0.3,
        "peak_step": max(
            1, int(0.05 * getattr(module.trainer, "estimated_stepping_batches", 1))
        ),
    },
    "LinearWarmupCosineAnnealingLR": lambda module, opt: {
        "warmup_steps": max(
            1, int(0.01 * getattr(module.trainer, "estimated_stepping_batches", 1))
        ),
        "max_steps": getattr(module.trainer, "estimated_stepping_batches", None),
        "warmup_start_lr": 0.0,
        "eta_min": 0.0,
    },
}


def _resolve_scheduler_callable(name_or_class):
    """Resolve a scheduler by name from torch or this module.

    Accepts a string name, class, or callable. Returns a callable to construct the scheduler.
    """
    if not isinstance(name_or_class, str):
        return name_or_class

    # Try torch.optim.lr_scheduler first
    if hasattr(torch.optim.lr_scheduler, name_or_class):
        return getattr(torch.optim.lr_scheduler, name_or_class)

    # Then try this module (custom functions/classes)
    if name_or_class in globals():
        return globals()[name_or_class]

    raise ValueError(
        f"Scheduler '{name_or_class}' not found in torch.optim.lr_scheduler or stable_pretraining.optim.lr_scheduler."
    )


def _build_default_params(name: str, module, optimizer):
    factory = DEFAULT_SCHEDULER_FACTORIES.get(name)
    if factory is None:
        return {}
    params = factory(module, optimizer)
    # Remove None in case trainer context is missing
    return {k: v for k, v in params.items() if v is not None}


def create_scheduler(optimizer, scheduler_config, module=None):
    """Create a learning rate scheduler instance from a flexible config.

    Args:
        optimizer: torch optimizer
        scheduler_config: str | dict | partial | Class | callable
        module: optional calling module for defaults (expects module.trainer)

    Returns:
        Instantiated scheduler
    """
    # partial -> call directly
    if isinstance(scheduler_config, type(lambda: None)) and hasattr(
        scheduler_config, "func"
    ):
        # It's a functools.partial (duck-typing), call with optimizer
        return scheduler_config(optimizer)

    # dict -> pop type + params
    if isinstance(scheduler_config, dict):
        cfg = dict(scheduler_config)
        scheduler_type = cfg.pop("type", "CosineAnnealingLR")
        params = cfg
    else:
        scheduler_type = scheduler_config
        params = {}

    scheduler_ctor = _resolve_scheduler_callable(scheduler_type)

    # If no params provided, use smart defaults if known
    if not params:
        name = (
            scheduler_ctor.__name__
            if hasattr(scheduler_ctor, "__name__")
            else str(scheduler_ctor)
        )
        try:
            params = _build_default_params(name, module, optimizer)
        except Exception:
            params = {}

    # Instantiate. Works for both torch classes and our function factories.
    return scheduler_ctor(optimizer, **params)


class CosineDecayer:
    """Apply cosine decay with multiple cycles for learning rate scheduling.

    This class implements a cosine decay function with multiple cycles that can be used
    as a learning rate scheduler. The decay follows a cosine curve with additional
    cyclic variations.

    Args:
        total_steps (int): Total number of training steps.
        n_cycles (int, optional): Number of cycles in the cosine decay. Defaults to 3.
        gamma (float, optional): Gamma parameter for cycle amplitude. Defaults to 0.2.

    Example:
        >>> decayer = CosineDecayer(total_steps=1000, n_cycles=3)
        >>> lr_factor = decayer(step=500)
    """

    def __init__(self, total_steps, n_cycles=3, gamma=0.2):
        self.total_steps = total_steps
        self.n_cycles = n_cycles

    def __call__(self, step):
        """Compute the learning rate factor for the given step.

        Args:
            step (int): Current training step.

        Returns:
            float: Learning rate multiplier factor.
        """
        alpha = 1 - step / self.total_steps
        cycle = 1 + np.sin(self.n_cycles * 2 * np.pi * step / self.total_steps) / 2
        return alpha * cycle


def LinearWarmup(optimizer, total_steps, start_factor=0.01, peak_step=0.1):
    """Create a linear warmup learning rate scheduler.

    This function creates a linear warmup scheduler that gradually increases the
    learning rate from a small value to the full learning rate over a specified
    number of steps.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to schedule.
        total_steps (int): Total number of training steps.
        start_factor (float, optional): Initial learning rate factor. Defaults to 0.01.
        peak_step (float, optional): Step at which warmup peaks (as fraction of total_steps).
                                   Defaults to 0.1.

    Returns:
        torch.optim.lr_scheduler.LinearLR: Linear warmup scheduler.

    Example:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        >>> scheduler = LinearWarmup(optimizer, total_steps=1000, start_factor=0.01)
    """
    if peak_step < 1:
        peak_step = int(peak_step * total_steps)
    warmup = LinearLR(optimizer, start_factor, total_iters=peak_step)
    return warmup


def LinearWarmupCosineAnnealing(
    optimizer, total_steps, start_factor=0.01, end_lr=0.0, peak_step=0.01
):
    """Combine linear warmup with cosine annealing decay.

    This function creates a scheduler that first linearly warms up the learning rate,
    then applies cosine annealing decay. This is commonly used in self-supervised
    learning to achieve better convergence.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to schedule.
        total_steps (int): Total number of training steps.
        start_factor (float, optional): Initial learning rate factor for warmup. Defaults to 0.01.
        end_lr (float, optional): Final learning rate after annealing. Defaults to 0.0.
        peak_step (float, optional): Step at which warmup ends (as fraction of total_steps).
                                   Defaults to 0.01.

    Returns:
        torch.optim.lr_scheduler.SequentialLR: Combined warmup and annealing scheduler.

    Example:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        >>> scheduler = LinearWarmupCosineAnnealing(optimizer, total_steps=1000)
    """
    if peak_step < 1:
        peak_step = int(peak_step * total_steps)
    warmup = LinearLR(optimizer, start_factor, total_iters=peak_step)
    anneal = CosineAnnealingLR(optimizer, T_max=total_steps - peak_step, eta_min=end_lr)
    scheduler = SequentialLR(
        optimizer,
        [warmup, anneal],
        milestones=[peak_step],
    )
    return scheduler


def LinearWarmupCyclicAnnealing(
    optimizer, total_steps, start_factor=0.01, peak_step=0.1
):
    """Combine linear warmup with cyclic cosine annealing.

    This function creates a scheduler that combines linear warmup with cyclic cosine
    annealing. The cyclic annealing provides multiple learning rate cycles which can
    help escape local minima during training.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to schedule.
        total_steps (int): Total number of training steps.
        start_factor (float, optional): Initial learning rate factor for warmup. Defaults to 0.01.
        peak_step (float, optional): Step at which warmup ends (as fraction of total_steps).
                                   Defaults to 0.1.

    Returns:
        torch.optim.lr_scheduler.SequentialLR: Combined warmup and cyclic annealing scheduler.

    Example:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        >>> scheduler = LinearWarmupCyclicAnnealing(optimizer, total_steps=1000)
    """
    if peak_step < 1:
        peak_step = int(peak_step * total_steps)

    warmup = LinearLR(optimizer, start_factor, total_iters=peak_step)
    decay = LambdaLR(optimizer, CosineDecayer(total_steps - peak_step))
    scheduler = SequentialLR(
        optimizer,
        [warmup, decay],
        milestones=[peak_step],
    )
    return scheduler


def LinearWarmupThreeStepsAnnealing(
    optimizer, total_steps, start_factor=0.001, gamma=0.3, peak_step=0.05
):
    """Combine linear warmup with a three-step learning rate annealing.

    This function creates a scheduler that combines linear warmup with a three-step
    annealing schedule. The annealing reduces the learning rate at three predefined
    milestones, which can help with fine-tuning and convergence.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to schedule.
        total_steps (int): Total number of training steps.
        start_factor (float, optional): Initial learning rate factor for warmup. Defaults to 0.001.
        gamma (float, optional): Multiplicative factor for learning rate reduction. Defaults to 0.3.
        peak_step (float, optional): Step at which warmup ends (as fraction of total_steps).
                                   Defaults to 0.05.

    Returns:
        torch.optim.lr_scheduler.SequentialLR: Combined warmup and three-step annealing scheduler.

    Example:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        >>> scheduler = LinearWarmupThreeStepsAnnealing(optimizer, total_steps=1000)
    """
    if peak_step < 1:
        peak_step = int(peak_step * total_steps)
    warmup = LinearLR(optimizer, start_factor, total_iters=peak_step)
    anneal = MultiStepLR(
        optimizer,
        milestones=[
            (total_steps - peak_step) * 0.4,
            (total_steps - peak_step) * 0.6,
            (total_steps - peak_step) * 0.8,
        ],
        gamma=gamma,
    )
    scheduler = SequentialLR(
        optimizer,
        [warmup, anneal],
        milestones=[peak_step],
    )
    return scheduler


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """Learning rate scheduler with linear warmup followed by cosine annealing.

    This scheduler implements a custom learning rate schedule that combines linear
    warmup with cosine annealing. It provides more control over the warmup and
    annealing phases compared to the factory function approach.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to schedule.
        warmup_steps (int): Number of steps for linear warmup.
        max_steps (int): Total number of training steps.
        warmup_start_lr (float, optional): Starting learning rate for warmup. Defaults to 0.0.
        eta_min (float, optional): Minimum learning rate after annealing. Defaults to 0.0.
        last_epoch (int, optional): The index of last epoch. Defaults to -1.

    Example:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        >>> scheduler = LinearWarmupCosineAnnealingLR(
        ...     optimizer, warmup_steps=100, max_steps=1000
        ... )
    """

    def __init__(
        self,
        optimizer,
        warmup_steps,
        max_steps,
        warmup_start_lr=0.0,
        eta_min=0.0,
        last_epoch=-1,
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute the learning rate for the current epoch.

        Returns:
            list: List of learning rates for each parameter group.
        """
        if self.last_epoch < self.warmup_steps:
            return [
                (
                    self.warmup_start_lr
                    + (base_lr - self.warmup_start_lr)
                    * self.last_epoch
                    / self.warmup_steps
                )
                for base_lr in self.base_lrs
            ]
        else:
            return [
                self.eta_min
                + (base_lr - self.eta_min)
                * (
                    1
                    + math.cos(
                        math.pi
                        * (self.last_epoch - self.warmup_steps)
                        / (self.max_steps - self.warmup_steps)
                    )
                )
                / 2
                for base_lr in self.base_lrs
            ]
