"""Shared utilities for optimizer and scheduler configuration."""

import inspect
from functools import partial
from typing import Any, Union

import torch
from hydra.utils import instantiate

from .. import optim as ssl_optim
from ..optim import lr_scheduler as ssl_lr


def create_optimizer(
    params,
    optimizer_config: Union[str, dict, partial, type],
) -> torch.optim.Optimizer:
    """Create an optimizer from flexible configuration.

    This function provides a unified way to create optimizers from various configuration formats,
    used by both Module and OnlineProbe for consistency.

    Args:
        params: Parameters to optimize (e.g., model.parameters())
        optimizer_config: Can be:
            - str: optimizer name from torch.optim or stable_pretraining.optim (e.g., "AdamW", "LARS")
            - dict: {"type": "AdamW", "lr": 1e-3, ...}
            - partial: pre-configured optimizer factory
            - class: optimizer class (e.g., torch.optim.AdamW)

    Returns:
        Configured optimizer instance

    Examples:
        >>> # String name (uses default parameters)
        >>> opt = create_optimizer(model.parameters(), "AdamW")

        >>> # Dict with parameters
        >>> opt = create_optimizer(
        ...     model.parameters(), {"type": "SGD", "lr": 0.1, "momentum": 0.9}
        ... )

        >>> # Using partial
        >>> from functools import partial
        >>> opt = create_optimizer(
        ...     model.parameters(), partial(torch.optim.Adam, lr=1e-3)
        ... )

        >>> # Direct class
        >>> opt = create_optimizer(model.parameters(), torch.optim.RMSprop)
    """
    # Handle Hydra config objects
    if hasattr(optimizer_config, "_target_"):
        return instantiate(optimizer_config, params=params, _convert_="object")

    # partial -> call with params
    if isinstance(optimizer_config, partial):
        return optimizer_config(params)

    # callable (including optimizer factories, but not classes)
    if callable(optimizer_config) and not isinstance(optimizer_config, type):
        return optimizer_config(params)

    # dict -> extract type and kwargs
    if isinstance(optimizer_config, dict):
        config_copy = optimizer_config.copy()
        opt_type = config_copy.pop("type", "AdamW")
        kwargs = config_copy
    else:
        opt_type = optimizer_config
        kwargs = {}

    # resolve class
    if isinstance(opt_type, str):
        if hasattr(torch.optim, opt_type):
            opt_class = getattr(torch.optim, opt_type)
        elif hasattr(ssl_optim, opt_type):
            opt_class = getattr(ssl_optim, opt_type)
        else:
            torch_opts = [n for n in dir(torch.optim) if n[0].isupper()]
            ssl_opts = [n for n in dir(ssl_optim) if n[0].isupper()]
            raise ValueError(
                f"Optimizer '{opt_type}' not found. Available in torch.optim: "
                + ", ".join(torch_opts)
                + ". Available in stable_pretraining.optim: "
                + ", ".join(ssl_opts)
            )
    else:
        opt_class = opt_type

    try:
        return opt_class(params, **kwargs)
    except TypeError as e:
        sig = inspect.signature(opt_class.__init__)
        required = [
            p.name
            for p in sig.parameters.values()
            if p.default == inspect.Parameter.empty and p.name not in ["self", "params"]
        ]
        raise TypeError(
            f"Failed to create {opt_class.__name__}. Required parameters: {required}. "
            f"Provided: {list(kwargs.keys())}. Original error: {e}"
        )


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_config: Union[str, dict, partial, type],
    module: Any = None,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Create a learning rate scheduler with flexible configuration.

    This function provides a unified way to create schedulers from various configuration formats,
    used by both Module and OnlineProbe for consistency.

    Args:
        optimizer: The optimizer to attach the scheduler to
        scheduler_config: Can be:
            - str: Name of scheduler (e.g., "CosineAnnealingLR")
            - dict: {"type": "CosineAnnealingLR", "T_max": 1000, ...}
            - partial: Pre-configured scheduler (e.g., partial(CosineAnnealingLR, T_max=1000))
            - class: Direct scheduler class (will use smart defaults)
        module: Optional module instance for accessing trainer properties (for smart defaults)

    Returns:
        Configured scheduler instance

    Examples:
        >>> # Simple string (uses smart defaults)
        >>> scheduler = create_scheduler(opt, "CosineAnnealingLR")

        >>> # With custom parameters
        >>> scheduler = create_scheduler(
        ...     opt, {"type": "StepLR", "step_size": 30, "gamma": 0.1}
        ... )

        >>> # Using partial for full control
        >>> from functools import partial
        >>> scheduler = create_scheduler(
        ...     opt, partial(torch.optim.lr_scheduler.ExponentialLR, gamma=0.95)
        ... )
    """
    # Handle Hydra config objects
    if hasattr(scheduler_config, "_target_"):
        return instantiate(scheduler_config, optimizer=optimizer, _convert_="object")

    # Delegate to central factory in stable_pretraining.optim.lr_scheduler
    return ssl_lr.create_scheduler(optimizer, scheduler_config, module=module)
