from typing import Optional

import numpy as np
from lightning.pytorch import Callback, LightningModule, Trainer
from loguru import logger as logging
from sklearn.base import ClassifierMixin, RegressorMixin
from tabulate import tabulate


class SklearnCheckpoint(Callback):
    """Callback for saving and loading sklearn models in PyTorch Lightning checkpoints.

    This callback automatically detects sklearn models (Regressors and Classifiers)
    attached to the Lightning module and handles their serialization/deserialization
    during checkpoint save/load operations. This is necessary because sklearn models
    are not natively supported by PyTorch's checkpoint system.

    The callback will:
    1. Automatically discover sklearn models attached to the Lightning module
    2. Save them to the checkpoint dictionary during checkpoint saving
    3. Restore them from the checkpoint during checkpoint loading
    4. Log information about discovered sklearn modules during setup

    Note:
        - Only attributes that are sklearn RegressorMixin or ClassifierMixin instances are saved
        - Private attributes (starting with '_') are ignored
        - The callback will raise an error if a sklearn model name conflicts with existing checkpoint keys
    """

    def setup(
        self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None
    ) -> None:
        sklearn_modules = _get_sklearn_modules(pl_module)
        stats = []
        for name, module in sklearn_modules.items():
            stats.append((name, module.__str__(), type(module)))
        headers = ["Module", "Name", "Type"]
        logging.info("Sklearn Modules:")
        logging.info(f"\n{tabulate(stats, headers, tablefmt='heavy_outline')}")

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        logging.info("Checking for non PyTorch modules to save... 🔧")
        modules = _get_sklearn_modules(pl_module)
        for name, module in modules.items():
            if name in checkpoint:
                raise RuntimeError(
                    f"Can't pickle {name}, already present in checkpoint"
                )
            checkpoint[name] = module
            logging.info(f"Saving non PyTorch system: {name} 🔧")

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        logging.info("Checking for non PyTorch modules to load... 🔧")
        for name, item in checkpoint.items():
            if isinstance(item, RegressorMixin) or isinstance(item, ClassifierMixin):
                setattr(pl_module, name, item)
                logging.info(f"Loading non PyTorch system: {name} 🔧")


def _contains_sklearn_module(item):
    if isinstance(item, RegressorMixin) or isinstance(item, ClassifierMixin):
        return True
    if isinstance(item, list):
        return np.any([_contains_sklearn_module(m) for m in item])
    if isinstance(item, dict):
        return np.any([_contains_sklearn_module(m) for m in item.values()])
    return False


def _get_sklearn_modules(module):
    modules = dict()
    for name in dir(module):
        if name[0] == "_":
            continue
        item = getattr(module, name)
        if _contains_sklearn_module(item):
            modules[name] = item
    return modules
