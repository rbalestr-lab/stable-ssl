import re
import types
from functools import partial

import lightning as pl
import torch
import torchmetrics
from loguru import logger as logging
from tabulate import tabulate

from .utils import get_required_fn_parameters


class Module(pl.LightningModule):
    """PyTorch Lightning module with automatic optimization for SSL training."""

    def __init__(self, *args, forward: callable, hparams: dict = None, **kwargs):
        super().__init__()
        logging.info("Initializing Module configuration...")

        self._callbacks_modules = torch.nsn.ModuleDict()
        self._callbacks_metrics = torch.nn.ModuleDict()

        if len(args) > 0:
            raise ValueError(
                "Module does not accept positional arguments (*args). Please use keyword arguments instead (e.g., Module(forward=my_forward, hparams=my_hparams))."
            )

        if hparams is None:
            logging.warning(
                "No hyperparameters provided - hyperparameter logging is disabled."
            )
        else:
            logging.info("Saving provided hyperparameters.")
            self.save_hyperparameters(hparams)

        logging.info("Setting custom forward method.")
        setattr(self, "forward", types.MethodType(forward, self))

        for key, value in kwargs.items():
            logging.info(f"Setting attribute: self.{key} = {type(value)}")
            setattr(self, key, value)

        headers = ["Stage", "Inputs", "Metric"]
        if hasattr(self, "metrics"):
            stats = []
            assert isinstance(self.metrics, torch.nn.ModuleDict)
            logging.info("Metrics:")
            for stage, metrics in self.metrics.items():
                assert (
                    isinstance(metrics, torch.nn.ModuleDict)
                    or isinstance(metrics, torch.nn.ModuleList)
                    or isinstance(metrics, torchmetrics.Metric)
                )
                for name, metric in metrics.items():
                    stats.append([stage, name, str(metric)])
            logging.info(f"\n{tabulate(stats, headers, tablefmt='heavy_outline')}")
        else:
            self.metrics = dict(train={}, validate={}, test={}, predict={})
            logging.info(
                "No metrics configuration provided - automatic metric tracking is disabled."
            )

    def forward(self, *args, **kwargs):
        raise NotImplementedError("The forward() method must be implemented.")

    def training_step(self, batch, batch_idx):
        """Training step with automatic optimization."""
        state = self.forward(batch, stage="fit")

        # Return state directly - Lightning will extract 'loss' if present
        if "loss" in state:
            return state
        elif self.optim is None or self.optim is False:
            # No optimization needed
            return state
        else:
            logging.error(
                "Training step failed: The forward() method must return a dictionary containing a 'loss' key for optimization.\n"
                "To fix this, either:\n"
                "1. Add a 'loss' key to the dictionary returned by forward() to enable model training.\n"
                "2. Set optim=False when creating the module to indicate inference-only mode (no training)."
            )

    def validation_step(self, batch, batch_idx):
        state = self.forward(batch, stage="validate")
        return state

    def test_step(self, batch, batch_idx):
        state = self.forward(batch, stage="test")
        return state

    def predict_step(self, batch, batch_idx):
        state = self.forward(batch, stage="predict")
        return state

    def _create_scheduler(self, optimizer, name: str = "CosineAnnealingLR"):
        if name == "CosineAnnealingLR":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.estimated_stepping_batches
            )
        elif name == "OneCycleLR":
            pct = min(10 / self.trainer.max_epochs, 0.01)
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=optimizer.param_groups[0]["lr"],
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=pct,
            )
        else:
            raise ValueError(
                f"Unsupported scheduler: {name}. Supported types: CosineAnnealingLR, OneCycleLR"
            )

    def configure_optimizers(self):
        """Configure optimizers and schedulers for automatic optimization.

        Returns:
            dict or tuple: Optimizer configuration with optional learning rate scheduler.
            For single optimizer: Returns a dict with optimizer and lr_scheduler.
            For multiple optimizers: Returns a tuple of (optimizers, schedulers).
        """
        logging.info("Configuring optimizers and learning rate schedulers...")

        # Early exit for disabled optimization
        if hasattr(self, "optim") and not self.optim:
            logging.info("Optimization disabled - skipping optimizer configuration.")
            return None

        if not hasattr(self, "optim"):
            logging.info(
                "Using default optimization setup: AdamW optimizer with CosineAnnealingLR scheduler."
            )
            self.optim = dict(optimizer=partial(torch.optim.AdamW))

        # Single optimizer case
        optimizer_fn = self.optim.get("optimizer")
        if isinstance(optimizer_fn, partial):
            logging.info("Configuring single optimizer.")
            assert callable(optimizer_fn)
            assert get_required_fn_parameters(optimizer_fn) == ["params"]

            # Direct parameter extraction - single pass
            params = [
                p
                for name, p in self.named_parameters()
                if "_callbacks_modules" not in name
            ]

            opt = optimizer_fn(params)

            # Create scheduler
            sched_name = self.optim.get("scheduler", "CosineAnnealingLR")
            sched = self._create_scheduler(opt, sched_name)

            logging.info(
                f"Configured {opt.__class__.__name__} optimizer with {sched_name} scheduler."
            )

            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": sched,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        # Multiple optimizers case - check once
        if not isinstance(self.optim, dict):
            raise ValueError(
                "Optimizer must be either a partial function or a dict of optimizer configs"
            )

        # Verify all values are dicts
        optim_items = list(self.optim.items())
        if not all(isinstance(v, dict) for _, v in optim_items):
            raise ValueError("For multiple optimizers, all config values must be dicts")

        logging.info(
            f"\tOptimizer specified by Dict with keys {[k for k, _ in optim_items]}... 🔧"
        )

        # Pre-compile regex patterns with their indices
        regex_map = [
            (i, re.compile(config["modules"]))
            for i, (_, config) in enumerate(optim_items)
        ]
        num_optimizers = len(optim_items)
        parameters = [[] for _ in range(num_optimizers)]

        # Single pass through modules with early matching
        for name, module in self.named_modules():
            if "_callbacks_modules" in name:
                continue

            # Use generator for params to avoid list creation if not needed
            module_params = list(module.parameters(recurse=False))
            if not module_params:
                continue

            for idx, regex in regex_map:
                if regex.match(name):
                    parameters[idx].extend(module_params)
                    break

        # Build optimizers and schedulers
        optimizers = []
        schedulers = []

        for (name, config), params in zip(optim_items, parameters):
            if not params:
                logging.warning(f"No parameters matched for optimizer {name}")
                continue

            opt = config["optimizer"](params)
            optimizers.append(opt)

            sched_name = config.get("scheduler", "CosineAnnealingLR")
            schedulers.append(
                {
                    "scheduler": self._create_scheduler(opt, sched_name),
                    "interval": "step",
                    "frequency": 1,
                    "name": name,
                }
            )

            logging.info(
                f"Configured optimizer '{name}' ({len(params)} parameters) with {sched_name} scheduler."
            )

        return optimizers, schedulers
