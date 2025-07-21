"""Online probe callback."""

from typing import Dict, Optional, Union

import torch
import torchmetrics
from hydra.utils import instantiate
from lightning.pytorch import Callback, LightningModule, Trainer
from loguru import logger as logging

from stable_ssl.utils import get_data_from_batch_or_outputs

from ..optim import LARS
from .utils import EarlyStopping, format_metrics_as_dict


class OnlineProbe(Callback):
    """Attaches a MLP for fine-tuning using the standard self-supervised protocol.

    This callback trains a probe (typically a linear classifier) on top of frozen
    features during the main training process. It manages its own optimizer and
    training loop without modifying the base model's methods.

    Args:
        name: Unique identifier for this probe instance
        input: Key in batch dict containing input features
        target: Key in batch dict containing target labels
        probe: The probe module (e.g., linear classifier)
        loss_fn: Loss function for probe training
        optimizer: Optimizer configuration for the probe
        scheduler: Learning rate scheduler configuration
        accumulate_grad_batches: Number of batches to accumulate gradients
        metrics: Metrics to track during training/validation
        early_stopping: Early stopping configuration
    """

    def __init__(
        self,
        name: str,
        input: str,
        target: str,
        probe: torch.nn.Module,
        loss_fn: callable,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        accumulate_grad_batches: int = 1,
        metrics: Optional[Union[dict, tuple, list, torchmetrics.Metric]] = None,
        early_stopping: Optional[EarlyStopping] = None,
    ) -> None:
        super().__init__()

        self.name = name
        self.input = input
        self.target = target
        self.loss_fn = loss_fn
        self.accumulate_grad_batches = accumulate_grad_batches
        self.early_stopping = early_stopping

        # Initialize probe module
        if isinstance(probe, torch.nn.Module):
            self.probe_module = probe
        elif callable(probe):
            self.probe_module = probe()
        else:
            self.probe_module = instantiate(probe, _convert_="object")

        # Store optimizer and scheduler configs
        self._optimizer_config = optimizer
        self._scheduler_config = scheduler

        # These will be initialized in setup
        self.optimizer = None
        self.scheduler = None
        self._train_metrics = None
        self._val_metrics = None

        # Format metrics
        self.metrics_config = metrics

        logging.info(f"Initialized OnlineProbe callback: {name}")
        logging.info(f"  - Input: {input}")
        logging.info(f"  - Target: {target}")
        logging.info(f"  - Accumulate grad batches: {accumulate_grad_batches}")

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Initialize optimizer, scheduler, and metrics."""
        if stage != "fit":
            return

        # Move probe to correct device
        self.probe_module = self.probe_module.to(pl_module.device)

        # Initialize optimizer
        if self._optimizer_config is None:
            logging.warning(f"{self.name}: No optimizer given, using default LARS")
            self.optimizer = LARS(
                self.probe_module.parameters(),
                lr=0.1,
                clip_lr=True,
                eta=0.02,
                exclude_bias_n_norm=True,
                weight_decay=0,
            )
        else:
            if callable(self._optimizer_config):
                self.optimizer = self._optimizer_config(self.probe_module.parameters())
            else:
                self.optimizer = instantiate(
                    self._optimizer_config,
                    params=self.probe_module.parameters(),
                    _convert_="object",
                )

        # Initialize scheduler
        if self._scheduler_config is None:
            logging.warning(f"{self.name}: No scheduler given, using ConstantLR")
            self.scheduler = torch.optim.lr_scheduler.ConstantLR(
                self.optimizer, factor=1.0
            )
        else:
            if callable(self._scheduler_config):
                self.scheduler = self._scheduler_config(self.optimizer)
            else:
                self.scheduler = instantiate(
                    self._scheduler_config, optimizer=self.optimizer, _convert_="object"
                )

        # Store probe module in pl_module for compatibility
        logging.info(f"{self.name}: Storing probe module in _callbacks_modules")
        if not hasattr(pl_module, "_callbacks_modules"):
            pl_module._callbacks_modules = {}
        pl_module._callbacks_modules[self.name] = self.probe_module

        # Initialize metrics
        logging.info(f"{self.name}: Setting up metrics")
        if not hasattr(pl_module, "_callbacks_metrics"):
            pl_module._callbacks_metrics = {}
        pl_module._callbacks_metrics[self.name] = format_metrics_as_dict(
            self.metrics_config
        )

        # Store references for easy access
        self._train_metrics = pl_module._callbacks_metrics[self.name]["_train"]
        self._val_metrics = pl_module._callbacks_metrics[self.name]["_val"]

        # Metrics will be automatically moved to correct device by Lightning

        logging.info(f"{self.name}: Setup complete")

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Dict,
        batch: Dict,
        batch_idx: int,
    ) -> None:
        """Perform probe training step."""
        # Get input and target data
        x = get_data_from_batch_or_outputs(
            self.input, batch, outputs, caller_name=self.name
        )
        y = get_data_from_batch_or_outputs(
            self.target, batch, outputs, caller_name=self.name
        )

        if x is None or y is None:
            return

        # Ensure probe is in training mode
        self.probe_module.train()

        # Forward pass with gradient enabled
        with torch.enable_grad():
            # Detach input features to prevent gradients flowing to main model
            x = x.detach()

            # Ensure input has same dtype as probe module
            # This handles mixed precision training where features might be float16
            probe_dtype = next(self.probe_module.parameters()).dtype
            if x.dtype != probe_dtype:
                x = x.to(probe_dtype)

            # Forward through probe
            preds = self.probe_module(x)

            # Compute loss
            loss = self.loss_fn(preds, y)

            # Scale loss for gradient accumulation
            loss = loss / self.accumulate_grad_batches

            # Backward pass
            loss.backward()

        # Store predictions in batch (detached)
        prediction_key = f"{self.name}_preds"
        if prediction_key not in batch:
            batch[prediction_key] = preds.detach()

        # Update metrics and log
        logs = {f"train/{self.name}_loss": loss.item() * self.accumulate_grad_batches}
        for metric_name, metric in pl_module._callbacks_metrics[self.name][
            "_train"
        ].items():
            metric(preds.detach(), y)
            logs[f"train/{self.name}_{metric_name}"] = metric

        pl_module.log_dict(logs, on_step=True, on_epoch=True)

        # Optimizer step (respecting gradient accumulation)
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            # Scheduler step (typically done per optimizer step, not per batch)
            if trainer.global_step % trainer.accumulate_grad_batches == 0:
                self.scheduler.step()

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Dict,
        batch: Dict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Compute probe predictions during validation."""
        # Get input and target data
        x = get_data_from_batch_or_outputs(
            self.input, batch, outputs, caller_name=self.name
        )
        y = get_data_from_batch_or_outputs(
            self.target, batch, outputs, caller_name=self.name
        )

        if x is None or y is None:
            return

        # Ensure probe is in eval mode
        self.probe_module.eval()

        # Forward pass without gradients
        with torch.no_grad():
            # Ensure input has same dtype as probe module
            # This handles mixed precision training where features might be float16
            probe_dtype = next(self.probe_module.parameters()).dtype
            if x.dtype != probe_dtype:
                x = x.to(probe_dtype)

            preds = self.probe_module(x)

        # Store predictions in batch
        prediction_key = f"{self.name}_preds"
        if prediction_key not in batch:
            batch[prediction_key] = preds

        # Update metrics and log
        logs = {}
        for metric_name, metric in pl_module._callbacks_metrics[self.name][
            "_val"
        ].items():
            metric(preds, y)
            logs[f"eval/{self.name}_{metric_name}"] = metric

        pl_module.log_dict(logs, on_step=False, on_epoch=True)

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Handle early stopping if configured."""
        if self.early_stopping is None:
            return

        # Get the metric value for early stopping
        metric_name = f"eval/{self.name}_{self.early_stopping.monitor}"
        if metric_name not in trainer.callback_metrics:
            logging.warning(
                f"{self.name}: Early stopping metric {metric_name} not found"
            )
            return

        current_value = trainer.callback_metrics[metric_name]
        should_stop = self.early_stopping.should_stop(
            current_value, trainer.current_epoch
        )

        if should_stop:
            logging.info(
                f"{self.name}: Early stopping triggered at epoch {trainer.current_epoch}"
            )
            trainer.should_stop = True

    def state_dict(self) -> Dict:
        """Save callback state including probe module and optimizer states."""
        return {
            "probe_module": self.probe_module.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        """Load callback state including probe module and optimizer states."""
        if "probe_module" in state_dict:
            self.probe_module.load_state_dict(state_dict["probe_module"])
        if "optimizer" in state_dict and self.optimizer:
            self.optimizer.load_state_dict(state_dict["optimizer"])
        if "scheduler" in state_dict and self.scheduler:
            self.scheduler.load_state_dict(state_dict["scheduler"])
