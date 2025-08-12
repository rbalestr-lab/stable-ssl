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
    """Online probe for evaluating learned representations during self-supervised training.

    This callback implements the standard linear evaluation protocol by training a probe
    (typically a linear classifier) on top of frozen features from the main model. The probe
    is trained simultaneously with the main model but maintains its own optimizer, scheduler,
    and training loop. This allows monitoring representation quality throughout training
    without modifying the base model.

    Key features:
    - Automatic gradient detachment to prevent probe gradients affecting the main model
    - Independent optimizer and scheduler management
    - Support for gradient accumulation
    - Mixed precision training compatibility through automatic dtype conversion
    - Built-in early stopping support
    - Metric tracking and logging

    Args:
        name: Unique identifier for this probe instance. Used for logging and storing
            metrics/modules.
        input: Key in batch dict or outputs dict containing input features to probe.
        target: Key in batch dict containing ground truth target labels.
        probe: The probe module to train. Can be a nn.Module instance, callable that
            returns a module, or Hydra config to instantiate.
        loss_fn: Loss function for probe training (e.g., nn.CrossEntropyLoss()).
        optimizer: Optimizer configuration for the probe. Can be optimizer instance,
            callable, or Hydra config. Defaults to LARS if not specified.
        scheduler: Learning rate scheduler configuration. Can be scheduler instance,
            callable, or Hydra config. Defaults to ConstantLR if not specified.
        accumulate_grad_batches: Number of batches to accumulate gradients before
            optimizer step. Default is 1 (no accumulation).
        metrics: Metrics to track during training/validation. Can be dict, list, tuple,
            or single metric instance.
        early_stopping: Early stopping configuration to halt training if validation
            metric stops improving.

    Note:
        - The probe module is stored in pl_module._callbacks_modules[name]
        - Metrics are stored in pl_module._callbacks_metrics[name]
        - Predictions are stored in batch dict with key '{name}_preds'
        - Loss is logged as 'train/{name}_loss'
        - Metrics are logged with prefix 'train/{name}_' and 'eval/{name}_'
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

        # Store probe configuration for later initialization
        # The actual module will be stored in pl_module._callbacks_modules
        self._probe_config = probe
        self._pl_module = None  # Will be set in setup()

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

        # Store reference to pl_module for accessing _callbacks_modules
        self._pl_module = pl_module

        # Initialize probe module if needed
        if isinstance(self._probe_config, torch.nn.Module):
            probe_module = self._probe_config
        elif callable(self._probe_config):
            probe_module = self._probe_config()
        else:
            probe_module = instantiate(self._probe_config, _convert_="object")

        # Move probe to correct device
        probe_module = probe_module.to(pl_module.device)

        # Store probe module in pl_module._callbacks_modules (single source of truth)
        logging.info(f"{self.name}: Storing probe module in _callbacks_modules")
        if not hasattr(pl_module, "_callbacks_modules"):
            pl_module._callbacks_modules = {}
        pl_module._callbacks_modules[self.name] = probe_module

        # Initialize optimizer using the stored module
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

        logging.info(f"{self.name}: Setting up metrics")
        if not hasattr(pl_module, "_callbacks_metrics"):
            pl_module._callbacks_metrics = {}
        pl_module._callbacks_metrics[self.name] = format_metrics_as_dict(
            self.metrics_config
        )

        self._train_metrics = pl_module._callbacks_metrics[self.name]["_train"]
        self._val_metrics = pl_module._callbacks_metrics[self.name]["_val"]

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
        x = get_data_from_batch_or_outputs(
            self.input, batch, outputs, caller_name=self.name
        )
        y = get_data_from_batch_or_outputs(
            self.target, batch, outputs, caller_name=self.name
        )

        if x is None or y is None:
            return

        self.probe_module.train()

        with torch.enable_grad():
            x = x.detach()

            probe_dtype = next(self.probe_module.parameters()).dtype
            if x.dtype != probe_dtype:
                x = x.to(probe_dtype)

            preds = self.probe_module(x)

            loss = self.loss_fn(preds, y)

            loss = loss / self.accumulate_grad_batches

            loss.backward()

        prediction_key = f"{self.name}_preds"
        if prediction_key not in batch:
            batch[prediction_key] = preds.detach()

        logs = {f"train/{self.name}_loss": loss.item() * self.accumulate_grad_batches}
        for metric_name, metric in pl_module._callbacks_metrics[self.name][
            "_train"
        ].items():
            metric(preds.detach(), y)
            logs[f"train/{self.name}_{metric_name}"] = metric

        pl_module.log_dict(logs, on_step=True, on_epoch=True)

        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

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

    @property
    def probe_module(self):
        """Access probe module from pl_module._callbacks_modules.

        This property is only accessible after setup() has been called.
        The probe module is stored centrally in pl_module._callbacks_modules
        to avoid duplication in checkpoints.
        """
        if self._pl_module is None:
            raise AttributeError(
                f"{self.name}: probe_module not accessible before setup(). "
                "The probe module is initialized during the setup phase."
            )
        return self._pl_module._callbacks_modules[self.name]

    @property
    def state_key(self) -> str:
        return f"OnlineProbe[name={self.name}]"

    def state_dict(self) -> Dict:
        """Save callback state - only optimizer and scheduler states.

        The probe module itself is saved via pl_module._callbacks_modules,
        so we don't duplicate it here.
        """
        return {
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        """Load callback state - only optimizer and scheduler states.

        The probe module itself is loaded via pl_module._callbacks_modules,
        so we don't handle it here.
        """
        if "optimizer" in state_dict and self.optimizer:
            self.optimizer.load_state_dict(state_dict["optimizer"])
        if "scheduler" in state_dict and self.scheduler:
            self.scheduler.load_state_dict(state_dict["scheduler"])
