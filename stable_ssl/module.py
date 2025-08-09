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
    """Stable-SSL module with manual optimization for SSL training (supports multiple optimizers)."""

    def __init__(self, *args, forward: callable, hparams: dict = None, **kwargs):
        super().__init__()
        logging.info("Initializing Module configuration...")

        # Manual optimization to support multiple optimizers and custom stepping
        self.automatic_optimization = False

        self._callbacks_modules = torch.nn.ModuleDict()
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

        # Internal optimizer metadata filled in configure_optimizers
        self._optimizer_names = None
        self._optimizer_index_by_name = None
        self._optimizer_frequencies = None

    def forward(self, *args, **kwargs):
        raise NotImplementedError("The forward() method must be implemented.")

    def training_step(self, batch, batch_idx):
        """Manual optimization training step with support for multiple optimizers.

        Expected outputs from forward during training (stage="fit"):
        - Single optimizer or joint loss: state["loss"]: torch.Tensor
          If multiple optimizers are configured and only `loss` is provided, it is treated
          as a joint loss for all optimizers.
        - Multiple optimizers with distinct losses: state["losses"]: Mapping where keys are
          optimizer names matching self.optim keys (preferred), or integer indices matching
          optimizer order.
        """
        state = self.forward(batch, stage="fit")

        # Early exit if optimization disabled
        if getattr(self, "optim", None) is None or self.optim is False:
            return state

        if not ("loss" in state or "losses" in state):
            raise ValueError(
                "Training step requires 'loss' (single/joint) or 'losses' (multi-optimizer) in the output state."
            )

        # Resolve optimizers and schedulers (can be single or list)
        optimizers = self.optimizers()
        if not isinstance(optimizers, (list, tuple)):
            optimizers = [optimizers]

        schedulers = self.lr_schedulers()
        if schedulers is None:
            schedulers = []
        elif not isinstance(schedulers, (list, tuple)):
            schedulers = [schedulers]

        num_optimizers = len(optimizers)

        # Build mapping from optimizer index -> loss tensor
        losses_by_index = {}
        if "losses" in state:
            if "loss" in state:
                logging.warning(
                    "Both 'losses' and 'loss' provided; ignoring 'loss' and using 'losses'."
                )
            losses = state["losses"]
            if isinstance(losses, dict):
                for key, loss in losses.items():
                    if isinstance(key, str):
                        if (
                            self._optimizer_index_by_name
                            and key in self._optimizer_index_by_name
                        ):
                            idx = self._optimizer_index_by_name[key]
                            if 0 <= idx < num_optimizers:
                                losses_by_index[idx] = loss
                            else:
                                logging.warning(
                                    f"Mapped index {idx} for optimizer '{key}' is out of range (have {num_optimizers})."
                                )
                        else:
                            logging.warning(
                                f"Loss for optimizer name '{key}' provided, but no matching optimizer was found."
                            )
                    elif isinstance(key, int):
                        if 0 <= key < num_optimizers:
                            losses_by_index[key] = loss
                        else:
                            logging.warning(
                                f"Loss index {key} is out of range for {num_optimizers} optimizers."
                            )
                    else:
                        logging.warning(
                            f"Unsupported key type in 'losses': {type(key)}. Expected str (name) or int (index)."
                        )
            else:
                raise TypeError(
                    "state['losses'] must be a dict mapping optimizer name or index to a loss tensor."
                )
        elif "loss" in state:
            # Treat as joint loss for all configured optimizers
            for idx in range(num_optimizers):
                losses_by_index[idx] = state["loss"]

        if not losses_by_index:
            raise ValueError(
                "No valid losses could be mapped to configured optimizers."
            )

        # Gradient accumulation factor
        accum = max(int(getattr(self.trainer, "accumulate_grad_batches", 1)), 1)
        scale = 1.0 / float(accum)

        # Deduplicate losses: map unique loss id -> (loss_tensor, representative_optimizer_idx)
        unique_losses = []
        seen = {}
        for idx, loss in losses_by_index.items():
            key = id(loss)
            if key not in seen:
                seen[key] = len(unique_losses)
                unique_losses.append((loss, idx))
        num_unique = len(unique_losses)

        # Backward once per unique loss tensor, toggling a representative optimizer
        for i, (loss, rep_idx) in enumerate(unique_losses):
            opt = optimizers[rep_idx]
            self.toggle_optimizer(opt)
            retain = i < (num_unique - 1)
            self.manual_backward(loss * scale, retain_graph=retain)
            self.untoggle_optimizer(opt)

        # Optional user-provided per-step callbacks
        if hasattr(self, "callbacks_training_step"):
            try:
                for fn in self.callbacks_training_step:
                    fn(batch_idx)
            except Exception as e:
                logging.warning(f"callbacks_training_step execution failed: {e}")

        # Stepping and gradient clipping at accumulation boundary
        if (batch_idx + 1) % accum == 0:
            for idx, opt in enumerate(optimizers):
                # Honor per-optimizer frequency if available
                step_freq = 1
                if self._optimizer_names and self._optimizer_frequencies:
                    name = self._optimizer_names[idx]
                    step_freq = int(self._optimizer_frequencies.get(name, 1))
                if step_freq < 1:
                    step_freq = 1

                if (batch_idx + 1) % step_freq != 0:
                    continue

                # Clip gradients for this optimizer then step
                self.clip_gradients(
                    opt,
                    gradient_clip_val=self.trainer.gradient_clip_val,
                    gradient_clip_algorithm=self.trainer.gradient_clip_algorithm,
                )
                opt.step()
                opt.zero_grad(set_to_none=True)

                # Step matching scheduler if it exists
                if idx < len(schedulers) and schedulers[idx] is not None:
                    try:
                        schedulers[idx].step()
                    except Exception as e:
                        logging.warning(
                            f"Scheduler step failed for optimizer index {idx}: {e}"
                        )

        return state

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
        """Configure optimizers and schedulers for manual optimization.

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

            # Track names/frequencies for training_step
            self._optimizer_names = ["default"]
            self._optimizer_index_by_name = {"default": 0}
            self._optimizer_frequencies = {
                "default": int(self.optim.get("frequency", 1))
            }

            # Return in list/dict style compatible with lr_schedulers() access
            return [opt], [
                {
                    "scheduler": sched,
                    "interval": "step",
                    "frequency": 1,
                }
            ]

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

        # Build a map of module name -> assigned optimizer index using nearest ancestor match
        assigned_group_by_module = {}
        for name, module in self.named_modules():
            if "_callbacks_modules" in name:
                continue
            # determine parent's assigned group
            parent_name = name.rsplit(".", 1)[0] if "." in name else None
            parent_group = assigned_group_by_module.get(parent_name, None)
            # current match overrides parent if any
            current_group = parent_group
            for idx, regex in regex_map:
                if regex.match(name):
                    current_group = idx
                    break
            assigned_group_by_module[name] = current_group
            # collect this module's direct parameters for the assigned group
            if current_group is not None:
                module_params = list(module.parameters(recurse=False))
                if module_params:
                    parameters[current_group].extend(module_params)

        # Build optimizers and schedulers
        optimizers = []
        schedulers = []

        self._optimizer_names = []
        self._optimizer_index_by_name = {}
        self._optimizer_frequencies = {}

        for (name, config), params in zip(optim_items, parameters):
            if not params:
                logging.warning(f"No parameters matched for optimizer {name}")
                # skip registration when there are no parameters
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

            # Track names and frequencies aligned to optimizer order
            self._optimizer_names.append(name)
            self._optimizer_index_by_name[name] = len(optimizers) - 1
            self._optimizer_frequencies[name] = int(config.get("frequency", 1))

        return optimizers, schedulers
