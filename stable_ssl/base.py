# -*- coding: utf-8 -*-
"""Base class for training a model."""
#
# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#         Randall Balestriero <randallbalestriero@gmail.com>
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import time
import hydra
import gc
import numpy as np
import submitit
import jsonlines
from tqdm import tqdm
import subprocess
import os
import omegaconf
from abc import abstractmethod

from dataclasses import asdict

import torch

from .data import DistributedSamplerWrapper
from . import reader
from .config import LoggerConfig, WandbConfig, HardwareConfig, OptimConfig
from .monitors import Monitor
from .modules import TeacherModule

try:
    import wandb
except ModuleNotFoundError:
    logging.warning(
        "Wandb module is not installed, make sure not to use wandb for logging "
        "or an error will be thrown."
    )
from .utils import (
    BreakAllEpochs,
    BreakEpoch,
    NanError,
    seed_everything,
    to_device,
    get_gpu_info,
    log_and_raise,
)


class BaseTrainer(torch.nn.Module):
    r"""Base class for training a model.

    This class provides a general boilerplate for common operations that
    occur during the training lifecycle.
    These operations include training, evaluation, checkpointing, and training restart.

    The class is highly configurable, enabling customization of its internal workflows
    to suit diverse project requirements and use cases.

    This class is intended to be subclassed for specific training methods
    (see examples for more details). For each subclass, the following methods must
    be implemented: `forward`, `predict`, and `compute_loss`.

    Execution flow when calling `launch`:
            - `self.before_fit` (nothing by default)
            - `self._fit` (executes all the training/intermittent evaluation by default)
                - for `self.optim["epochs"]` epochs:
                    - `self.before_fit_epoch` (setup in train mode)
                    - `self._fit_epoch` (one training epoch by default)
                        - loop over mini-batches
                            - `self.before_fit_step` (moves data to device)
                            - `self._fit_step` (optimization step)
                            - `self.after_fit_step` (perf monitoring and teacher update)
                    - `self.after_fit_epoch` (nothing by default)
                    - `self._evaluate` (if asked, looping over all non-train datasets)
                        - `self.before_eval` (setup in eval mode)
                        - loop over mini-batches
                            - `self.before_eval_step` (moves data to device)
                            - `self._eval_step` (computes eval metric)
                            - `self.after_eval_step` (nothing by default)
                        - `self.after_eval` (nothing by default)
                    - Save intermittent checkpoint if asked by user config
                - Save final checkpoint if asked by user config
            - `self.after_fit` (evaluates by default)

    Parameters
    ----------
    data: dict
        Names and construction of the dataloaders with their transform pipelines.
        The dataset named `train` is used for training.
        Any other dataset is used for validation.
    module: dict
        Names and definition of the modules (neural networks).
        See :mod:`stable_ssl.modules` for examples of available modules.
    loss: dict
        Loss function used in the final criterion to be minimized.
        See :mod:`stable_ssl.losses` for examples.
    hardware: dict
        Hardware parameters. See :mod:`stable_ssl.config.HardwareConfig`
        for the full list of parameters and their defaults.
    optim: dict
        Optimization parameters. See :mod:`stable_ssl.config.OptimConfig`
        for the full list of parameters and their defaults.
    logger: dict
        Logging and checkpointing parameters.
        See :mod:`stable_ssl.config.LoggerConfig`
        for the full list of parameters and their defaults.
    """

    def __init__(self, data, module, loss, hardware, optim, logger, **kwargs):
        super().__init__()
        logging.info(f"=> INIT OF {self.__class__.__name__} STARTED.")
        for key, value in kwargs.items():
            setattr(self, key, value)
        self._data = data
        self._module = module
        self._loss = loss
        self._hardware = hardware
        self._optim = optim
        self._logger = logger

        # Set the logger defaults.
        self._logger = asdict(LoggerConfig(**self._logger))
        if type(self._logger["wandb"]) is bool and self._logger["wandb"] is True:
            self._logger["wandb"] = {}
        if isinstance(self._logger.get("wandb"), dict) and self.rank == 0:
            self._logger["wandb"] = asdict(WandbConfig(**self._logger["wandb"]))

        # Set the hardware defaults.
        self._hardware = asdict(HardwareConfig(**self._hardware))

        # Set the optimizer defaults.
        self._optim = asdict(OptimConfig(**self._optim))

        # Save the full config with defaults.
        c = self.get_config()
        c["trainer"]["logger"] = self._logger
        c["trainer"]["hardware"] = self._hardware
        c["trainer"]["optim"] = self._optim
        with open(self._logger["dump_path"] / ".hydra" / "config.yaml", "w") as f:
            omegaconf.OmegaConf.save(c, f)

        logging.info(f"=> INIT OF {self.__class__.__name__} COMPLETED.")

    def __call__(self):
        """Call the setup and launch methods."""
        self.setup()
        self.launch()

    def setup(self):
        """Instantiate components and load the checkpoint (if applicable)."""
        logging.getLogger().setLevel(self._logger["level"])
        logging.info(f"=> SETUP OF {self.__class__.__name__} STARTED.")
        self._instanciate()
        self._load_checkpoint()
        logging.info(f"=> SETUP OF {self.__class__.__name__} COMPLETED.")

    def launch(self):
        """Execute the core training and evaluation routine.

        This method runs the training and evaluation process,
        with a customizable boilerplate structure.

        The default flow includes:
        - Running evaluation and cleanup if no "train" dataset is found in `self.data`.
        - Otherwise performing pre-training, training, and post-training tasks.

        Exceptions
        ----------
        BreakAllEpochs
            Raised if the training is interrupted by the user.

        """
        if "train" not in self.data:
            self._evaluate()
            self._cleanup()
            return
        try:
            self.before_fit()
            self._fit()
            self.after_fit()
        except BreakAllEpochs:
            logging.exception("Training stopped by user.")
            raise
        if self.logger["wandb"]:
            wandb.finish()
        self._cleanup()

    @abstractmethod
    def forward(self):
        """Forward pass of the model."""
        pass

    @abstractmethod
    def predict(self):
        """Prediction of the model used for evaluation."""
        pass

    @abstractmethod
    def compute_loss(self):
        """Compute the global loss to be minimized."""
        pass

    def get_logs(self, keys=None):
        """Retrieve the logs from the logger."""
        if self.logger["wandb"] is None:
            return reader.jsonl(self.logger["dump_path"])
        else:
            return reader.wandb(
                self.logger["wandb"]["entity"],
                self.logger["wandb"]["project"],
                self.logger["wandb"]["ID"],
                keys=keys,
            )

    def get_config(self):
        """Retrieve the configuration file of the trainer."""
        config = omegaconf.OmegaConf.load(
            self._logger["dump_path"] / ".hydra" / "config.yaml"
        )
        return config

    def check_module(self):
        """Check if modules are compatible with trainer. Specific to each trainer."""
        pass

    def before_fit(self):
        """Initialize training by setting the starting epoch."""
        self.epoch = 0

    def after_fit(self):
        """Evaluate the model after completing the training process."""
        self._evaluate()

    def before_fit_epoch(self):
        """Prepare the training state and set the epoch for distributed training."""
        self.train()
        if self.world_size > 1:
            self.data["train"].set_epoch(self.epoch)

    def after_fit_epoch(self):
        """Handle post-epoch tasks after training (currently does nothing)."""
        pass

    def before_fit_step(self):
        """Prepare batch for training step by moving it to the appropriate device."""
        self.batch = to_device(self.batch, self.device)

    def after_fit_step(self):
        """Handle per-step monitoring and teacher update (if applicable)."""
        # Compute and log the monitoring metrics.
        if "train" in self.logger["monitor"]:
            for metric in self.logger["monitor"]["train"].values():
                metric: Monitor
                score = metric.compute(self._latest_forward)
                if self.global_step % self.logger["log_every_step"] == 0:
                    self._log({f"train/{metric.name}": score})

        # Update the teacher network if there is one.
        for m in self.modules():
            if isinstance(m, TeacherModule):
                m.step()

    def before_eval(self):
        """Set the model to evaluation mode before validation/testing."""
        self.eval()

    def after_eval(self):
        """Handle tasks after completing evaluation (currently does nothing)."""
        pass

    def before_eval_step(self):
        """Prepare batch for evaluation step by moving it to the appropriate device."""
        self.batch = to_device(self.batch, self.device)

    def after_eval_step(self):
        """Handle post-step tasks after an evaluation step (currently does nothing)."""
        pass

    def checkpoint(self) -> submitit.helpers.DelayedSubmission:
        """Create a checkpoint of the current state of the model.

        This method is called asynchronously when the SLURM manager sends a
        preemption signal. It is invoked with the same arguments as the `__call__`
        method. At this point, `self` represents the current state of the model.

        Returns
        -------
        submitit.helpers.DelayedSubmission: A delayed submission object
            representing the requeued task with the current model state.
        """
        logging.info("Requeuing the task...")
        self._save_checkpoint("tmp_checkpoint.ckpt", model_only=False)
        model = type(self)(
            self._data,
            self._module,
            self._loss,
            self._hardware,
            self._optim,
            self._logger,
        )
        logging.info("Cleaning up the current task before submitting a new one.")
        self._cleanup()
        return submitit.helpers.DelayedSubmission(model)

    @property
    def rank(self):
        if self.world_size > 1:
            return torch.distributed.get_rank()
        return 0

    @property
    def world_size(self):
        if torch.distributed.is_initialized():
            return torch.distributed.get_world_size()
        return 1

    @property
    def batch_idx(self):
        if not hasattr(self, "_batch_idx"):
            return None
        return self._batch_idx

    @property
    def device(self):
        return self._device

    @property
    def epoch(self):
        if not hasattr(self, "_epoch"):
            return None
        return self._epoch

    @property
    def step(self):
        if not hasattr(self, "_step"):
            return None
        return self._step

    @property
    def latest_forward(self):
        if not hasattr(self, "_latest_forward"):
            return None
        return self._latest_forward

    @epoch.setter
    def epoch(self, value):
        self._epoch = value

    @step.setter
    def step(self, value):
        self._step = value

    @latest_forward.setter
    def latest_forward(self, value):
        self._latest_forward = value

    def _instanciate(self):
        seed_everything(self._hardware.get("seed", None))

        self.start_time = time.time()
        # we skip optim as we may not need it (see below)
        self.data = hydra.utils.instantiate(self._data, _convert_="object")
        self.module = hydra.utils.instantiate(self._module, _convert_="object")
        self.loss = hydra.utils.instantiate(self._loss, _convert_="object")
        self.hardware = hydra.utils.instantiate(self._hardware, _convert_="object")
        self.logger = hydra.utils.instantiate(self._logger, _convert_="partial")

        self._set_device(self.hardware)

        self.loss = self.loss.to(self._device)

        logging.info("Logger:")
        logging.info(f"\t- Dump path: `{self.logger['dump_path']}`")
        if self.logger["wandb"]:
            logging.info("\t- Wandb:")
            try:
                wandb.init(**self.logger["wandb"], resume="allow")
                self.logger["wandb"]["entity"] = wandb.run.entity
                self.logger["wandb"]["project"] = wandb.run.project
                self.logger["wandb"]["name"] = wandb.run.name
                self.logger["wandb"]["id"] = wandb.run.id
                logging.info(f"\t\t- entity: {wandb.run.entity}")
                logging.info(f"\t\t- project: {wandb.run.project}")
                logging.info(f"\t\t- name: {wandb.run.name}")
                logging.info(f"\t\t- id: {wandb.run.id}")
            except Exception:
                logging.exception("Failed to initialize wandb.")
                raise
        else:
            logging.info("\t- JSONL")

        # we do metrics before data to allow logging in data
        logging.info("Metrics:")
        for m in self.logger["metric"]:
            if type(self.logger["metric"][m]) is dict:
                self.logger["metric"][m] = torch.nn.ModuleDict(self.logger["metric"][m])
        if type(self.logger["metric"]) is dict:
            self.logger["metric"] = torch.nn.ModuleDict(self.logger["metric"])
        self.logger["metric"] = self.logger["metric"].to(self._device)

        # Data
        logging.info("Data:")
        for name, loader in self.data.items():
            if name[0] == "_":
                logging.info(f"\t- `{name}` ignored (starts with `_`).")
                continue
            logging.info(f"\t\t- length: {len(loader)}.")

            if name in self.logger["metric"]:
                logging.info("\t\t- metrics:")
                for mname in self.logger["metric"][name]:
                    logging.info(f"\t\t\t- {mname}.")
            else:
                if name != "train":
                    log_and_raise(
                        ValueError,
                        f"Metrics for dataset {name} are not defined in the config. "
                        "All datasets which are not `train` must have metrics defined.",
                    )

            if not len(loader):
                log_and_raise(ValueError, f"Length of dataset {name} is 0.")
            if self.world_size > 1:
                if not isinstance(
                    loader.sampler,
                    (
                        torch.utils.data.SequentialSampler,
                        torch.utils.data.RandomSampler,
                    ),
                ):
                    log_and_raise(
                        ValueError,
                        "Custom sampler with distributed version is not supported.",
                    )

                self.data[name] = DistributedSamplerWrapper(
                    loader, self.world_size, self.rank
                )
                logging.info(
                    f"\t- Length after DDS on this process `{len(self.data[name])}."
                )

        # Modules and scaler
        logging.info("Modules:")
        for name, module in self.module.items():
            # if self.config.model.memory_format == "channels_last":
            #     module.to(memory_format=torch.channels_last)
            if self.world_size > 1:
                module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module)
            module.to(self.device)
            has_parameters = False
            if sum(p.numel() for p in module.parameters() if p.requires_grad) > 0:
                has_parameters = True
            if self.world_size > 1 and has_parameters:
                module = torch.nn.parallel.DistributedDataParallel(
                    module, device_ids=[self._device]
                )
            self.module[name] = module
            trainable = sum(
                param.numel() for param in module.parameters() if param.requires_grad
            )
            logging.info(f"\t- {name} with {trainable} trainable parameters.")
        self.module = torch.nn.ModuleDict(self.module)
        self.check_module()
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.hardware["float16"])

        self.register_buffer("global_step", torch.zeros((1,), dtype=int))

        if "train" in self.data:
            logging.info("Setting up self.optim.")
            self.optim = hydra.utils.instantiate(self._optim, _convert_="object")
            self.optim["optimizer"] = self.optim["optimizer"](self.parameters())
            self.optim["scheduler"] = self.optim["scheduler"](self.optim["optimizer"])
        else:
            logging.info(
                "No `train` in data, skipping optimizer and scheduler initializations."
            )

        self._log_buffer = {}

    def _fit(self):
        while self.epoch < self.optim["epochs"]:
            try:
                self._fit_epoch()
            except BreakEpoch:
                logging.info(
                    "Train epoch interrupted by user. Proceeding to the next one."
                )
            except NanError:
                logging.error("NaN error encountered during training.", exc_info=True)
                return
            except Exception:
                logging.exception("An unexpected error occurred during training.")
                raise

            if self.epoch % self.logger["eval_every_epoch"] == 0:
                self._evaluate()
            self.epoch = self.epoch + 1

            if (
                self.logger["checkpoint_frequency"]
                and self.epoch % self.logger["checkpoint_frequency"] == 0
            ):
                self._save_checkpoint(
                    f"checkpoint_{self.epoch}.ckpt",
                    model_only=self.logger["checkpoint_model_only"],
                )

        # At the end of training, we (optionally) save the final model.
        if self.logger["save_final_model"]:
            self._save_checkpoint(
                f"{self.logger['save_final_model']}.ckpt", model_only=True
            )

        logging.info("Cleaning up any temporary checkpoint.")
        (self.logger["dump_path"] / "tmp_checkpoint.ckpt").unlink(missing_ok=True)

    def _fit_epoch(self):
        assert "train" in self.data
        self.before_fit_epoch()
        # We do not ensure that the model is still in train mode to not
        # override any user desired behavior, simply speak out.
        if not self.training:
            logging.warning(
                "Starting training epoch but model is no longer in "
                "train mode after call to before_fit_epoch()."
            )

        loader = self.data["train"]
        # If max_steps is negative, train on the full dataset.
        if self.optim["max_steps"] < 0:
            max_steps = len(loader)
        # If max_steps is a float between 0 and 1, treat it as a percentage.
        elif 0 < self.optim["max_steps"] < 1:
            max_steps = int(self.optim["max_steps"] * len(loader))
        # Otherwise, set max_steps to the length of the dataset.
        else:
            max_steps = min(self.optim["max_steps"], len(loader))

        for self._batch_idx, self.batch in enumerate(
            tqdm(loader, total=max_steps, desc=f"Training: {self.epoch}")
        ):
            self.before_fit_step()
            self._fit_step()
            self.after_fit_step()

            self.global_step.add_(1)
            if self.batch_idx >= max_steps:
                break

        self.after_fit_epoch()
        self.batch = None

    def _evaluate(self) -> dict:
        self.before_eval()
        # We do not ensure that the model is still in eval mode to not
        # override any user desired behavior.
        if self.training:
            logging.warning(
                "Starting eval epoch but model is not in "
                "eval mode after call to before_eval()."
            )

        packet = {"epoch": min(self.epoch, self.optim["epochs"] - 1)}
        for name_loader, loader in self.data.items():
            if name_loader == "train" or name_loader[0] == "_":
                continue
            # Reset the metrics for the epoch.
            if name_loader in self.logger["metric"]:
                for _, v in self.logger["metric"][name_loader].items():
                    v.reset()

            try:
                max_steps = len(loader)
                with torch.inference_mode():
                    for self._batch_idx, self.batch in tqdm(
                        enumerate(loader),
                        total=max_steps,
                        desc=f"Eval {name_loader}: {self.epoch=}",
                    ):

                        # Call any user specified pre-step function.
                        self.before_eval_step()

                        # Call the eval step.
                        with torch.amp.autocast(
                            "cuda", enabled=self.hardware["float16"]
                        ):
                            self._eval_step(name_loader=name_loader)

                        # Call any user specified post-step function.
                        self.after_eval_step()
            except BreakEpoch:
                logging.info("Eval epoch interrupted by user.")
            except Exception:
                logging.exception("An unexpected error occurred during evaluation.")
                raise

            # Be sure to clean up to avoid silent bugs.
            self.batch = None
            # Compute the final metrics for the epoch.
            if name_loader in self.logger["metric"]:
                for name, metric in self.logger["metric"][name_loader].items():
                    packet[f"{name_loader}/{name}"] = metric.compute()
        self._log(packet, commit=True)
        # Call any user specified post-epoch function.
        self.after_eval()

    def _fit_step(self):
        with torch.amp.autocast("cuda", enabled=self.hardware["float16"]):
            returned_loss = self.compute_loss()

        if isinstance(returned_loss, torch.Tensor) and returned_loss.numel() == 1:
            loss = returned_loss
        elif isinstance(returned_loss, list):
            loss = sum(returned_loss)
        elif isinstance(returned_loss, dict):
            loss = sum(returned_loss.values())
        else:
            log_and_raise(
                ValueError,
                "Returned loss must be a float, a list of floats or a dict of floats.",
            )

        if torch.isnan(loss):
            log_and_raise(NanError, "Loss is NaN. Stopping training.")

        self.scaler.scale(loss).backward()
        scale = self.scaler.get_scale()
        if (1 + self.batch_idx) % self.optim["accumulation_steps"] == 0:
            # Unscales the gradients of optimizer's assigned params in-place.
            self.scaler.unscale_(self.optim["optimizer"])
            if self.optim["grad_max_norm"] is not None:
                # Since the gradients of optimizer's assigned params are unscaled,
                # clips as usual:
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(), self.optim["grad_max_norm"]
                )
            self.scaler.step(self.optim["optimizer"])
            self.scaler.update()
            self.optim["optimizer"].zero_grad(set_to_none=True)

        # to avoid warning
        # see https://discuss.pytorch.org/t/
        # optimizer-step-before-lr-scheduler-step-error-using-gradscaler/92930/7
        if scale <= self.scaler.get_scale():
            self.optim["scheduler"].step()

        if self.global_step % self.logger["log_every_step"] == 0:
            bucket = {}
            if isinstance(returned_loss, dict):
                for name, value in returned_loss.items():
                    bucket[f"train/{name}"] = value.item()
            else:
                bucket["train/loss"] = loss.item()
            bucket["train/lr"] = self.optim["scheduler"].get_last_lr()[0]
            bucket["step"] = self.batch_idx
            bucket["epoch"] = self.epoch
            self._log(bucket, commit=True)

    def _eval_step(self, name_loader):
        output = self.predict()
        if name_loader in self.logger["metric"]:
            for metric in self.logger["metric"][name_loader].values():
                metric.update(output, self.batch[1])

        if name_loader in self.logger["monitor"]:
            for metric in self.logger["monitor"][name_loader].values():
                metric: Monitor
                # NOTE To make this more general (e.g. for GradNorm, etc.)
                # we should pass in the BaseModel in its entirety and let the
                # compute method use what it needs.
                score = metric.compute(output)
                if self.global_step % self.logger["log_every_step"] == 0:
                    self._log({f"{name_loader}/{metric.name}": score})

    def _set_device(self, hardware):
        # Check if CUDA is available, otherwise set to CPU.
        if not torch.cuda.is_available() or hardware["device"] == "cpu":
            logging.warning("CUDA is not available. Setting device to CPU.")
            self._device = "cpu"
            return

        if hardware["world_size"] > 1:
            logging.info("Setting up Distributed model.")

            dist_env = submitit.JobEnvironment()
            port = 10000 + int(os.environ["SLURM_JOBID"]) % 55000
            if "SLURM_JOB_NODELIST" in os.environ:

                cmd = ["scontrol", "show", "hostnames", os.getenv("SLURM_JOB_NODELIST")]
                host_name = subprocess.check_output(cmd).decode().splitlines()[0]
                dist_url = f"tcp://{host_name}:{port}"
            else:
                dist_url = f"tcp://localhost:{port}"

            logging.info(f"\tos MASTER_ADDR: {os.getenv('MASTER_ADDR')}")
            logging.info(f"\tos MASTER_PORT: {os.getenv('MASTER_PORT')}")
            logging.info(f"\tprocess group: {dist_env.num_tasks} tasks")
            logging.info(f"\tmaster: {dist_url}")
            logging.info(f"\tglobal rank: {dist_env.global_rank}")
            logging.info(f"\tlocal rank: {dist_env.local_rank}")
            logging.info(f"\tnumber of nodes: {dist_env.num_nodes}")
            logging.info(f"\tworld size: {dist_env.num_nodes*dist_env.num_tasks}")

            torch.distributed.init_process_group(
                "nccl",
                init_method=dist_url,
                rank=dist_env.global_rank,
                world_size=dist_env.num_nodes * dist_env.num_tasks,
            )
            assert dist_env.global_rank == torch.distributed.get_rank()
            assert (
                dist_env.num_nodes * dist_env.num_tasks
            ) == torch.distributed.get_world_size()
            self._device = f"cuda:{dist_env.local_rank}"
        else:
            self._device = hardware["device"]

        # Cleanup the device.
        logging.info("Device status at start of process:")
        get_gpu_info()

        # Set the CUDA device.
        torch.cuda.set_device(self._device)

    def _log(self, packet=None, commit=True):
        # Update the log buffer with the new packet.
        packet = packet or {}
        assert "_global_step" not in packet, logging.error(
            "'_global_step' is reserved but present in log packet."
        )
        self._log_buffer.update(packet)
        if not commit or len(self._log_buffer) == 0:
            return

        # Make values JSON serializable.
        for name, value in self._log_buffer.items():
            if torch.is_tensor(value):
                if torch.numel(value) == 1:
                    self._log_buffer[name] = value.item()
                else:
                    self._log_buffer[name] = value.tolist()

        # Log in WandB.
        if self.logger["wandb"] and self.rank == 0:
            for name, value in self._log_buffer.items():
                if isinstance(value, list):
                    # Create a WandB table if the value is a list.
                    table = wandb.Table(columns=["epoch", name])
                    for i, v in enumerate(np.asarray(value).flatten()):
                        table.add_data(i, v)
                    self._log_buffer[name] = table
                else:
                    self._log_buffer[name] = value
            wandb.log(self._log_buffer, step=self.global_step.item())

        # Log in jsonl.
        else:
            self._log_buffer.update(self._generate_logging_default_bucket())
            with jsonlines.open(
                self.logger["dump_path"] / f"logs_rank_{self.rank}.jsonl", mode="a"
            ) as writer:
                writer.write(self._log_buffer)

        # Clear the log buffer.
        self._log_buffer = {}

    def _load_checkpoint(self):
        logging.info("load_checkpoint:")
        target = self.logger["dump_path"] / "tmp_checkpoint.ckpt"
        if not target.is_file():
            logging.info(f"\t=> `{target}` not present... training from scratch.")
            self.epoch = 0
            return

        logging.info(f"\t=> folder `{target}` exists... loading it.")
        checkpoint = self.logger["dump_path"] / "tmp_checkpoint.ckpt"

        ckpt = torch.load(checkpoint, map_location="cpu")

        for name, model in self.named_children():
            if name not in ckpt:
                logging.info(f"\t\t=> {name} not in ckpt, skipping.")
                continue
            model.load_state_dict(ckpt[name])
            logging.info(f"\t\t=> {name} successfully loaded.")
        # model and metrics are children of the model so already
        # handles by the above for loop. But we need special case for
        # the optimizer(s) and scheduler(s)
        if "optimizer" in ckpt:
            self.optim["optimizer"].load_state_dict(ckpt["optimizer"])
            logging.info("\t\t=> optimizer successfully loaded.")
        if "scheduler" in ckpt:
            self.optim["optimizer"].load_state_dict(ckpt["scheduler"])
            logging.info("\t\t=> scheduler successfully loaded.")
        if "epoch" in ckpt:
            self.epoch = ckpt["epoch"]
            logging.info(f"\t\t=> training will start from epoch {ckpt['epoch']}.")
        else:
            self.epoch = 0

    def _save_checkpoint(self, name, model_only):
        if self.world_size > 1:
            curr_rank = torch.distributed.get_rank()
            if curr_rank != 0:
                logging.info(f"On rank {curr_rank}, only rank 0 saves the checkpoint.")
                return
        saving_name = self.logger["dump_path"] / name
        state = {}
        for subname, model in self.named_children():
            state[subname] = model.state_dict()
        if model_only:
            torch.save(state, saving_name)
            logging.info(f"Model saved at {saving_name}.")
            return
        if hasattr(self.optim, "optimizer"):
            state["optimizer"] = self.optim["optimizer"].state_dict()
        if hasattr(self.optim, "scheduler"):
            state["scheduler"] = self.optim["scheduler"].state_dict()
        state["epoch"] = self.epoch

        torch.save(state, saving_name)
        if model_only:
            mess = "(model only)"
        else:
            mess = "(model, optimizer, scheduler, epoch)"
        logging.info(f"Checkpoint {mess} saved at {saving_name}.")

    def _generate_logging_default_bucket(self):
        cur_time = time.time()
        rel_time = cur_time - self.start_time
        bucket = {
            "rank": self.rank,
            "timestamp": cur_time,
            "relative_time": rel_time,
            "training": self.training,
            "epoch": self.epoch,
            "step": self.batch_idx,
        }
        return bucket

    def _cleanup(self):
        logging.info("Cleaning up process, device status before cleaning:")
        get_gpu_info()

        if torch.distributed.is_initialized():
            logging.info("Cleaning distributed processes.")
            torch.distributed.destroy_process_group()

        gc.collect()
        torch.cuda.empty_cache()

        logging.info("Device status after cleaning.")
        get_gpu_info()
