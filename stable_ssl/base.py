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
from pathlib import Path
from tqdm import tqdm
import subprocess
import os
import omegaconf
import copy
import torch
import torch.nn.functional as F

from .data import DistributedSamplerWrapper
from . import reader
from .utils import deactivate_requires_grad, update_momentum

try:
    import wandb
except ModuleNotFoundError:
    logging.warning(
        "Wandb module is not installed, make sure not to use wandb for logging "
        "or an error will be thrown."
    )
from hydra.core.hydra_config import HydraConfig
from .utils import (
    BreakAllEpochs,
    BreakEpoch,
    NanError,
    seed_everything,
    to_device,
    get_gpu_info,
    log_and_raise,
)


# https://github.com/Lightning-AI/pytorch-lightning/blob/master/src/
# lightning/pytorch/overrides/distributed.py#L224
# class UnrepeatedDistributedSampler(DistributedSampler):
#     def __init__(self, *args: Any, **kwargs: Any) -> None:
#         super().__init__(*args, **kwargs)
#         if not isinstance(self.dataset, Sized):
#             raise TypeError("The given dataset must implement the `__len__` method.")
#         self.num_samples = len(range(self.rank, len(self.dataset), self.num_replicas))
#         self.total_size = len(self.dataset)
#         # If any process has at least one batch, every other process needs to
#         # have at least one batch, or the DistributedDataParallel could lock up.
#         assert self.num_samples >= 1 or self.total_size == 0

#     @override
#     def __iter__(self) -> Iterator[List[int]]:
#         if not isinstance(self.dataset, Sized):
#             raise TypeError("The given dataset must implement the `__len__` method.")
#         if self.shuffle:
#             # deterministically shuffle based on epoch
#             g = torch.Generator()
#             g.manual_seed(self.epoch)
#             indices = torch.randperm(len(self.dataset), generator=g).tolist()
#         else:
#             indices = list(range(len(self.dataset)))

#         assert len(indices) == self.total_size

#         # subsample
#         indices = indices[self.rank : self.total_size : self.num_replicas]
#         assert len(indices) == self.num_samples

#         return iter(indices)


# class UnrepeatedDistributedSamplerWrapper(UnrepeatedDistributedSampler):

#     def __init__(
#         self, sampler: Union[Sampler, Iterable], *args: Any, **kwargs: Any
#     ) -> None:
#         super().__init__(_DatasetSamplerWrapper(sampler), *args, **kwargs)

#     @override
#     def __iter__(self) -> Iterator:
#         self.dataset.reset()
#         return (self.dataset[index] for index in super().__iter__())


class BaseModel(torch.nn.Module):
    r"""Base class for training a model.

    That method provides a general boilerplate for all the internal operations
    that always occur no matter the actual application and project. This includes
    training, evaluation, checkpointing, restarting training, ... the internals
    can be modified from the configs.

    This class should be subclassed by your specific method (see examples).

    Execution flow when calling `launch`:

    - self.before_fit (nothing by default)
    - self.fit (executes all the training/intermitent evaluation by default)
      - for `self.optim["epochs"]` epochs:
        - self.fit_epoch (one training epoch by default)
          - self.before_fit_epoch (setup in train mode)
          - loop over mini-batches
            - self.before_fit_step (moves data to device)
            - self.fit_step
            - self.after_fit_step (nothing by default)
          - self.after_fit_epoch
        - self.evaluate (if asked by user config, looping over all non train datasets)
          - self.before_eval (setup in eval mode)
          - loop over mini-batches
            - self.before_eval_step (moves data to device)
            - self.eval_step
            - self.after_eval_step (nothing by default)
          - self.after_eval
        - save intermitent checkpoint if asked by user config
      - save final checkpoint if asked by user config
    - self.after_fit (evaluates by default)

    Parameters
    ----------
    data: dict
        Data mapper of name->mini-batch. The `train` name is used for training.
        Any other name is used for validation.
    modules: dict
        Modules (NNs) configuration.
    objective: dict
        Objective configuration.
    hardware: dict
        Hardware configuration.
    optim: dict
        Optimizer configuration.
    logger: dict
        Logger configuration.
    """

    def __init__(
        self,
        data,
        modules,
        objective,
        hardware,
        optim,
        logger,
    ):
        super().__init__()
        logging.info(f"=> INIT OF {self.__class__.__name__} STARTED")
        self._data = data
        self._modules = modules
        self._objective = objective
        self._hardware = hardware
        self._optim = optim
        self._logger = logger
        self.set_logger_defaults(self._logger)
        self.set_optim_defaults(self._optim)
        c = self.get_config()
        c["trainer"]["logger"] = self._logger
        c["trainer"]["optim"] = self._optim

        # dumps to file with defaults:
        with open(self._logger["dump_path"] / ".hydra" / "config.yaml", "w") as f:
            omegaconf.OmegaConf.save(c, f)

        logging.info(f"=> INIT OF {self.__class__.__name__} COMPLETED")

    def instanciate(self):
        seed_everything(self._hardware.get("seed", None))

        self.start_time = time.time()
        # we skip optim as we may not need it (see below)
        self.data = hydra.utils.instantiate(self._data, _convert_="object")
        self.modules = hydra.utils.instantiate(self._modules, _convert_="object")
        self.objective = hydra.utils.instantiate(self._objective, _convert_="object")
        self.hardware = hydra.utils.instantiate(self._hardware, _convert_="object")
        self.logger = hydra.utils.instantiate(self._logger, _convert_="object")

        self._set_device(self.hardware)

        self.objective = self.objective.to(self._device)

        logging.info("Logger:")
        logging.info(f"\t- Dump path: `{self.logger['dump_path']}`")
        if self.logger["wandb"]:
            logging.info("\t- Wandb:")
            try:
                wandb.init(
                    entity=self.logger["wandb"]["entity"],
                    project=self.logger["wandb"]["project"],
                    name=self.logger["wandb"]["name"],
                    dir=str(self.logger["dump_path"]),
                    resume="allow",
                )
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
        for m in self.logger["metrics"]:
            if type(self.logger["metrics"][m]) is dict:
                self.logger["metrics"][m] = torch.nn.ModuleDict(
                    self.logger["metrics"][m]
                )
        if type(self.logger["metrics"]) is dict:
            self.logger["metrics"] = torch.nn.ModuleDict(self.logger["metrics"])
        self.logger["metrics"] = self.logger["metrics"].to(self._device)

        # Data
        logging.info("Data:")
        for name, loader in self.data.items():
            if name[0] == "_":
                logging.info(f"\t- `{name}` ignored (starts with `_`).")
                continue
            logging.info(f"\t\t- length: {len(loader)}.")
            if name in self.logger["metrics"]:
                logging.info("\t\t- metrics:")
                for mname in self.logger["metrics"][name]:
                    logging.info(f"\t\t\t- {mname}.")
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
                    logging.warn(
                        "Custom sampler with distributed version is not supported"
                    )
                    raise ValueError("ERROR")
                self.data[name] = DistributedSamplerWrapper(
                    loader, self.world_size, self.rank
                )
                logging.info(
                    f"\t- Length after DDS on this process `{len(self.data[name])}."
                )

        # Modules and scaler
        logging.info("Modules:")
        for name, module in self.modules.items():
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
            self.modules[name] = module
            trainable = sum(
                param.numel() for param in module.parameters() if param.requires_grad
            )
            logging.info(f"\t- {name} with {trainable} trainable parameters.")
        self.modules = torch.nn.ModuleDict(self.modules)
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

    def setup(self):
        logging.getLogger().setLevel(self._logger["level"])
        logging.info(f"=> SETUP OF {self.__class__.__name__} STARTED")
        self.instanciate()
        self.load_checkpoint()
        logging.info(f"=> SETUP OF {self.__class__.__name__} COMPLETED")

    def set_logger_defaults(self, logger):
        logger["dump_path"] = logger.get(
            "dump_path", Path(HydraConfig.get().runtime.output_dir)
        )
        logger["wandb"] = logger.get("wandb", None)
        if type(logger["wandb"]) is bool and logger["wandb"] is True:
            logger["wandb"] = {}
        if logger["wandb"] is not None and self.rank == 0:
            logger["wandb"]["entity"] = logger["wandb"].get("entity", None)
            logger["wandb"]["project"] = logger["wandb"].get("project", None)
            logger["wandb"]["name"] = logger["wandb"].get("name", None)
            logger["wandb"]["id"] = logger["wandb"].get("id", None)

        logger["level"] = logger.get("level", 20)
        logger["metrics"] = logger.get("metrics", {})
        logger["save_final_model"] = logger.get("save_final_model", "final")
        logger["eval_every_epoch"] = logger.get("eval_every_epoch", 1)
        logger["every_step"] = logger.get("every_step", 1)
        logger["checkpoint_frequency"] = logger.get("checkpoint_frequency", 10)
        logger["checkpoint_model_only"] = logger.get("checkpoint_model_only", True)

    @staticmethod
    def set_optim_defaults(optim):
        optim["accumulation_steps"] = optim.get("accumulation_steps", 1)
        optim["grad_max_norm"] = optim.get("grad_max_norm", None)

    def forward(self):
        return self.modules["backbone"](self.batch[0])

    def predict(self):
        return self.forward()

    def compute_loss(self):
        return self.objective(self.predict(), self.batch[1])

    def __call__(self):
        self.setup()
        self.launch()

    def launch(self):
        """Routine that is launchd after the class is initialized.

        This will commonly consist of training + evaluation.
        Can be customized by the user to fit the use-cases.
        This is just a boilerplate version that provides minimal things.
        """
        if not "train" in self.data:
            self.evaluate()
            self.cleanup()
            return
        try:
            self.before_fit()
            self.fit()
            self.after_fit()
        except BreakAllEpochs:
            logging.exception("Training stopped by user.")
            raise
        if self.logger["wandb"]:
            wandb.finish()
        self.cleanup()

    def fit(self):
        while self.epoch < self.optim["epochs"]:
            try:
                self.fit_epoch()
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
                self.evaluate()
            self.epoch = self.epoch + 1

            if self.epoch % self.logger["checkpoint_frequency"] == 0:
                self.save_checkpoint(
                    f"checkpoint_{self.epoch}.ckpt",
                    model_only=self.logger["checkpoint_model_only"],
                )

        # At the end of training, we (optionally) save the final model.
        if self.logger["save_final_model"]:
            self.save_checkpoint(
                f"{self.logger['save_final_model']}.ckpt", model_only=True
            )

        logging.info("Cleaning up any temporary checkpoint.")
        (self.logger["dump_path"] / "tmp_checkpoint.ckpt").unlink(missing_ok=True)

    def fit_epoch(self):
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
            self.fit_step()
            self.after_fit_step()

            self.global_step.add_(1)
            if self.batch_idx >= max_steps:
                break

        self.after_fit_epoch()
        self.batch = None

    def evaluate(self) -> dict:

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
            if name_loader in self.logger["metrics"]:
                for _, v in self.logger["metrics"][name_loader].items():
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
                            self.eval_step(name_loader=name_loader)

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
            if name_loader in self.logger["metrics"]:
                for name, metric in self.logger["metrics"][name_loader].items():
                    packet[f"{name_loader}/{name}"] = metric.compute()
        self.log(packet, commit=True)
        # Call any user specified post-epoch function.
        self.after_eval()

    def fit_step(self):

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

        if self.global_step % self.logger["every_step"] == 0:
            bucket = {}
            if isinstance(returned_loss, dict):
                for name, value in returned_loss.items():
                    bucket[f"train/{name}"] = value.item()
            else:
                bucket["train/loss"] = loss.item()
            bucket["train/lr"] = self.optim["scheduler"].get_last_lr()[0]
            bucket["step"] = self.batch_idx
            bucket["epoch"] = self.epoch
            self.log(bucket, commit=True)

    def eval_step(self, name_loader):
        output = self.predict()
        if name_loader in self.logger["metrics"]:
            for metric in self.logger["metrics"][name_loader].values():
                metric.update(output, self.batch[1])

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

    def checkpoint(self) -> submitit.helpers.DelayedSubmission:
        # the checkpoint method is called asynchroneously when the slurm manager
        # sends a preemption signal, with the same arguments as the __call__ method
        # "self" is your callable, at its current state.
        # "self" therefore holds the current version of the model:
        logging.info("Requeuing the task...")
        self.save_checkpoint("tmp_checkpoint.ckpt", model_only=False)
        model = type(self)(
            self._data,
            self._modules,
            self._objective,
            self._hardware,
            self._optim,
            self._logger,
        )
        logging.info("Cleaning up the current task before submitting a new one.")
        self.cleanup()
        return submitit.helpers.DelayedSubmission(model)

    def log(self, packet=None, commit=True):
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
            with jsonlines.open(
                self.logger["dump_path"] / f"logs_rank_{self.rank}.jsonl", mode="a"
            ) as writer:
                writer.write(self._log_buffer)

        # Clear the log buffer.
        self._log_buffer = {}

    def load_checkpoint(self):
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

    def save_checkpoint(self, name, model_only):
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

    def generate_logging_default_bucket(self):
        cur_time = time.time()
        rel_time = cur_time - self.start_time
        if self.world_size > 1:
            rank = torch.distributed.get_rank()
        else:
            rank = 0
        bucket = {
            "rank": rank,
            "timestamp": cur_time,
            "relative_time": rel_time,
            "training": self.training,
            "epoch": self.epoch,
            "step": self.batch_idx,
        }
        return bucket

    def cleanup(self):
        logging.info("Cleaning up process, device status before cleaning:")
        get_gpu_info()

        if torch.distributed.is_initialized():
            logging.info("Cleaning distributed processes.")
            torch.distributed.destroy_process_group()

        gc.collect()
        torch.cuda.empty_cache()

        logging.info("Device status after cleaning.")
        get_gpu_info()

    def get_logs(self, keys=None):
        if self.logger["wandb"] is None:
            return reader.jsonl(self.logger["dump_path"])
        else:
            return reader.wandb(
                self.logger["wandb"]["entity"],
                self.logger["wandb"]["project"],
                self.logger["wandb"]["id"],
                keys=keys,
            )

    def get_config(self):
        # Load the config file.
        config = omegaconf.OmegaConf.load(
            self._logger["dump_path"] / ".hydra" / "config.yaml"
        )
        return config

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
    def batch_idx(self):
        if not hasattr(self, "_batch_idx"):
            return None
        return self._batch_idx

    @property
    def device(self):
        return self._device

    @epoch.setter
    def epoch(self, value):
        self._epoch = value

    @step.setter
    def step(self, value):
        self._step = value

    def before_fit(self):
        self.epoch = 0

    def after_fit(self):
        self.evaluate()

    def before_fit_epoch(self):
        self.train()
        if self.world_size > 1:
            self.data["train"].set_epoch(self.epoch)

    def after_fit_epoch(self):
        pass

    def before_fit_step(self):
        # set up the data to have easy access throughout the methods
        self.batch = to_device(self.batch, self.device)

    def after_fit_step(self):
        pass

    def before_eval(self):
        self.eval()

    def after_eval(self):
        pass

    def before_eval_step(self):
        self.batch = to_device(self.batch, self.device)

    def after_eval_step(self):
        pass


class JointEmbedding(BaseModel):
    r"""Base class for training a joint-embedding SSL model."""

    def format_views_labels(self):
        if (
            len(self.batch) == 2
            and torch.is_tensor(self.batch[1])
            and not torch.is_tensor(self.batch[0])
        ):
            # we assume the second element are the labels
            views, labels = self.batch
        elif (
            len(self.batch) > 1
            and all([torch.is_tensor(b) for b in self.batch])
            and len(set([b.ndim for b in self.batch])) == 1
        ):
            # we assume all elements are views
            views = self.batch
            labels = None
        else:
            msg = """You are using the JointEmbedding class with only 1 view! 
            Make sure to double check your config and datasets definition. 
            Most methods expect 2 views, some can use more."""
            log_and_raise(ValueError, msg)
        return views, labels

    def predict(self):
        return self.modules["backbone_classifier"](self.forward())

    def compute_loss(self):
        views, labels = self.format_views_labels()
        embeddings = [self.modules["backbone"](view) for view in views]
        projections = [self.modules["projector"](embed) for embed in embeddings]

        loss_ssl = self.objective(*projections)

        # classifiers, but only if given labels
        if labels is not None:
            loss_backbone_classifier = 0
            loss_proj_classifier = 0
            for embed, proj in zip(embeddings, projections):
                loss_backbone_classifier += F.cross_entropy(
                    self.modules["backbone_classifier"](embed.detach()), labels
                )
                loss_proj_classifier += F.cross_entropy(
                    self.modules["projector_classifier"](proj.detach()), labels
                )
        else:
            loss_backbone_classifier = 0
            loss_proj_classifier = 0
        return {
            "train/loss_ssl": loss_ssl,
            "train/loss_backbone_classifier": loss_backbone_classifier,
            "train/loss_projector_classifier": loss_proj_classifier,
        }


class SelfDistillation(JointEmbedding):
    r"""Base class for training a self-distillation SSL model."""

    def setup(self):
        logging.getLogger().setLevel(self._logger["level"])
        logging.info(f"=> SETUP OF {self.__class__.__name__} STARTED")
        self.instanciate()
        self.modules["backbone_target"] = copy.deepcopy(self.modules["backbone"])
        self.modules["projector_target"] = copy.deepcopy(self.modules["projector"])

        self.modules["backbone_target"].requires_grad_(False)
        self.modules["projector_target"].requires_grad_(False)
        self.load_checkpoint()
        logging.info(f"=> SETUP OF {self.__class__.__name__} COMPLETED")

    def before_fit_step(self):
        """Update the target parameters as EMA of the online model parameters."""
        update_momentum(
            self.backbone, self.backbone_target, m=self.config.model.momentum
        )
        update_momentum(
            self.projector, self.projector_target, m=self.config.model.momentum
        )
