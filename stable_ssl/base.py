# -*- coding: utf-8 -*-
"""Base class for training a model."""
#
# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#         Randall Balestriero <randallbalestriero@gmail.com>
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
import logging
import time
import copy
import dataclasses
import gc
import numpy as np
import submitit
import jsonlines
import omegaconf
from pathlib import Path
from tqdm import tqdm
import torch
import subprocess
import os
from typing_extensions import override

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
    BreakStep,
    seed_everything,
    to_device,
    get_gpu_info,
    log_and_raise,
)

__all__ = ["BreakStep"]


class _DatasetSamplerWrapper(torch.utils.data.Dataset):
    """Dataset to create indexes from `Sampler` or `Iterable`."""

    def __init__(self, sampler) -> None:
        self._sampler = sampler
        # defer materializing an iterator until it is necessary
        self._sampler_list = None

    @override
    def __getitem__(self, index: int):
        if self._sampler_list is None:
            self._sampler_list = list(self._sampler)
        return self._sampler_list[index]

    def __len__(self) -> int:
        return len(self._sampler)

    def reset(self) -> None:
        """Reset the sampler list in order to get new sampling."""
        self._sampler_list = list(self._sampler)


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


class DistributedSamplerWrapper(torch.utils.data.DistributedSampler):
    def __init__(self, sampler, *args, **kwargs) -> None:
        super().__init__(_DatasetSamplerWrapper(sampler), *args, **kwargs)

    @override
    def __iter__(self):
        self.dataset.reset()
        return (self.dataset[index] for index in super().__iter__())


class BaseModel(torch.nn.Module):
    r"""Base class for training a model.

    Parameters
    ----------
    config : TrainerConfig
        Parameters for BaseModel organized in groups.
        For details, see the `TrainerConfig` class in `config.py`.
    """

    def __init__(
        self,
        data,
        networks,
        objective,
        train_on,
        hardware,
        optim,
        log,
        eval_only,
        metrics=None,
        seed=None,
        save_final_model=None,
        checkpoint_frequency=1,
        eval_every_epoch=1,
    ):
        super().__init__()
        # FIXME: wandb logging
        self.use_wandb = False
        self._train_on = train_on
        self._eval_only = eval_only
        self._log_every_step = log["every_step"]
        self.save_final_model = save_final_model
        self.checkpoint_frequency = checkpoint_frequency
        self.eval_every_epoch = eval_every_epoch
        seed_everything(seed)
        self._set_device(hardware)
        self.hardware = hardware
        self.dump_path = Path(HydraConfig.get().runtime.output_dir)
        logging.info(f"\t=> Dump path: `{self.dump_path}`")

        self.objective = objective.to(self._device)

        # Data
        self.data = data
        for name, loader in self.data.items():
            logging.info(f"\t=> Found dataloader `{name}` with length {len(loader)}.")
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
                f"\t=> New length after DDS on local process `{len(self.data[name])}."
            )

        if not eval_only and train_on not in self.data:
            log_and_raise(ValueError, f"Training data ({train_on}) not in {self.data}.")

        # Modules and scaler
        for name, module in networks.items():
            # if self.config.model.memory_format == "channels_last":
            #     module.to(memory_format=torch.channels_last)
            if self.world_size > 1:
                module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module)
            module.to(self.this_device)
            has_parameters = False
            if sum(p.numel() for p in module.parameters() if p.requires_grad) > 0:
                has_parameters = True
            if self.world_size > 1 and has_parameters:
                module = torch.nn.parallel.DistributedDataParallel(
                    module, device_ids=[self._device]
                )
            networks[name] = module
            trainable = sum(
                param.numel() for param in module.parameters() if param.requires_grad
            )
            logging.info(
                f"\t=> Found module '{name}' with {trainable} trainable parameters."
            )
        self.networks = torch.nn.ModuleDict(networks)
        self.scaler = torch.amp.GradScaler("cuda", enabled=hardware["float16"])

        if not eval_only:
            logging.info("Setting up self.optim.")
            optim["optimizer"] = optim["optimizer"](self.parameters())
            optim["scheduler"] = optim["scheduler"](optim["optimizer"])
            self.optim = optim
        else:
            logging.info(
                "Mode is eval_only, skipping optimizer and scheduler initializations."
            )

        self.register_buffer("global_step", torch.zeros((1,), dtype=int))
        self.metrics = metrics or {}
        logging.info(f"\t=> Found metric(s) `{self.metrics}`.")
        for m in self.metrics:
            for k in self.metrics[m]:
                self.metrics[m][k] = self.metrics[m][k].to(self._device)

        logging.getLogger().setLevel(log["level"])
        self._log_buffer = {}

        logging.info("\tCalling load_checkpoint() method.")
        self.load_checkpoint()

    @abstractmethod
    def forward(self):
        """Define the forward pass of the model."""
        pass

    @abstractmethod
    def compute_loss(self):
        """Compute the loss for the current batch."""
        pass

    def __call__(self):
        seed_everything(self.config.hardware.seed)

        # Use WandB if an entity or project name is provided.
        self.use_wandb = (
            (self.config.log.api is not None)
            and (self.config.log.api.lower() == "wandb")
            and (self.rank == self.config.log.rank_to_log)
        )

        if self.use_wandb:
            logging.info(f"\t=> Initializing wandb for logging in {self.dump_path}.")

            try:
                wandb.init(
                    entity=self.config.log.entity,
                    project=self.config.log.project,
                    config=dataclasses.asdict(self.config),
                    name=self.config.log.run,
                    dir=str(self.dump_path),
                    resume="allow",
                )
            except Exception:
                logging.exception("Failed to initialize wandb.")
                raise

        logging.info(f"\t=> Dumping config file in {self.dump_path}.")
        try:
            omegaconf.OmegaConf.save(self.config, self.dump_path / "hparams.yaml")
        except Exception:
            logging.exception("Failed to dump config file.")
            raise

        self.start_time = time.time()
        self.execute()

    def execute(self):
        """Routine that is executed after the class is initialized.

        This will commonly consist of training + evaluation.
        Can be customized by the user to fit the use-cases.
        This is just a boilerplate version that provides minimal things.
        """
        if self._eval_only:
            self.evaluate()
        else:
            try:
                self.before_train()
                self.fit()
                self.after_train()
                self.evaluate()  # always eval the model after training
            except BreakAllEpochs:
                logging.exception("Training stopped by user.")
                raise
            if self.use_wandb:
                wandb.finish()
            self.cleanup()

    def fit(self):
        while self.epoch < self.optim["epochs"]:
            try:
                self.train_epoch()
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

            if self.epoch % self.eval_every_epoch == 0:
                self.evaluate()
            self.epoch = self.epoch + 1

            if self.epoch % self.checkpoint_frequency == 0:
                self.save_checkpoint("tmp_checkpoint.ckpt", model_only=False)

        # At the end of training, we (optionally) save the final model.
        if self.save_final_model:
            self.save_checkpoint(f"{self.save_final_model}.ckpt", model_only=True)

        logging.info("Cleaning up any temporary checkpoint.")
        (self.dump_path / "tmp_checkpoint.ckpt").unlink(missing_ok=True)

    def train_epoch(self):
        self.before_train_epoch()
        # We do not ensure that the model is still in train mode to not
        # override any user desired behavior, simply speak out.
        if not self.training:
            logging.warning(
                "Starting training epoch but model is no longer in "
                "train mode after call to before_train_epoch()."
            )

        loader = self.data[self.train_on]
        # If max_steps is negative, train on the full dataset.
        if self.optim["max_steps"] < 0:
            max_steps = len(loader)
        # If max_steps is a float between 0 and 1, treat it as a percentage.
        elif 0 < self.optim["max_steps"] < 1:
            max_steps = int(self.optim["max_steps"] * len(loader))
        # Otherwise, set max_steps to the length of the dataset.
        else:
            max_steps = min(self.optim["max_steps"], len(loader))

        for i, data in enumerate(
            tqdm(loader, total=max_steps, desc=f"Training: {self.epoch}")
        ):
            # set up the data to have easy access throughout the methods
            self.batch_idx = i
            self.batch = to_device(data, self.this_device)

            self.before_train_step()
            self.train_step()
            self.after_train_step()

            self.global_step.add_(1)
            if i >= max_steps:
                break

        self.after_train_epoch()
        self.batch = None

    def evaluate(self) -> dict:

        if set(self.data) == set([self.train_on]):
            logging.info("No val_loader hence skipping eval epoch.")
            return

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
            if name_loader == self.train_on:
                continue
            # Reset the metrics for the epoch.
            for _, v in self.metrics.get(name_loader, {}).items():
                v.reset()

            try:
                max_steps = len(loader)
                with torch.inference_mode():
                    for step, data in tqdm(
                        enumerate(loader),
                        total=max_steps,
                        desc=f"Eval {name_loader}: {self.epoch=}",
                    ):
                        self.batch_idx = step
                        self.batch = to_device(data, self.this_device)

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
            for name, metric in self.metrics.get(name_loader, {}).items():
                packet[f"{name_loader}/{name}"] = metric.compute()
        self.log(packet, commit=True)
        # Call any user specified post-epoch function.
        self.after_eval()

    def train_step(self):
        self.optim["optimizer"].zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=self.hardware["float16"]):
            loss = self.compute_loss()

        if np.isnan(loss.item()):
            log_and_raise(NanError, "Loss is NaN. Stopping training.")

        self.scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place.
        self.scaler.unscale_(self.optim["optimizer"])
        if self.optim.get("grad_max_norm", None) is not None:
            # Since the gradients of optimizer's assigned params are unscaled,
            # clips as usual:
            torch.nn.utils.clip_grad_norm_(
                self.parameters(), self.optim.get("grad_max_norm")
            )
        self.scaler.step(self.optim["optimizer"])
        self.scaler.update()

        self.optim["scheduler"].step()

        if self.global_step % self._log_every_step == 0:
            bucket = {}
            bucket["train/loss"] = loss.item()
            bucket["train/lr"] = self.optim["scheduler"].get_last_lr()[0]
            bucket["step"] = self.batch_idx
            bucket["epoch"] = self.epoch
            self.log(bucket, commit=True)

    def eval_step(self, name_loader):
        output = self.forward(self.batch[0])
        for metric in self.metrics[name_loader].values():
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

        # cleanup the device
        logging.info("Device status at start of process:")
        get_gpu_info()

        # Set the CUDA device.
        torch.cuda.set_device(self._device)

    def checkpoint(self):
        # the checkpoint method is called asynchroneously when the slurm manager
        # sends a preemption signal, with the same arguments as the __call__ method
        # "self" is your callable, at its current state.
        # "self" therefore holds the current version of the model:
        logging.info("Requeuing the task.")
        config = copy.deepcopy(self.config)
        model = type(self)(config)
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
        if self.use_wandb:
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
                self.dump_path / f"logs_rank_{self.rank}.jsonl", mode="a"
            ) as writer:
                writer.write(self._log_buffer)

        # Clear the log buffer.
        self._log_buffer = {}

    def load_checkpoint(self):
        if not (self.dump_path / "tmp_checkpoint.ckpt").is_file():
            logging.info(
                f"\t=> `{self.dump_path / 'tmp_checkpoint.ckpt'}` does not exist "
            )
            logging.info("\t=> training from scratch...")
            self.epoch = 0
            return

        logging.info(
            f"\t=> folder `{self.dump_path}/tmp_checkpoint.ckpt` exists "
            "\n\t=> loading it."
        )
        checkpoint = self.dump_path / "tmp_checkpoint.ckpt"

        ckpt = torch.load(checkpoint, map_location="cpu")

        for name, model in self.named_children():
            if name not in ckpt:
                logging.info(f"\t\t=> {name} not in ckpt, skipping.")
                continue
            model.load_state_dict(ckpt[name])
            logging.info(f"\t\t=> {name} successfully loaded.")
        if "optimizer" in ckpt:
            self.optim["optimizer"].load_state_dict(ckpt["optimizer"])
            logging.info("\t\t=> optimizer successfully loaded.")
        if "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])
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
        saving_name = self.dump_path / name
        state = {}
        for subname, model in self.named_children():
            state[subname] = model.state_dict()
        if model_only:
            torch.save(state, saving_name)
            logging.info(f"Model saved at {saving_name}.")
            return
        if hasattr(self, "optimizer"):
            state["optimizer"] = self.optim["optimizer"].state_dict()
        if hasattr(self, "scheduler"):
            state["scheduler"] = self.scheduler.state_dict()
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
    def train_on(self):
        return self._train_on

    @property
    def epoch(self):
        if not hasattr(self, "_epoch"):
            return None
        return self._epoch

    @property
    def config(self):
        return self._config

    @property
    def step(self):
        if not hasattr(self, "_step"):
            return None
        return self._step

    @property
    def data(self):
        return self._data

    @property
    def this_device(self):
        return self._device

    @epoch.setter
    def epoch(self, value):
        self._epoch = value

    @step.setter
    def step(self, value):
        self._step = value

    @data.setter
    def data(self, value):
        self._data = value

    def before_train(self):
        self.epoch = 0

    def after_train(self):
        pass

    def before_train_epoch(self):
        self.train()
        if self.world_size > 1:
            self.data[self._train_on].set_epoch(self.epoch)

    def after_train_epoch(self):
        pass

    def before_train_step(self):
        pass

    def after_train_step(self):
        pass

    def before_eval(self):
        self.eval()

    def after_eval(self):
        pass

    def before_eval_step(self):
        pass

    def after_eval_step(self):
        pass
