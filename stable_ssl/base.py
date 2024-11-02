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
import numpy as np
import submitit
import jsonlines
import omegaconf
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass, make_dataclass
from torchmetrics.classification import MulticlassAccuracy
import os

from .reader import jsonl_run


try:
    import wandb
except ModuleNotFoundError:
    logging.warning(
        "Wandb module is not installed, make sure not to use wandb for logging "
        "or an error will be thrown."
    )

import torch

from .utils import (
    BreakAllEpochs,
    BreakEpoch,
    NanError,
    BreakStep,
    seed_everything,
    setup_distributed,
    FullGatherLayer,
    LARS,
    LinearWarmupCosineAnnealing,
    to_device,
    get_gpu_info,
)


@dataclass
class BaseModelConfig:
    """Base configuration for the 'model' parameters.

    Parameters
    ----------
    model : str
        Type of model to use. Default is "Supervised".
    backbone_model : str
        Neural network architecture to use for the backbone. Default is "resnet9".
    sync_batchnorm : bool, optional
        Whether to use synchronized batch normalization. Default is False.
    memory_format : str, optional
        Memory format for tensors (e.g., "channels_last"). Default is "channels_last".
    pretrained : bool, optional
        Whether to use the torchvision pretrained weights or use random initialization.
    with_classifier : bool, optional
        Whether to keep the last layer(s) of the backbone (classifier)
        when loading the model. Default is True.
    """

    name: str = "Supervised"
    backbone_model: str = "resnet18"
    sync_batchnorm: bool = False
    memory_format: str = "channels_last"
    pretrained: bool = False
    with_classifier: bool = True


class BaseModel(torch.nn.Module):
    r"""Base class for training a model.

    Parameters
    ----------
    config : TrainerConfig
        Parameters for BaseModel organized in groups.
        For details, see the `TrainerConfig` class in `config.py`.
    """

    def __new__(cls, config, *args, **kwargs):
        if len(args):
            raise ValueError(
                "You should only provide named arguments to ensure they are "
                "logged in the config."
            )
        trainer = super(BaseModel, cls).__new__(cls)
        config.__class__ = make_dataclass(
            "TrainerConfig",
            fields=[(name, type(v), v) for name, v in kwargs.items()],
            bases=(type(config),),
        )
        trainer._config = copy.deepcopy(config)
        return trainer

    def __init__(self, config, *args, **kwargs):
        self._set_device()
        super().__init__()

    @abstractmethod
    def initialize_modules(self):
        """Initialize the modules required for the model."""
        pass

    @abstractmethod
    def forward(self):
        """Define the forward pass of the model."""
        pass

    @abstractmethod
    def compute_loss(self):
        """Compute the loss for the current batch."""
        pass

    def __call__(self):

        logging.basicConfig(
            level=self.config.log.level, format="[stable-SSL] %(message)s"
        )
        get_gpu_info()
        seed_everything(self.config.hardware.seed)

        # Use WandB if an entity or project name is provided.
        self.use_wandb = bool(
            self.config.log.wandb_entity or self.config.log.wandb_project
        )

        if self.use_wandb:
            logging.info(
                f"\t=> Initializating wandb for logging in {self.config.log.dump_path}."
            )
            if os.environ.get("HOME") is None:
                os.environ["HOME"] = "/users/hvanasse"
            wandb.init(
                entity=self.config.log.wandb_entity,
                project=self.config.log.wandb_project,
                config=dataclasses.asdict(self.config),
                name=self.config.log.run,
                dir=str(self.config.log.dump_path),
                resume="allow",
            )
        else:
            logging.info(f"\t=> Dumping config file in {self.config.log.dump_path}")
            omegaconf.OmegaConf.save(
                self.config, self.config.log.dump_path / "hparams.yaml"
            )

        self.scaler = torch.amp.GradScaler("cuda", enabled=self.config.hardware.float16)

        # Set up the dataloaders.
        logging.info("Creating dataloaders.")
        dataloaders = self.config.data.get_dataloaders(
            world_size=self.config.hardware.world_size
        )
        for name, loader in dataloaders.items():
            logging.info(f"\t=> Found dataloader `{name}` with length {len(loader)}.")
        if self.config.log.eval_only:
            for name in dataloaders:
                if name in self.config.data.train_on:
                    logging.info(f"\t=> `{name}` will be ignored (eval_only=True).")
        else:
            assert len(self.config.data.train_on)
            if self.config.data.train_on not in dataloaders:
                raise RuntimeError(f"eval_only=False and `{name}` not given.")
        self.dataloaders = dataloaders

        # Set up the model's modules. Should be implemented by the child class.
        logging.info("Calling initialize_modules() method.")
        self.initialize_modules()

        # Set up fthe metrics. Should be implemented by the child class.
        if hasattr(self, "metrics"):
            raise RuntimeError(
                "You can't assign any value to `self.metrics`, this will be "
                "used for metrics only."
            )
        self.initialize_metrics()
        if not hasattr(self, "metrics"):
            raise RuntimeError(
                "The `initialize_metrics` method should create a `self.metrics` "
                "ModuleDict object."
            )
        if not isinstance(self.metrics, torch.nn.ModuleDict):
            raise RuntimeError("The `self.metrics` should be a ModuleDict.")
        self._log_buffer = {}
        self.register_buffer("global_step", torch.zeros((1,), dtype=int))

        for name, module in self.named_children():
            if self.config.model.memory_format == "channels_last":
                module.to(memory_format=torch.channels_last)
            if self.config.model.sync_batchnorm:
                module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module)
            module.to(self.this_device)
            has_parameters = False
            if sum(p.numel() for p in module.parameters() if p.requires_grad) > 0:
                has_parameters = True
            if self.config.hardware.world_size > 1 and has_parameters:
                module = torch.nn.parallel.DistributedDataParallel(
                    module, device_ids=[self.config.hardware.gpu_id]
                )
            setattr(self, name, module)

            trainable = sum(
                param.numel() for param in module.parameters() if param.requires_grad
            )
            logging.info(
                f"\t=> Found module '{name}' with {trainable} trainable parameters."
            )

        if not self.config.log.eval_only:
            logging.info("Calling _initialize_optimizer() method.")
            self.optimizer = self._initialize_optimizer()
            logging.info("Calling _initialize_scheduler() method.")
            try:
                self.scheduler = self._initialize_scheduler()
            except NotImplementedError:
                logging.info("No scheduler given.")
        else:
            logging.info(
                "Mode is eval_only, skipping optimizer and "
                "scheduler initializations."
            )

        logging.info("Calling _load_checkpoint() method.")
        self._load_checkpoint()
        self.start_time = time.time()
        self.execute()

    def execute(self):
        """Routine that is executed after the class is initialized.

        This will commonly consist of training + evaluation.
        Can be customized by the user to fit the use-cases.
        This is just a boilerplate version that provides minimal things.
        """
        if self.config.log.eval_only:
            self.eval_epoch()
        else:
            try:
                self.before_train_all_epochs()
                self._train_all_epochs()
                self.after_train_all_epochs()
                self.eval_epoch()  # always eval the model after training
            except BreakAllEpochs:
                logging.exception("Exception during training (self.evaluate).")
                raise
            if self.use_wandb:
                wandb.finish()
            self.cleanup()

    def initialize_metrics(self):
        nc = self.config.data.datasets[self.config.data.train_on].num_classes
        train_acc1 = MulticlassAccuracy(num_classes=nc, top_k=1)

        # Initialize the metrics dictionary with the train metric.
        self.metrics = torch.nn.ModuleDict({"train/acc1": train_acc1})

        # Add unique evaluation metrics for each eval dataset.
        name_eval_loaders = set(self.dataloaders.keys()) - set(
            [self.config.data.train_on]
        )
        for name_loader in name_eval_loaders:
            self.metrics.update(
                {
                    f"eval/{name_loader}/acc1": MulticlassAccuracy(
                        num_classes=nc, top_k=1
                    ),
                    f"eval/{name_loader}/acc5": MulticlassAccuracy(
                        num_classes=nc, top_k=5
                    ),
                    f"eval/{name_loader}/acc1_by_class": MulticlassAccuracy(
                        num_classes=nc, average="none", top_k=1
                    ),
                    f"eval/{name_loader}/acc5_by_class": MulticlassAccuracy(
                        num_classes=nc, average="none", top_k=5
                    ),
                }
            )

    def _train_all_epochs(self):
        while self.epoch < self.config.optim.epochs:

            self.config.data.set_epoch_train_sampler(self.epoch)

            try:
                self._train_epoch()
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

            if self.epoch % self.config.log.eval_epoch_freq == 0:
                self.eval_epoch()
            self.epoch = self.epoch + 1

            freq = self.config.log.checkpoint_frequency
            if self.epoch % freq == 0:
                logging.info("Checkpointing everything to restart if needed.")
                self.save_checkpoint("tmp_checkpoint.ckpt", model_only=False)

        # At the end of training, we (optionally) save the final model.
        if self.config.log.save_final_model:
            self.save_checkpoint(
                f"{self.config.log.final_model_name}.ckpt", model_only=True
            )
        # Remove any temporary checkpoint.
        (self.config.log.dump_path / "tmp_checkpoint.ckpt").unlink(missing_ok=True)

    def _train_epoch(self):

        self.train()  # hierarchically set up all modules in train mode
        self.before_train_epoch()

        # We do not ensure that the model is still in train mode to not
        # override any user desired behavior, simply speak out.
        if not self.training:
            logging.warning(
                "Starting training epoch but model is no longer in "
                "train mode after call to before_train_epoch()."
            )

        # If max_steps is negative, train on the full dataset.
        if self.config.optim.max_steps < 0:
            max_steps = len(self.dataloaders[self.config.data.train_on])
        # If max_steps is a float between 0 and 1, treat it as a percentage.
        elif 0 < self.config.optim.max_steps < 1:
            max_steps = int(
                self.config.optim.max_steps
                * len(self.dataloaders[self.config.data.train_on])
            )
        # Otherwise, set max_steps to the length of the dataset.
        else:
            max_steps = min(max_steps, len(self.dataloaders[self.config.data.train_on]))

        for batch_idx, data in enumerate(
            tqdm(
                self.dataloaders[self.config.data.train_on],
                total=max_steps,
                desc=f"Training: {self.epoch=}",
            )
        ):
            # set up the data to have easy access throughout the methods
            self.batch_idx = batch_idx
            self.global_step.add_(1)
            self.data = to_device(data, self.this_device)

            try:
                self.before_train_step()
                self.train_step()
                self.after_train_step()

            except BreakStep:
                logging.info("Method `train_step` has been interrupted by user.")

            if batch_idx >= max_steps:
                break

        self.after_train_epoch()

        # clean up to avoid silent bugs
        self.data = None

    def eval_epoch(self) -> dict:

        if set(self.dataloaders) == set([self.config.data.train_on]):
            logging.info("No val_loader hence skipping eval epoch.")
            return

        self.eval()  # Set model in eval mode.
        self.before_eval_epoch()
        # We do not ensure that the model is still in eval mode to not
        # override any user desired behavior.
        if self.training:
            logging.warning(
                "Starting eval epoch but model is not in "
                "eval mode after call to before_eval_epoch()."
            )

        # Reset the metrics for the epoch.
        for name, metric in self.metrics.items():
            if name.startswith("eval/"):
                metric.reset()

        for name_loader, loader in self.dataloaders.items():
            if name_loader == self.config.data.train_on:
                continue

            try:
                max_steps = len(loader)
                with torch.inference_mode():
                    for step, data in tqdm(
                        enumerate(loader),
                        total=max_steps,
                        desc=f"Eval {name_loader}: {self.epoch=}",
                    ):
                        self.batch_idx = step
                        self.data = to_device(data, self.this_device)

                        # Call any user specified pre-step function.
                        self.before_eval_step()

                        # Call the eval step.
                        with torch.amp.autocast(
                            "cuda", enabled=self.config.hardware.float16
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
            self.data = None

        # Compute the final metrics for the epoch.
        packet = {}
        for name, metric in self.metrics.items():
            if name.startswith("eval/"):
                packet[name] = metric.compute()
        self.log(packet, commit=True)

        # Call any user specified post-epoch function.
        self.after_eval_epoch()

    def train_step(self):
        self.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=self.config.hardware.float16):
            loss = self.compute_loss()

        if np.isnan(loss.item()):
            raise NanError

        self.scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place.
        self.scaler.unscale_(self.optimizer)
        if self.config.optim.grad_max_norm is not None:
            # Since the gradients of optimizer's assigned params are unscaled,
            # clips as usual:
            torch.nn.utils.clip_grad_norm_(
                self.parameters(), self.config.optim.grad_max_norm
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.scheduler.step()
        self.log(
            {
                "train/loss": loss.item(),
                "train/lr": self.scheduler.get_last_lr()[0],
                "step": self.batch_idx,
                "epoch": self.epoch,
            },
            commit=True,
        )

    def eval_step(self, name_loader):
        output = self.forward(self.data[0])
        for name, metric in self.metrics.items():
            if name.startswith(f"eval/{name_loader}/"):
                metric.update(output, self.data[1])
            elif name.startswith(f"eval/{name_loader}/"):
                self.log({name: metric(output, self.data[1])}, commit=False)
        self.log(commit=True)

    def _set_device(self):
        # Check if CUDA is available, otherwise set to CPU.
        if not torch.cuda.is_available():
            logging.warning("CUDA is not available. Setting device to CPU.")
            self._device = "cpu"
            return

        try:
            # Setup distributed hardware configuration.
            self.config.hardware = setup_distributed(self.config.hardware)
            self._device = f"cuda:{self.config.hardware.gpu_id}"
        except RuntimeError:
            # Log the error and set the device to default GPU (cuda:0) as a fallback.
            logging.exception(
                "Error setting up distributed hardware. "
                "Falling back to default GPU configuration."
            )
            self._device = "cuda:0"
            self.config.hardware.gpu_id = 0
            self.config.hardware.world_size = 1

        # Set the CUDA device.
        torch.cuda.set_device(self._device)

    def checkpoint(self):
        # the checkpoint method is called asynchroneously when the slurm manager
        # sends a preemption signal, with the same arguments as the __call__ method
        # "self" is your callable, at its current state.
        # "self" therefore holds the current version of the model:
        logging.info("Requeuing the task.")
        config = copy.deepcopy(self.config)
        config.log.add_version = False
        config.log.folder = self.config.log.dump_path.as_posix()
        model = type(self)(config)
        return submitit.helpers.DelayedSubmission(model)

    def log(self, packet=None, commit=True):
        # Check if the process should be logged.
        if self.config.log.log_process > 0 and (
            self.config.log.log_process != self.rank
        ):
            return

        # Update the log buffer with the new packet.
        packet = packet or {}
        assert "_global_step" not in packet
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

            # Add the rank suffix to the log.
            for name, value in self._log_buffer.items():
                suffix_name = f"{name}/rank_{self.rank}"
                wandb.log({suffix_name: value}, step=self.global_step.item())

        # Log in jsonl.
        else:
            with jsonlines.open(
                self.config.log.dump_path / f"logs_rank_{self.rank}.jsonl", mode="a"
            ) as writer:
                writer.write(self._log_buffer)

        # Clear the log buffer.
        self._log_buffer = {}

    def _load_checkpoint(self):
        load_from = Path(self.config.log.load_from)
        if load_from.is_file():
            logging.info(f"\t=> file {load_from} exists\n\t=> loading it.")
            checkpoint = load_from
        elif (self.config.log.dump_path / "tmp_checkpoint.ckpt").is_file():
            logging.info(
                f"\t=> folder {self.config.log.dump_path} contains "
                "`tmp_checkpoint.ckpt` file\n\t=> loading it."
            )
            checkpoint = self.config.log.dump_path / "tmp_checkpoint.ckpt"
        else:
            logging.info(f"\t=> no checkpoint at `{load_from}`")
            logging.info(
                "\t=> no checkpoint at "
                f"`{self.config.log.dump_path / 'tmp_checkpoint.ckpt'}`. "
            )
            logging.info("\t=> training from scratch...")
            self.epoch = 0
            return

        ckpt = torch.load(checkpoint, map_location="cpu")

        for name, model in self.named_children():
            if name not in ckpt:
                logging.info(f"\t\t=> {name} not in ckpt, skipping.")
                continue
            model.load_state_dict(ckpt[name])
            logging.info(f"\t\t=> {name} successfully loaded.")
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
            logging.info("\t\t=> optimizer successfully loaded.")
        if "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])
            logging.info("\t\t=> scheduler successfully loaded.")
        if "epoch" in ckpt:
            self.epoch = ckpt["epoch"]
            logging.info(f"\t\t=> training will start from epoch {ckpt['epoch']}.")
        else:
            self.epoch = 0

    def _initialize_optimizer(self):
        if self.config.optim.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.config.optim.lr,
                weight_decay=self.config.optim.weight_decay,
            )
        if self.config.optim.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.config.optim.lr,
                weight_decay=self.config.optim.weight_decay,
                betas=self.config.optim.betas,
            )
        elif self.config.optim.optimizer == "RMSprop":
            optimizer = torch.optim.RMSprop(
                self.parameters(),
                lr=self.config.optim.lr,
                weight_decay=self.config.optim.weight_decay,
                momentum=self.config.optim.momentum,
            )
        elif self.config.optim.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.config.optim.lr,
                weight_decay=self.config.optim.weight_decay,
                momentum=self.config.optim.momentum,
            )
        elif self.config.optim.optimizer == "LARS":
            optimizer = LARS(
                self.parameters(),
                lr=self.config.optim.lr,
                weight_decay=self.config.optim.weight_decay,
                momentum=self.config.optim.momentum,
            )
        return optimizer

    def _initialize_scheduler(self):
        min_lr = self.config.optim.lr * 0.005
        peak_step = 5 * len(self.dataloaders[self.config.data.train_on])
        total_steps = self.config.optim.epochs * len(
            self.dataloaders[self.config.data.train_on]
        )
        return LinearWarmupCosineAnnealing(
            self.optimizer, end_lr=min_lr, peak_step=peak_step, total_steps=total_steps
        )

    def save_checkpoint(self, name, model_only):
        if self.config.hardware.world_size > 1:
            if torch.distributed.get_rank() != 0:
                return
        saving_name = self.config.log.dump_path / name
        state = {}
        for subname, model in self.named_children():
            state[subname] = model.state_dict()
        if model_only:
            torch.save(state, saving_name)
            return
        if hasattr(self, "optimizer"):
            state["optimizer"] = self.optimizer.state_dict()
        if hasattr(self, "scheduler"):
            state["scheduler"] = self.scheduler.state_dict()
        state["epoch"] = self.epoch

        torch.save(state, saving_name)

    def generate_logging_default_bucket(self):
        cur_time = time.time()
        rel_time = cur_time - self.start_time
        if self.config.hardware.world_size > 1:
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
        if self.config.hardware.world_size > 1:
            logging.info("Cleaning distributed processes.")
            torch.distributed.destroy_process_group()
        else:
            logging.info("Not using distributed. Nothing to clean.")

    def gather(self, x):
        return FullGatherLayer.apply(x)

    @property
    def rank(self):
        if self.config.hardware.world_size > 1:
            return torch.distributed.get_rank()
        return 0

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

    def before_train_all_epochs(self):
        pass

    def after_train_all_epochs(self):
        pass

    def before_train_epoch(self):
        pass

    def after_train_epoch(self):
        pass

    def before_train_step(self):
        pass

    def after_train_step(self):
        pass

    def before_eval_epoch(self):
        pass

    def after_eval_epoch(self):
        pass

    def before_eval_step(self):
        pass

    def after_eval_step(self):
        pass

    # FIXME: to remove since this is now handled by the data config
    # def dataset_to_loader(self, dataset, train):
    #     if self.config.hardware.world_size > 1:
    #         sampler = torch.utils.data.distributed.DistributedSampler(
    #             dataset, shuffle=not train, drop_last=train
    #         )
    #         assert self.config.optim.batch_size % self.config.hardware.world_size == 0
    #         drop_last = None
    #         shuffle = None
    #     else:
    #         sampler = None
    #         drop_last = train
    #         shuffle = not train

    #     per_device_batch_size = (
    #         self.config.optim.batch_size // self.config.hardware.world_size
    #     )

    #     loader = torch.utils.data.DataLoader(
    #         dataset,
    #         batch_size=per_device_batch_size,
    #         num_workers=self.config.data.num_workers,
    #         pin_memory=True,
    #         sampler=sampler,
    #         drop_last=drop_last,
    #         shuffle=shuffle,
    #     )

    #     return loader

    # def initialize_train_loader(self):
    #     train_dataset = load_dataset(
    #         dataset_name=self.config.data.dataset,
    #         data_path=self.config.data.data_path,
    #         train=True,
    #     )

    #     return self.dataset_to_loader(train_dataset, True)

    # def initialize_val_loader(self):
    #     eval_dataset = load_dataset(
    #         dataset_name=self.config.data.dataset,
    #         data_path=self.config.data.data_path,
    #         train=False,
    #     )

    #     return self.dataset_to_loader(eval_dataset, False)
