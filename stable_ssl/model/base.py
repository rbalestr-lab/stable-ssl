# -*- coding: utf-8 -*-
"""Main function for training a model."""
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import copy
import logging
import warnings
import time
import dataclasses
from pathlib import Path
from tqdm import tqdm
import submitit
import jsonlines
import omegaconf
from ..reader import jsonl_run


try:
    import wandb
except ModuleNotFoundError:
    logging.warning(
        "Wandb module is not installed, make sure to not use wandb for logging "
        "or an error will be thrown"
    )

import torch

from ..utils import (
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
)

from ..data import load_dataset
from dataclasses import make_dataclass
from dataclasses import dataclass


@dataclass
class BaseModelConfig:
    """
    Configuration for the SSL model parameters.

    Parameters:
    -----------
    model : str
        Type of model to use. Default is "SimCLR".
    backbone_model : str
        Neural network architecture to use for the backbone. Default is "resnet9".
    sync_batchnorm : bool, optional
        Whether to use synchronized batch normalization. Default is False.
    memory_format : str, optional
        Memory format for tensors (e.g., "channels_last"). Default is "channels_last".
    pretrained : bool
        Whether to use the torchvision pretrained weights or use random initialization.
    with_classifier : int
        Whether to keep the last layer(s) of the backbone (classifier)
        when loading the model.
    """

    name: str = "Supervised"
    backbone_model: str = "resnet18"
    sync_batchnorm: bool = False
    memory_format: str = "channels_last"
    pretrained: bool = False
    with_classifier: bool = True


class BaseModel(torch.nn.Module):
    r"""Base class for training a model.

    Parameters:
    -----------
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

    @property
    def config(self):
        return self._config

    def __init__(self, config, *args, **kwargs):
        super().__init__()

    def __call__(self):

        if self.config.log.api == "wandb":
            print(
                "[stable-SSL] \t=> Initializating wandb for logging in "
                f"{self.config.log.dump_path}."
            )
            wandb.init(
                entity=self.config.log.entity,
                project=self.config.log.project,
                config=dataclasses.asdict(self.config),
                name=self.config.log.run,
                dir=str(self.config.log.dump_path),
                resume="allow",
            )
        else:
            print(
                f"[stable-SSL] \t=> Dumping config file in {self.config.log.dump_path}"
            )
            omegaconf.OmegaConf.save(
                self.config, self.config.log.dump_path / "hparams.yaml"
            )

        logging.basicConfig(level=self.config.log.level)
        self.seed_everything(self.config.hardware.seed)

        self.scaler = torch.amp.GradScaler("cuda", enabled=self.config.hardware.float16)
        self._set_device()

        if not self.config.log.eval_only:
            logging.info("[stable-SSL] Creating train_loader dataset.")
            self.train_loader = self.initialize_train_loader()
            assert hasattr(self, "train_loader")
            logging.info(
                "[stable-SSL] \t=> Found training set of length "
                f"{len(self.train_loader)}."
            )
        else:
            logging.info("[stable-SSL] \t=> No training set loaded since eval_only.")

        logging.info("[stable-SSL] Creating val_loader dataset.")
        try:
            self.val_loader = self.initialize_val_loader()
            logging.info(
                "[stable-SSL] \t=> Found validation set of length "
                f"{len(self.val_loader)}."
            )
        except NotImplementedError:
            logging.info(
                "[stable-SSL] \t=> Found no implementation of initialize_val_loader. "
                "Skipping for now."
            )
            self.val_loader = None

        logging.info("[stable-SSL] Calling initialize_modules() method.")
        self.initialize_modules()
        if hasattr(self, "metrics"):
            raise RuntimeError(
                "You can't assign any value to `self.metrics`, this will be "
                "used for metrics only"
            )
        self.initialize_metrics()
        if not hasattr(self, "metrics"):
            raise RuntimeError(
                "The `initialize_metrics` method should create a `self.metrics` "
                "ModuleDict object"
            )
        if not isinstance(self.metrics, torch.nn.ModuleDict):
            raise RuntimeError("The `self.metrics` should be a ModuleDict")
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
                    module, device_ids=[self.config.hardware.gpu]
                )
            setattr(self, name, module)

            trainable = sum(
                param.numel() for param in module.parameters() if param.requires_grad
            )
            logging.info(
                f"[stable-SSL] \t=> Found module '{name}' with {trainable} "
                "trainable parameters."
            )

        if not self.config.log.eval_only:
            logging.info("[stable-SSL] Calling initialize_optimizer() method.")
            self.optimizer = self.initialize_optimizer()
            logging.info("[stable-SSL] Calling initialize_scheduler() method.")
            try:
                self.scheduler = self.initialize_scheduler()
            except NotImplementedError:
                logging.info("[stable-SSL] No scheduler given...")
        else:
            logging.info(
                "[stable-SSL] Mode is eval_only, skipping optimizer and "
                "scheduler initializations."
            )

        logging.info("[stable-SSL] Calling load_checkpoint() method.")
        self.load_checkpoint()
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
                self.eval_epoch()  # always eval the model after training
            except BreakAllEpochs:
                self.cleanup()
                wandb.finish()

    def seed_everything(self, seed):
        seed_everything(seed)

    def initialize_metrics(self):
        self.metrics = torch.nn.ModuleDict()

    def _train_all_epochs(self):
        while self.epoch < self.config.optim.epochs:

            if hasattr(self, "train_sampler"):
                self.train_sampler.set_epoch(self.epoch)

            try:
                self._train_epoch()
            except BreakEpoch:
                print("[stable-SSL] Train epoch cut by user. Going to the next one.")
            except NanError:
                print("[stable-SSL] Nan error.")
                return
            except Exception as e:
                raise (e)

            if self.config.log.eval_each_epoch:
                self.eval_epoch()
            self.epoch = self.epoch + 1

            freq = self.config.log.checkpoint_frequency
            if self.epoch % freq == 0:
                print("[stable-SSL] Checkpointing everything to restart if needed.")
                self.save_checkpoint("tmp_checkpoint.ckpt", model_only=False)

        # at the end of training, we (optionally) save the final model
        if self.config.log.save_final_model:
            self.save_checkpoint(
                f"{self.config.log.final_model_name}.ckpt", model_only=True
            )
        # and remove any temporary checkpoint
        (self.config.log.dump_path / "tmp_checkpoint.ckpt").unlink(missing_ok=True)

        wandb.finish()

    def _train_epoch(self):

        # hierarchically set up all modules in train mode
        self.before_train_epoch()

        # we do not ensure that the model is still in train mode to not
        # override any user desired behavior, simply speak out
        if not self.training:
            logging.warn(
                "[stable-SSL] Starting training epoch but model is no longer in "
                "train mode after call to before_train_epoch()."
            )

        if self.config.optim.max_steps < 0:
            max_steps = len(self.train_loader)
        elif 0 < self.config.optim.max_steps < 1:
            logging.info(
                f"[stable-SSL] \t=> Training on {self.config.optim.max_steps*100}% of "
                "the training dataset"
            )
            max_steps = int(self.config.optim.max_steps * len(self.train_loader))
        else:
            max_steps = self.config.optim.max_steps

        for batch_idx, data in enumerate(
            tqdm(self.train_loader, total=max_steps, desc=f"Training: {self.epoch=}")
        ):
            # set up the data to have easy access throughout the methods
            self.batch_idx = batch_idx
            self.global_step.add_(1)
            self.data = to_device(data, self.this_device)

            try:
                # call any user specified pre-step functions
                self.before_train_step()

                # perform the gradient step
                self.train_step()

                # call any user specified post-step function
                self.after_train_step()
            except BreakStep:
                logging.warn("[stable-SSL] train_step has been interrupted by user.")

            # we cut early in case the user specifies to only use
            # X% of the training dataset
            if batch_idx >= max_steps:
                break

        # call any user specified post-epoch function
        self.after_train_epoch()

        # be sure to clean up to avoid silent bugs
        self.data = None

    def eval_epoch(self) -> dict:

        if self.val_loader is None:
            logging.info("[stable-SSL] No val_loader hence skipping eval epoch.")
            return

        # set-up model in eval mode + reset metrics
        self.before_eval_epoch()

        # we do not ensure that the model is still in eval mode to not
        # override any user desired behavior
        if self.training:
            warnings.warn(
                "[stable-SSL] Starting eval epoch but model is not in "
                "eval mode after call to before_eval_epoch()."
            )

        try:
            max_steps = len(self.val_loader)
            with torch.inference_mode():
                for step, data in tqdm(
                    enumerate(self.val_loader),
                    total=max_steps,
                    desc=f"Eval: {self.epoch=}",
                ):
                    self.batch_idx = step
                    self.data = to_device(data, self.this_device)

                    # call any user specified pre-step function
                    self.before_eval_step()

                    # call the eval step
                    with torch.amp.autocast(
                        "cuda", enabled=self.config.hardware.float16
                    ):
                        self.eval_step()

                    # call any user specified post-step function
                    self.after_eval_step()

        except BreakEpoch:
            print("[stable-SSL] Eval epoch cut by user...")
        except Exception as e:
            raise (e)

        # be sure to clean up to avoid silent bugs
        self.data = None

        # call any user specified post-epoch function
        self.after_eval_epoch()

    def train_step(self):
        with torch.amp.autocast("cuda", enabled=self.config.hardware.float16):
            loss = self.compute_loss()

        if np.isnan(loss.item()):
            raise NanError

        self.log(
            {"train/loss": loss.item(), "epoch": self.epoch, "step": self.batch_idx},
            commit=False,
        )

        self.scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
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
                "train/lr": self.scheduler.get_last_lr()[0],
                "step": self.batch_idx,
                "epoch": self.epoch,
            },
            commit=False,
        )

    def _set_device(self):
        if torch.cuda.is_available() is False:
            self._device = "cpu"
            return
        try:
            self.config.hardware = setup_distributed(self.config.hardware)
            self._device = f"cuda:{self.config.hardware.gpu}"
        except RuntimeError as e:
            print(e)
            # self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._device = f"cuda:{self.config.hardware.gpu}"
            self.config.hardware.gpu = 0
            self.config.hardware.world_size = 1
        torch.cuda.set_device(self.config.hardware.gpu)

    def checkpoint(self):
        # the checkpoint method is called asynchroneously when the slurm manager
        # sends a preemption signal, with the same arguments as the __call__ method
        # "self" is your callable, at its current state.
        # "self" therefore holds the current version of the model:
        print("[stable-SSL] Requeuing...")
        config = copy.deepcopy(self.config)
        config.log.add_version = False
        config.log.folder = self.config.log.dump_path.as_posix()
        model = type(self)(config)
        return submitit.helpers.DelayedSubmission(model)

    def log(self, packet=None, commit=True):
        packet = packet or {}
        assert "_global_step" not in packet
        self._log_buffer.update(packet)
        if not commit or len(self._log_buffer) == 0:
            return
        # make values JSON serializable
        for name, value in self._log_buffer.items():
            if torch.is_tensor(value):
                if torch.numel(value) == 1:
                    self._log_buffer[name] = value.item()
                else:
                    self._log_buffer[name] = value.tolist()
        # log in wandb
        if self.config.log.api == "wandb":
            for name, value in self._log_buffer.items():
                if isinstance(value, list):
                    table = wandb.Table(columns=["index", "epoch", name])
                    for i, v in enumerate(np.asarray(value).flatten()):
                        table.add_data(i, v)
                    self._log_buffer[name] = table
            wandb.log(self._log_buffer, step=self.global_step.item())
        else:
            with jsonlines.open(
                self.config.log.dump_path / "csv_logs.jsonl", mode="a"
            ) as writer:
                writer.write(self._log_buffer)
        self._log_buffer = {}

    def load_checkpoint(self):
        """load a model and optionnally a scheduler/optimizer/epoch
        from a given path to checkpoint

        Args:
            load_from (str): path to checkpoint
        """
        load_from = Path(self.config.log.load_from)
        if load_from.is_file():
            logging.info(
                f"[stable-SSL] \t=> file {load_from} exists\n\t=> loading it..."
            )
            checkpoint = load_from
        elif (self.config.log.dump_path / "tmp_checkpoint.ckpt").is_file():
            logging.info(
                f"[stable-SSL] \t=> folder {self.config.log.dump_path} contains "
                "`tmp_checkpoint.ckpt` file\n\t=> loading it..."
            )
            checkpoint = self.config.log.dump_path / "tmp_checkpoint.ckpt"
        else:
            logging.info(f"[stable-SSL] \t=> no checkpoint at `{load_from}`")
            logging.info(
                "[stable-SSL] \t=> no checkpoint at "
                f"`{self.config.log.dump_path / 'tmp_checkpoint.ckpt'}`. "
            )
            logging.info("[stable-SSL] \t=> training from scratch...")
            self.epoch = 0
            return

        ckpt = torch.load(checkpoint, map_location="cpu")

        for name, model in self.named_children():
            if name not in ckpt:
                logging.info(f"[stable-SSL] \t\t=> {name} not in ckpt, skipping...")
                continue
            model.load_state_dict(ckpt[name])
            logging.info(f"[stable-SSL] \t\t=> {name} successfully loaded...")
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
            logging.info("[stable-SSL] \t\t=> optimizer successfully loaded...")
        if "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])
            logging.info("[stable-SSL] \t\t=> scheduler successfully loaded...")
        if "epoch" in ckpt:
            self.epoch = ckpt["epoch"]
            logging.info(
                f"[stable-SSL] \t\t=> training will start from epoch {ckpt['epoch']}"
            )
        else:
            self.epoch = 0

    def initialize_optimizer(self):
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

    @property
    def logs(self):
        if self.config.log.api == "wandb":
            raise NotImplementedError
        else:
            return jsonl_run(self.config.log.dump_path)[1]

    def initialize_scheduler(self):
        min_lr = self.config.optim.lr * 0.005
        peak_step = 5 * len(self.train_loader)
        total_steps = self.config.optim.epochs * len(self.train_loader)
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
            print("Cleaning distributed processes...")
            torch.distributed.destroy_process_group()
        else:
            print("Not using distributed... nothing to clean")

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

    @epoch.setter
    def epoch(self, value):
        self._epoch = value

    @property
    def step(self):
        if not hasattr(self, "_step"):
            return None
        return self._step

    @step.setter
    def step(self, value):
        self._step = value

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def this_device(self):
        return self._device

    def before_train_all_epochs(self):
        return

    def before_train_epoch(self):
        self.train()

    def before_train_step(self):
        self.optimizer.zero_grad(set_to_none=True)
        return

    def after_train_step(self):
        self.log(commit=True)

    def after_train_epoch(self):
        return

    def before_eval_epoch(self):
        self.eval()
        for name, metric in self.metrics.items():
            if name.startswith("eval/epoch/"):
                metric.reset()

    def after_eval_epoch(self):
        packet = {}
        for name, metric in self.metrics.items():
            if name.startswith("eval/epoch/"):
                packet[name] = metric.compute()
        self.log(packet, commit=True)

    def before_eval_step(self):
        return

    def after_eval_step(self):
        return

    def eval_step(self):
        output = self.forward(self.data[0])
        for name, metric in self.metrics.items():
            if name.startswith("eval/epoch/"):
                metric.update(output, self.data[1])
            elif name.startswith("eval/step/"):
                self.log({name: metric(output, self.data[1])}, commit=False)
        self.log(commit=True)

    def dataset_to_loader(self, dataset, train):
        if self.config.hardware.world_size > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, shuffle=not train, drop_last=train
            )
            assert self.config.optim.batch_size % self.config.hardware.world_size == 0
            drop_last = None
            shuffle = None
        else:
            sampler = None
            drop_last = train
            shuffle = not train

        per_device_batch_size = (
            self.config.optim.batch_size // self.config.hardware.world_size
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=per_device_batch_size,
            num_workers=self.config.data.num_workers,
            pin_memory=True,
            sampler=sampler,
            drop_last=drop_last,
            shuffle=shuffle,
        )

        return loader

    def initialize_train_loader(self):
        train_dataset = load_dataset(
            dataset_name=self.config.data.dataset,
            data_path=self.config.data.data_path,
            train=True,
        )

        return self.dataset_to_loader(train_dataset, True)

    def initialize_val_loader(self):
        eval_dataset = load_dataset(
            dataset_name=self.config.data.dataset,
            data_path=self.config.data.data_path,
            train=False,
        )

        return self.dataset_to_loader(eval_dataset, False)

    def initialize_modules(self):
        raise NotImplementedError("You need to implement your own `initialize_modules`")

    def forward(self):
        raise NotImplementedError("You need to implement your own `forward`")

    def compute_loss(self):
        raise NotImplementedError("You need to implement your own `compute_loss`")
