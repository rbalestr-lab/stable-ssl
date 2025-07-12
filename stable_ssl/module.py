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
    def __init__(self, *args, forward: callable, hparams: dict = None, **kwargs):
        super().__init__()
        logging.info("Configuring module! 🔧")
        logging.info("Setting `automatic_optimization` to False! 🔧")
        self.automatic_optimization = False
        self._callbacks_modules = torch.nn.ModuleDict()
        self._callbacks_metrics = torch.nn.ModuleDict()

        if len(args) > 0:
            raise ValueError("takes no args! this is to simplify logging")
        if hparams is None:
            logging.warning("No hparams given, none will be logged!")
        else:
            logging.info("Saving user's hparams!")
            self.save_hyperparameters(hparams)
        logging.warning("Using forward method from user")
        setattr(self, "forward", types.MethodType(forward, self))
        for key, value in kwargs.items():
            logging.info(f"Setting `self.{key}` with {type(value)} from user")
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
            logging.info("No `metrics` given, automatic metric disabled")

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        # self.toggle_optimizer(opt)
        state = self.forward(batch, stage="fit")
        if "loss" in state:
            self.manual_backward(state["loss"])
        if hasattr(self, "callbacks_training_step"):
            for fn in self.callbacks_training_step:
                fn(batch_idx)
        # if no optimization is happening we can leave early
        if self.optim is None or self.optim is False:
            return state
        if "loss" not in state:
            logging.error(
                "the forward dictionary should contain `loss` otherwise use `optim=False` in your module"
                "if you don't need training/optimization"
            )
        # self.untoggle_optimizer(opt)
        # accumulate gradients of N batches
        N = self.trainer.accumulate_grad_batches
        if (batch_idx + 1) % N == 0:
            # clip gradients
            opt = self.optimizers()
            if type(opt) is list:
                opt = opt[0]
            self.clip_gradients(
                opt,
                gradient_clip_val=self.trainer.gradient_clip_val,
                gradient_clip_algorithm=self.trainer.gradient_clip_algorithm,
            )

            opt.step()
            opt.zero_grad(set_to_none=True)
            sch = self.lr_schedulers()
            # TODO: should we always use 0?
            if type(sch) is list:
                sch = sch[0]
            if sch is not None:
                sch.step()
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

    def create_scheduler(self, optim, name):
        if name == "CosineAnnealingLR":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optim, T_max=self.trainer.estimated_stepping_batches
            )
        elif name == "OneCycleLR":
            pct = min(10 / self.trainer.max_epochs, 0.01)
            return torch.optim.lr_scheduler.OneCycleLR(
                optim,
                max_lr=optim.param_groups[0]["lr"],
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=pct,
            )
        else:
            raise ValueError

    def configure_optimizers(self):
        logging.info("`configure_optimizers` (main) 🔧")
        if not hasattr(self, "optim"):
            logging.info(
                "No optimizer specified, using default AdamW and no scheduler!"
            )
            self.optim = dict(optimizer=partial(torch.optim.AdamW))
        elif self.optim is None or self.optim is False:
            logging.warning("⚠️⚠️⚠️ No optimizer given! Skipping...")
            return
        if isinstance(self.optim.get("optimizer", None), partial):
            logging.info("\tretreived a single optimizer with correct type!")
            assert callable(self.optim["optimizer"])
            assert get_required_fn_parameters(self.optim["optimizer"]) == ["params"]
            params = self.named_parameters()
            params = [u[1] for u in params if "_callbacks_modules" not in u[0]]
            opt = [self.optim["optimizer"](params)]
            sched_name = self.optim.get("scheduler", "CosineAnnealingLR")
            sched = self.create_scheduler(opt[0], sched_name)
            opt_name = opt[0].__class__.__name__
            logging.info(
                f"\t\t- optimizer {opt_name}: with trainable parameters, {sched_name} sched. ✅"
            )
            logging.info("Configuring optimizers, done!  ✅")
            return opt, [{"scheduler": sched, "interval": "step"}]
        elif not isinstance(self.optim, dict):
            logging.info(
                "\toptimizer specified by type (type(optimizer))..."
                "we need a torch.optim.Optimizer type or dict!"
            )
            raise ValueError
        logging.info(
            f"\toptimizer specified by Dict with keys {list(self.optim.keys())}... 🔧"
        )
        regexes = [re.compile(u["modules"]) for u in self.optim.values()]
        parameters = [[] for _ in range(len(regexes))]
        for name, module in self.named_modules():
            for i, regex in enumerate(regexes):
                if regex.match(name):
                    parameters[i].extend(module.parameters())
                    break
        optimizer = [
            opti["optimizer"](params)
            for opti, params in zip(self.optim.values(), parameters)
        ]
        scheduler = []
        for name, optim, params in zip(self.optim, optimizer, parameters):
            sched_name = self.optim[name].get("scheduler", "CosineAnnealingLR")
            sched = self.create_scheduler(optim, sched_name)
            scheduler.append({"scheduler": sched, "interval": "step", "name": name})
            logging.info(
                f"\t\t- optimizer {name}: {len(params)} parameters, {sched_name} sched. ✅"
            )
        logging.info("Configuring optimizers, done!  ✅")
        return optimizer, scheduler
