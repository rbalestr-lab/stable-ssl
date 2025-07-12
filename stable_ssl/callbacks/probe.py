import types
from functools import partial
from typing import Optional, Union

import torch
import torchmetrics
from hydra.utils import instantiate
from lightning.pytorch import Callback, LightningModule
from loguru import logger as logging

from ..optim import LARS
from ..utils import get_required_fn_parameters
from .utils import EarlyStopping, format_metrics_as_dict


def add_optimizer_scheduler(fn, optimizer, scheduler, name):
    def ffn(self, fn=fn, optimizer=optimizer, scheduler=scheduler, name=name):
        logging.info(f"`configure_optimizers` (wrapped by {name})")
        existing_optimizer_scheduler = fn()
        if existing_optimizer_scheduler is None:
            _optimizer = []
            _scheduler = []
        else:
            _optimizer, _scheduler = existing_optimizer_scheduler
        optimizer = optimizer(self._callbacks_modules[name].parameters())
        scheduler = scheduler(optimizer)
        if not hasattr(self, "_callbacks_optimizers_index"):
            self._callbacks_optimizers_index = dict()
        self._callbacks_optimizers_index[name] = len(_optimizer)
        logging.info(f"{name} optimizer/scheduler are at index {len(_optimizer)}")
        return _optimizer + [optimizer], _scheduler + [scheduler]

    return ffn


def wrap_forward(fn, target, input, name, loss_fn):
    def ffn(
        self,
        *args,
        fn=fn,
        name=name,
        target=target,
        input=input,
        loss_fn=loss_fn,
        **kwargs,
    ):
        batch = fn(*args, **kwargs)
        y = batch[target].detach().clone()
        x = batch[input].detach().clone()
        preds = self._callbacks_modules[name](x)

        # add predictions to the batch dict
        prediction_key = f"{name}_preds"

        if prediction_key in batch:
            msg = (
                f"Asking to save predictions for callback `{name}`"
                f"but `{prediction_key}` already exists in the batch dict."
            )
            logging.error(msg)
            raise ValueError(msg)
        batch[prediction_key] = preds.detach()

        # to avoid ddp unused parameter bug... need a better fix!
        if self.training:
            loss = loss_fn(preds, y)
            if "loss" in batch:
                batch["loss"] += loss
            else:
                batch["loss"] = loss
            logs = {f"train/{name}_loss": loss}
            for k, metric in self._callbacks_metrics[name]["_train"].items():
                metric(preds, y)
                logs[f"train/{name}_{k}"] = metric
            self.log_dict(logs, on_step=True, on_epoch=True)
            return batch
        else:
            logs = {}
            for k, metric in self._callbacks_metrics[name]["_val"].items():
                metric(preds, y)
                logs[f"eval/{name}_{k}"] = metric
            self.log_dict(logs, on_step=False, on_epoch=True)
            return batch

    return ffn


# def wrap_validation_step(fn, target, input, name):
#     def ffn(self, batch, batch_idx, fn=fn, name=name, target=target, input=input):
#         fn(batch, batch_idx)
#         y = batch[target]
#         x = batch[input]
#         preds = self._callbacks_modules[name](x)
#         logs = {}
#         for k, metric in self._callbacks_metrics[name]["_val"].items():
#             metric(preds, y)
#             logs[f"eval/{name}_{k}"] = metric
#         self.log_dict(logs, on_step=False, on_epoch=True)

#     return ffn


class OnlineProbe(Callback):
    """Attaches a MLP for fine-tuning using the standard self-supervised protocol."""

    def __init__(
        self,
        name: str,
        pl_module: LightningModule,
        input: str,
        target: str,
        probe: torch.nn.Module,
        loss_fn: callable,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        accumulate_grad_batches: int = 1,
        metrics: Optional[Union[dict, tuple, list, torchmetrics.Metric]] = None,
        early_stopping: EarlyStopping = None,
    ) -> None:
        super().__init__()
        logging.info(f"Setting up callback ({name=})")
        logging.info(f"\t- {input=}")
        logging.info(f"\t- {target=}")
        logging.info("\t- caching modules into `_callbacks_modules`")
        self.accumulate_grad_batches = accumulate_grad_batches
        if name in pl_module._callbacks_modules:
            raise ValueError(f"{name=} already used in callbacks")
        if isinstance(probe, torch.nn.Module):
            model = probe
        elif callable(probe):
            model = probe()
        else:
            model = instantiate(probe, _convert_="object")
        pl_module._callbacks_modules[name] = model
        logging.info(
            f"`_callbacks_modules` now contains ({list(pl_module._callbacks_modules.keys())})"
        )

        logging.info("\t- caching metrics into `_callbacks_metrics`")
        metrics = format_metrics_as_dict(metrics)
        print(metrics)
        pl_module._callbacks_metrics[name] = metrics
        for k in pl_module._callbacks_metrics[name].keys():
            logging.info(f"\t\t- {k}")

        logging.info("\t- overriding base pl_module `configure_optimizers`")
        if optimizer is None:
            logging.warning(
                "\t- No optimizer given to OnlineProbe, using default's LARS"
            )
            optimizer = partial(
                LARS,
                lr=0.1,
                clip_lr=True,
                eta=0.02,
                exclude_bias_n_norm=True,
                weight_decay=0,
            )
        if scheduler is None:
            logging.warning("\t- No scheduler given to OnlineProbe, using constant")
            scheduler = partial(torch.optim.lr_scheduler.ConstantLR, factor=1)
        if get_required_fn_parameters(optimizer) != ["params"]:
            names = get_required_fn_parameters(optimizer)
            raise ValueError(
                "Optimizer is supposed to be a partial with only"
                f"`params` left as arg, current has {names}"
            )

        logging.info("\t- wrapping the `configure_optimizers`")
        fn = add_optimizer_scheduler(
            pl_module.configure_optimizers,
            optimizer=optimizer,
            scheduler=scheduler,
            name=name,
        )
        pl_module.configure_optimizers = types.MethodType(fn, pl_module)

        logging.info("\t- wrapping the `training_step`")
        fn = wrap_forward(pl_module.forward, target, input, name, loss_fn)
        pl_module.forward = types.MethodType(fn, pl_module)

        logging.info("\t- wrapping the `validation_step`")
        # fn = wrap_validation_step(pl_module.validation_step, target, input, name)
        pl_module.validation_step = types.MethodType(fn, pl_module)
        self.name = name
        if hasattr(pl_module, "callbacks_training_step"):
            pl_module.callbacks_training_step.append(self.training_step)
        else:
            pl_module.callbacks_training_step = [self.training_step]
        self.pl_module = pl_module
        self.early_stopping = early_stopping

    def on_validation_end(self, trainer, pl_module):
        # stop every ddp process if any world process decides to stop
        if self.early_stopping is None:
            return
        raise NotImplementedError
        metric = trainer.callback_metrics[self.name]
        should_stop = self.early_stopping.should_stop(trainer.current_epoch)
        trainer.should_stop = trainer.should_stop or should_stop
        return super().on_validation_end(trainer, pl_module)

    def training_step(self, batch_idx):
        opt_idx = self.pl_module._callbacks_optimizers_index[self.name]
        optimizers = self.pl_module.optimizers()
        schedulers = self.pl_module.lr_schedulers()
        if type(optimizers) in [list, tuple]:
            opt = optimizers[opt_idx]
            sched = schedulers[opt_idx]
        else:
            opt = optimizers
            sched = schedulers
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
        # logs = {f"train/{self.name}_loss": loss}
        # for k, metric in self._callbacks_metrics[name].items():
        #     metric(preds, y)
        #     logs[f"train/{name}_{k}"] = metric
        # self.log_dict(logs, on_step=True, on_epoch=False)
        # self.untoggle_optimizer(opt)
