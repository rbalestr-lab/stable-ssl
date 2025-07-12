from typing import Optional, Tuple, Union

import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.accelerators import Accelerator
from torch import Tensor
from torch.nn import functional as F  # noqa: N812
from ..utils import UnsortedQueue

from loguru import logger as logging
import types
import numpy as np
from .utils import format_metrics_as_dict


def wrap_training_step(fn, target, input, name):
    def ffn(
        self,
        batch,
        batch_idx,
        fn=fn,
        name=name,
        target=target,
        input=input,
    ):
        batch = fn(batch, batch_idx)
        with torch.no_grad():
            norm = self._callbacks_modules[name]["normalizer"](batch[input])
            self._callbacks_modules[name]["queue_X"].append(norm.half())
            self._callbacks_modules[name]["queue_y"].append(batch[target])
        return batch

    return ffn


def wrap_validation_step(fn, target, input, name, k, temperature):
    def ffn(
        self,
        batch,
        batch_idx,
        fn=fn,
        name=name,
        target=target,
        input=input,
        k=k,
        temperature=temperature,
    ):
        batch = fn(batch, batch_idx)
        feature_bank = getattr(self, f"_cached_{name}_X")
        if feature_bank.size(0) == 0:
            return batch
        labels = getattr(self, f"_cached_{name}_y")
        num_classes = labels.max().item() + 1
        norm = self._callbacks_modules[name]["normalizer"](batch[input]).half()
        dist_matrix = torch.cdist(feature_bank, norm)
        dist_weight, sim_indices = dist_matrix.topk(k=k, dim=0, largest=False)
        dist_weight = 1 / dist_weight.add_(temperature)

        one_hot_labels = torch.nn.functional.one_hot(
            labels[sim_indices], num_classes=num_classes
        )
        # weighted score ---> [B, C]
        preds = (dist_weight.unsqueeze(-1) * one_hot_labels).sum(0)
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

        logs = {}
        for k, metric in self._callbacks_metrics[name]["_val"].items():
            metric(preds, batch[target])
            logs[f"eval/{name}_{k}"] = metric
        self.log_dict(logs, on_step=False, on_epoch=True)

        return batch

    return ffn


class OnlineKNN(Callback):
    """Weighted KNN online evaluator for self-supervised learning.
    The weighted KNN classifier matches sec 3.4 of https://arxiv.org/pdf/1805.01978.pdf.
    The implementation follows:
        1. https://github.com/zhirongw/lemniscate.pytorch/blob/master/test.py
        2. https://github.com/leftthomas/SimCLR
        3. https://github.com/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
    """

    NAME = "OnlineKNN"

    def __init__(
        self,
        pl_module,
        name: str,
        input: str,
        target: str,
        queue_length: int,
        metrics,
        features_dim: Union[tuple[int], list[int], int],
        k: int = 5,
        temperature: float = 0.07,
        normalizer: str = "batch_norm",
    ) -> None:
        """
        Args:
            k: k for k nearest neighbor
            temperature: temperature. See tau in section 3.4 of https://arxiv.org/pdf/1805.01978.pdf.
        """
        logging.info(f"Setting up callback ({self.NAME})")
        logging.info(f"\t- {input=}")
        logging.info(f"\t- {target=}")
        logging.info(f"\t- caching modules into `_callbacks_modules`")
        if name in pl_module._callbacks_modules:
            raise ValueError(f"{name=} already used in callbacks")
        if type(features_dim) in [list, tuple]:
            features_dim = np.prod(features_dim)
        if normalizer not in ["batch_norm", "layer_norm"]:
            raise ValueError(
                "`normalizer` has to be one of `batch_norm` or `layer_norm`"
            )
        if normalizer == "batch_norm":
            normalizer = torch.nn.BatchNorm1d(features_dim, affine=False)
            dtype = torch.half
        elif normalizer == "layer_norm":
            normalizer = torch.nn.LayerNorm(
                features_dim, elementwise_affine=False, bias=False
            )
            dtype = torch.half
        else:
            normalizer = torch.nn.Identity()
            dtype = torch.float

        pl_module._callbacks_modules[name] = torch.nn.ModuleDict(
            {
                "normalizer": normalizer,
                "queue_X": UnsortedQueue(queue_length, features_dim, dtype),
                "queue_y": UnsortedQueue(queue_length, (), torch.long),
            }
        )
        logging.info(
            f"`_callbacks_modules` now contains ({list(pl_module._callbacks_modules.keys())})"
        )
        logging.info(f"\t- caching metrics into `_callbacks_metrics`")
        pl_module._callbacks_metrics[name] = format_metrics_as_dict(metrics)

        logging.info(f"\t- wrapping the `training_step`")
        fn = wrap_training_step(pl_module.training_step, target, input, name)
        pl_module.training_step = types.MethodType(fn, pl_module)
        logging.info(f"\t- wrapping the `validation_step`")
        fn = wrap_validation_step(
            pl_module.validation_step, target, input, name, k, temperature
        )
        pl_module.validation_step = types.MethodType(fn, pl_module)

        self.k = k
        self.name = name

    def on_validation_epoch_start(self, trainer, pl_module):
        logging.info(
            f"(Validation epoch start, {self.name}) gather queue from all processes"
        )

        qx = pl_module._callbacks_modules[self.name]["queue_X"]
        qy = pl_module._callbacks_modules[self.name]["queue_y"]
        if pl_module.trainer.world_size > 1:
            X = pl_module.all_gather(qx.get()).flatten(0, 1)
            y = pl_module.all_gather(qy.get()).flatten(0, 1)
        else:
            X = qx.get()
            y = qy.get()
        setattr(pl_module, f"_cached_{self.name}_X", X)
        setattr(pl_module, f"_cached_{self.name}_y", y)
        logging.info(
            f"(Validation epoch start, {self.name}) X cache:{X.shape}, y cache: {y.shape}"
        )

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        logging.info(f"(Validation epoch end, {self.name}) cleanup")
        delattr(pl_module, f"_cached_{self.name}_X")
        delattr(pl_module, f"_cached_{self.name}_y")
