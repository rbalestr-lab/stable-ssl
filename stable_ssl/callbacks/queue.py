from typing import Optional, Tuple, Union, Iterable
import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from ..utils import UnsortedQueue
from loguru import logger as logging
import types


def wrap_training_step(fn, name):
    def ffn(self, batch, batch_idx, fn=fn, name=name):
        batch = fn(batch, batch_idx)
        with torch.no_grad():
            for n, q in self._callbacks_modules[name].items():
                q.append(batch[n])
        return batch

    return ffn


class OnlineQueue(Callback):
    def __init__(
        self,
        pl_module,
        name: str,
        to_save: Union[str, Iterable],
        queue_length: int,
        dims: Union[list[tuple[int]], list[int], int],
        dtypes: Union[tuple[int], list[int], int],
    ) -> None:
        """
        Args:
            k: k for k nearest neighbor
            temperature: temperature. See tau in section 3.4 of https://arxiv.org/pdf/1805.01978.pdf.
        """
        if type(to_save) is str:
            to_save = [to_save]
        if type(dims[0]) not in [tuple, list]:
            dims = [dims]
        if type(dtypes) not in [tuple, list]:
            dtypes = [dtypes]
        logging.info(f"Setting up callback ({name=})")
        logging.info(f"\t- {to_save=}")
        logging.info(f"\t- {dims=}")
        logging.info(f"\t- {dtypes=}")
        logging.info(f"\t- caching modules into `_callbacks_modules`")
        if name in pl_module._callbacks_modules:
            raise ValueError(f"{name=} already used in callbacks")
        pl_module._callbacks_modules[name] = torch.nn.ModuleDict(
            {
                n: UnsortedQueue(queue_length, dim, dtype)
                for n, dim, dtype in zip(to_save, dims, dtypes)
            }
        )
        logging.info(
            f"`_callbacks_modules` now contains ({list(pl_module._callbacks_modules.keys())})"
        )
        logging.info(f"\t- wrapping the `training_step`")
        fn = wrap_training_step(pl_module.training_step, name)
        pl_module.training_step = types.MethodType(fn, pl_module)
        self.name = name

    def on_validation_epoch_start(self, trainer, pl_module):
        logging.info(f"{self.name}: validation epoch start, caching queue(s)")
        if not hasattr(pl_module, "_callbacks_queue"):
            pl_module._callbacks_queue = {}
        pl_module._callbacks_queue[self.name] = {}
        for n, q in pl_module._callbacks_modules[self.name].items():
            tensor = q.get()
            pl_module._callbacks_queue[self.name][n] = tensor
            logging.info(f"\t- {n}: {tensor.shape}, {tensor.dtype}")

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        logging.info(f"{self.name}: validation epoch end, cleaning up cache")
        del pl_module._callbacks_queue[self.name]
