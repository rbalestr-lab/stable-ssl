from typing import Dict, Iterable, Union

import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from loguru import logger as logging

from stable_ssl.utils import UnsortedQueue


class OnlineQueue(Callback):
    """Maintain per-key circular buffers during training and snapshot them during validation.

    This callback automatically creates an `UnsortedQueue` for each key in
    `to_save` on your `pl_module`, appends that batch tensor at the end of every
    training batch, and wraps around when it reaches `queue_length`.  At the
    start of validation it takes a read-only snapshot of all queues, storing
    them in `pl_module._callbacks_validation_cache[name]`, which downstream
    callbacks can use (e.g. for contrastive losses or nearest-neighbor metrics).

    Args:
        pl_module (LightningModule):
            Your LightningModule; this callback will store its queues in
            `pl_module._callbacks_modules[name]`.
        name (str):
            Unique identifier for this queue callback.  Must not already exist
            in `pl_module._callbacks_modules`.
        to_save (str or Iterable[str]):
            Batch keys (e.g. `"features"`, `"labels"`) whose tensors will be
            enqueued every training step.
        queue_length (int):
            Maximum number of elements to keep per queue.  Once full, oldest
            entries are overwritten in circular fashion.
        dims (int, tuple[int, ...], list[int or tuple[int, ...]], optional):
            Pre-allocate buffers with these shapes.  If a single value is given,
            it is broadcast to all `to_save` keys.  If None (default), shapes
            are inferred lazily from the first batch.
        dtypes (torch.dtype or list[torch.dtype], optional):
            Pre-allocate buffers with these dtypes.  Behaves like `dims` re:
            broadcasting and inference on first batch.
    """

    def __init__(
        self,
        pl_module,
        name: str,
        to_save: Union[str, Iterable],
        queue_length: int,
        dims: Union[list[tuple[int]], list[int], int, None] = None,
        dtypes: Union[tuple[int], list[int], int, None] = None,
    ) -> None:
        # Normalize to_save to always be a list
        if isinstance(to_save, str):
            to_save = [to_save]

        # Normalize dims to always be a list matching to_save length
        if dims is not None:
            if not isinstance(dims, (list, tuple)):
                # Single dim provided for all tensors
                dims = [dims] * len(to_save)
            elif len(dims) == 1 and len(to_save) > 1:
                # Single dim in list, apply to all tensors
                dims = dims * len(to_save)
        else:
            dims = [None] * len(to_save)

        # Normalize dtypes to always be a list matching to_save length
        if dtypes is not None:
            if not isinstance(dtypes, (list, tuple)):
                # Single dtype provided for all tensors
                dtypes = [dtypes] * len(to_save)
            elif len(dtypes) == 1 and len(to_save) > 1:
                # Single dtype in list, apply to all tensors
                dtypes = dtypes * len(to_save)
        else:
            dtypes = [None] * len(to_save)

        # Validate lengths match
        if len(dims) != len(to_save):
            raise ValueError(
                f"Length of dims ({len(dims)}) must match length of to_save ({len(to_save)})"
            )
        if len(dtypes) != len(to_save):
            raise ValueError(
                f"Length of dtypes ({len(dtypes)}) must match length of to_save ({len(to_save)})"
            )

        logging.info(f"Setting up callback ({name=})")
        logging.info(f"\t- {to_save=}")
        logging.info(f"\t- {dims=}")
        logging.info(f"\t- {dtypes=}")
        logging.info("\t- caching modules into `_callbacks_modules`")
        if name in pl_module._callbacks_modules:
            raise ValueError(f"{name=} already used in callbacks")

        pl_module._callbacks_modules[name] = torch.nn.ModuleDict(
            {
                n: UnsortedQueue(queue_length, dim, dtype)
                if dim is not None and dtype is not None
                else UnsortedQueue(queue_length, dim)
                if dim is not None
                else UnsortedQueue(queue_length)
                for n, dim, dtype in zip(to_save, dims, dtypes)
            }
        )
        logging.info(
            f"`_callbacks_modules` now contains ({list(pl_module._callbacks_modules.keys())})"
        )
        self.name = name
        self.to_save = to_save

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Dict,
        batch: Dict,
        batch_idx: int,
    ) -> None:
        """Store specified tensors from batch into persistent queues during training.

        This method appends tensors from the current batch to their respective queues
        in pl_module._callbacks_modules[self.name]. These queues accumulate data
        across all training batches and epochs until they reach queue_length, at which
        point older entries are overwritten in a circular buffer fashion.

        The stored data persists across validation epochs and is used to create
        snapshots during validation via on_validation_epoch_start.
        """
        with torch.no_grad():
            for key in self.to_save:
                if key not in batch:
                    logging.warning(
                        f"Key '{key}' not found in batch for {self.name} callback. Expected one of: {self.to_save}"
                    )
                    continue
                pl_module._callbacks_modules[self.name][key].append(batch[key])

    def on_validation_epoch_start(self, trainer, pl_module):
        """Cache queue contents at validation start for other callbacks to access.

        This creates a temporary snapshot of the current queue contents in
        pl_module._callbacks_validation_cache[self.name]. The actual queues in
        pl_module._callbacks_modules remain unchanged and continue accumulating
        data. The cache is cleared at validation end.
        """
        logging.info(f"{self.name}: validation epoch start, caching queue(s)")

        if not hasattr(pl_module, "_callbacks_validation_cache"):
            pl_module._callbacks_validation_cache = {}

        pl_module._callbacks_validation_cache[self.name] = {}
        for n, q in pl_module._callbacks_modules[self.name].items():
            tensor = q.get()
            pl_module._callbacks_validation_cache[self.name][n] = tensor
            logging.info(f"\t- {n}: {tensor.shape}, {tensor.dtype}")

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Clean up the validation cache at the end of validation epoch.

        This removes the temporary snapshot of queue contents that was created
        at validation start. The actual queues in pl_module._callbacks_modules
        remain intact and continue to accumulate data. Only the cached snapshot
        in pl_module._callbacks_validation_cache is deleted.

        Note: This cleanup is important to free memory and ensure fresh snapshots
        are created for each validation epoch.
        """
        logging.info(f"{self.name}: validation epoch end, cleaning up cache")
        del pl_module._callbacks_validation_cache[self.name]
