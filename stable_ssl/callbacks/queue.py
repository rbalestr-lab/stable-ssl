"""Simplified queue callback that only collects data without complex sharing logic."""

from typing import Optional, Union

import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from loguru import logger as logging

from stable_ssl.utils import UnsortedQueue


class OnlineQueue(Callback):
    """Maintain a circular buffer for a single batch key during training.

    This callback creates a circular buffer that accumulates data during training
    and provides snapshots during validation. Other callbacks can discover and
    use this queue based on the data key and properties.

    Args:
        key: The batch key whose tensor will be queued every training step.
        queue_length: Maximum number of elements to keep in the queue.
        dim: Pre-allocate buffer with this shape. If None, inferred from first batch.
        dtype: Pre-allocate buffer with this dtype. If None, inferred from first batch.
        gather_distributed: If True, gather data across distributed processes during validation.
    """

    def __init__(
        self,
        key: str,
        queue_length: int,
        dim: Optional[Union[int, tuple]] = None,
        dtype: Optional[torch.dtype] = None,
        gather_distributed: bool = False,
    ) -> None:
        super().__init__()

        self.key = key
        self.queue_length = queue_length
        self.dim = dim
        self.dtype = dtype
        self.gather_distributed = gather_distributed
        self._queue = None  # Will be initialized in setup
        self._snapshot = None

        logging.info(f"OnlineQueue initialized for key '{key}'")
        logging.info(f"\t- queue_length: {queue_length}")
        logging.info(f"\t- dim: {dim}")
        logging.info(f"\t- dtype: {dtype}")

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Initialize queue during setup phase."""
        # Initialize queue if not already done (handles fit, validate, test, etc.)
        if self._queue is None:
            self._queue = UnsortedQueue(self.queue_length, self.dim, self.dtype)
            # Register as a submodule so it moves with the model
            pl_module.add_module(f"queue_{self.key}_{id(self)}", self._queue)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict,
        batch: dict,
        batch_idx: int,
    ) -> None:
        """Append batch data to queue."""
        with torch.no_grad():
            if self.key not in batch:
                logging.warning(
                    f"OnlineQueue: Key '{self.key}' not found in batch. "
                    f"Available keys: {list(batch.keys())}"
                )
                return
            self._queue.append(batch[self.key])

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Create snapshot of queue contents for validation."""
        logging.info(f"OnlineQueue: Creating snapshot for key '{self.key}'")

        tensor = self._queue.get()

        if self.gather_distributed and trainer.world_size > 1:
            gathered = pl_module.all_gather(tensor).flatten(0, 1)
            self._snapshot = gathered
            logging.info(
                f"\t- {self.key}: {tensor.shape} -> {gathered.shape} (gathered)"
            )
        else:
            self._snapshot = tensor
            logging.info(f"\t- {self.key}: {tensor.shape}")

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Clean up snapshot after validation."""
        self._snapshot = None

    @property
    def data(self) -> Optional[torch.Tensor]:
        """Get snapshot data during validation."""
        if self._snapshot is None:
            logging.warning("No queue snapshot available. Called outside validation?")
            return None
        return self._snapshot


def find_or_create_queue_callback(
    trainer: Trainer,
    key: str,
    queue_length: int,
    dim: Optional[Union[int, tuple]] = None,
    dtype: Optional[torch.dtype] = None,
    gather_distributed: bool = False,
    create_if_missing: bool = True,
) -> "OnlineQueue":
    """Find an OnlineQueue callback or create one if it doesn't exist.

    Args:
        trainer: The Lightning trainer containing callbacks
        key: The batch key to look for
        queue_length: Required queue length
        dim: Required dimension (None means any)
        dtype: Required dtype (None means any)
        gather_distributed: Whether to gather across distributed processes
        create_if_missing: If True, create queue when not found

    Returns:
        The matching or newly created OnlineQueue callback

    Raises:
        ValueError: If no matching queue is found and create_if_missing is False
    """
    matching_queues = []

    for callback in trainer.callbacks:
        if isinstance(callback, OnlineQueue) and callback.key == key:
            # Check queue length match
            if callback.queue_length != queue_length:
                continue

            # Check dim compatibility (None matches anything)
            if dim is not None and callback.dim is not None and callback.dim != dim:
                continue

            # Check dtype compatibility (None matches anything)
            if (
                dtype is not None
                and callback.dtype is not None
                and callback.dtype != dtype
            ):
                continue

            matching_queues.append(callback)

    if not matching_queues:
        if create_if_missing:
            # Create a new queue callback
            logging.info(
                f"No queue found for key '{key}', creating new OnlineQueue with "
                f"length={queue_length}, dim={dim}, dtype={dtype}"
            )
            new_queue = OnlineQueue(
                key=key,
                queue_length=queue_length,
                dim=dim,
                dtype=dtype,
                gather_distributed=gather_distributed,
            )
            # Add to trainer callbacks
            trainer.callbacks.append(new_queue)
            # Run setup if trainer is already set up
            if (
                hasattr(trainer, "lightning_module")
                and trainer.lightning_module is not None
            ):
                new_queue.setup(trainer, trainer.lightning_module, "fit")
            return new_queue
        else:
            # List all available queues for better error message
            available = [
                f"(key='{cb.key}', length={cb.queue_length}, dim={cb.dim}, dtype={cb.dtype})"
                for cb in trainer.callbacks
                if isinstance(cb, OnlineQueue)
            ]
            raise ValueError(
                f"No OnlineQueue found for key '{key}' with queue_length={queue_length}, "
                f"dim={dim}, dtype={dtype}. Available queues: {available}"
            )

    if len(matching_queues) > 1:
        queue_details = [
            f"(length={cb.queue_length}, dim={cb.dim}, dtype={cb.dtype})"
            for cb in matching_queues
        ]
        logging.warning(
            f"Multiple OnlineQueue callbacks found for key '{key}': {queue_details}. "
            f"Using the first one."
        )

    return matching_queues[0]
