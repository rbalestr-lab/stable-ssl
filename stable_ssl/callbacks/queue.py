from typing import Optional, Union

import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from loguru import logger as logging

from stable_ssl.utils import UnsortedQueue, get_data_from_batch_or_outputs


class OnlineQueue(Callback):
    """Circular buffer callback for accumulating batch data during training.

    This callback maintains a circular buffer (queue) that accumulates data from
    specified batch keys during training. It provides snapshots of the accumulated
    data during validation, which other callbacks can discover and use for evaluation
    purposes (e.g., KNN evaluation, RankMe computation).

    The queue operates in two modes:
    1. Training: Continuously appends new batch data to the circular buffer
    2. Validation: Creates a snapshot of the buffer contents for evaluation

    Key features:
    - Automatic buffer initialization from first batch if dimensions not specified
    - Support for distributed training through optional data gathering
    - Discovery mechanism allowing other callbacks to find and use queues
    - Memory-efficient circular buffer implementation

    Args:
        key: The batch key whose tensor values will be queued at every training step.
            Can be any key present in the batch dict or outputs dict.
        queue_length: Maximum number of elements to keep in the circular buffer.
            Older elements are overwritten when the buffer is full.
        dim: Pre-allocate buffer with this shape. Can be int or tuple. If None,
            dimensions are inferred from the first batch.
        dtype: Pre-allocate buffer with this dtype (e.g., torch.float32). If None,
            dtype is inferred from the first batch.
        gather_distributed: If True, gather queue data across all distributed processes
            during validation. This creates a global view of the accumulated data.

    Attributes:
        data: Property that returns the current snapshot during validation, None otherwise.

    Note:
        - The queue is registered in pl_module._callbacks_modules for consistency
        - Snapshots are only available during validation epochs
        - The callback handles single-dimension data by adding a dimension if needed
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
            # Register in _callbacks_modules for consistency with other callbacks
            queue_key = f"queue_{self.key}_{id(self)}"
            pl_module._callbacks_modules[queue_key] = self._queue
            logging.info(
                f"OnlineQueue: Registered queue as '{queue_key}' in _callbacks_modules"
            )

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
            data = get_data_from_batch_or_outputs(
                self.key, batch, outputs, caller_name="OnlineQueue"
            )
            if data is None:
                return

            # If dim is specified as a single int and data is 1D, add a dimension
            if isinstance(self.dim, int) and data.dim() == 1:
                data = data.unsqueeze(1)

            self._queue.append(data)

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
