from typing import Dict, Iterable, Union

import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from loguru import logger as logging

from stable_ssl.utils import UnsortedQueue, broadcast_param_to_list


class OnlineQueue(Callback):
    """Maintain per-key circular buffers during training and snapshot them during validation.

    This callback automatically creates an `UnsortedQueue` for each key in
    `to_save` on your `pl_module`, appends that batch tensor at the end of every
    training batch, and wraps around when it reaches `queue_length`.  At the
    start of validation it takes a read-only snapshot of all queues, storing
    them in `pl_module._callbacks_queue_snapshots[name]`, which downstream
    callbacks can use (e.g. for nearest-neighbor metrics).

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
        gather_distributed (bool, optional):
            If True, automatically gather cached data across all distributed
            processes during validation. Default is False.
    """

    def __init__(
        self,
        pl_module,
        name: str,
        to_save: Union[str, Iterable],
        queue_length: int,
        dims: Union[list[tuple[int]], list[int], int, None] = None,
        dtypes: Union[tuple[int], list[int], int, None] = None,
        gather_distributed: bool = False,
    ) -> None:
        # Normalize to_save to always be a list
        if isinstance(to_save, str):
            to_save = [to_save]

        # Broadcast all parameters to lists matching to_save length
        dims = broadcast_param_to_list(dims, len(to_save), "dims")
        dtypes = broadcast_param_to_list(dtypes, len(to_save), "dtypes")

        logging.info(f"Setting up callback ({name=})")
        logging.info(f"\t- {to_save=}")
        logging.info(f"\t- {dims=}")
        logging.info(f"\t- {dtypes=}")
        logging.info("\t- caching modules into `_callbacks_modules`")
        if name in pl_module._callbacks_modules:
            raise ValueError(f"{name=} already used in callbacks")

        # Initialize shared queues registry if not exists
        if not hasattr(pl_module, "_shared_queues"):
            pl_module._shared_queues = {}
            pl_module._shared_queues_metadata = {}  # Track queue properties

        # Create ModuleDict for this callback
        pl_module._callbacks_modules[name] = torch.nn.ModuleDict()

        # Process each key to save
        for key, dim, dtype in zip(to_save, dims, dtypes):
            # Check if a shared queue already exists for this key
            if key in pl_module._shared_queues:
                existing_meta = pl_module._shared_queues_metadata[key]
                # Check compatibility
                # For dtype: None is compatible with anything (will be inferred)
                dtype_compatible = (
                    existing_meta["dtype"] == dtype
                    or existing_meta["dtype"] is None
                    or dtype is None
                )
                # For dim: None is compatible with anything (will be inferred)
                dim_compatible = (
                    existing_meta["dim"] == dim
                    or existing_meta["dim"] is None
                    or dim is None
                )

                if dtype_compatible and dim_compatible:
                    # Check queue length compatibility
                    if existing_meta["queue_length"] == queue_length:
                        # Fully compatible - reuse queue
                        logging.info(
                            f"\t- Reusing existing queue for '{key}' "
                            f"(length={queue_length}, dim={dim}, dtype={dtype})"
                        )
                        pl_module._callbacks_modules[name][key] = (
                            pl_module._shared_queues[key]
                        )
                        existing_meta["callbacks"].append(name)
                    else:
                        # Different queue length - create separate queue with warning
                        logging.warning(
                            f"Queue for '{key}' already exists with length "
                            f"{existing_meta['queue_length']}, but callback '{name}' "
                            f"needs length {queue_length}. Creating separate queue."
                        )
                        # Create new queue for this callback
                        queue = UnsortedQueue(queue_length, dim, dtype)
                        pl_module._callbacks_modules[name][key] = queue
                else:
                    # Incompatible dims or dtype - raise error
                    incompatible_parts = []
                    if not dim_compatible:
                        incompatible_parts.append(
                            f"dim (existing={existing_meta['dim']}, required={dim})"
                        )
                    if not dtype_compatible:
                        incompatible_parts.append(
                            f"dtype (existing={existing_meta['dtype']}, required={dtype})"
                        )

                    raise ValueError(
                        f"Incompatible queue configuration for '{key}': "
                        f"{' and '.join(incompatible_parts)}. "
                        f"Note: None values are inferred from first batch."
                    )
            else:
                # No existing queue - create new shared queue
                queue = UnsortedQueue(queue_length, dim, dtype)
                pl_module._shared_queues[key] = queue
                pl_module._shared_queues_metadata[key] = {
                    "queue_length": queue_length,
                    "dim": dim,
                    "dtype": dtype,
                    "callbacks": [name],
                    "owner": name,  # First callback to create queue owns the append
                }
                pl_module._callbacks_modules[name][key] = queue
                logging.info(
                    f"\t- Created new shared queue for '{key}' "
                    f"(length={queue_length}, dim={dim}, dtype={dtype})"
                )

        logging.info(
            f"`_callbacks_modules` now contains ({list(pl_module._callbacks_modules.keys())})"
        )
        self.name = name
        self.to_save = to_save
        self.gather_distributed = gather_distributed

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

                # Only append if this callback owns the shared queue or it's not a shared queue
                if (
                    hasattr(pl_module, "_shared_queues_metadata")
                    and key in pl_module._shared_queues_metadata
                ):
                    # This is a shared queue - only append if we're the owner
                    if pl_module._shared_queues_metadata[key]["owner"] == self.name:
                        pl_module._callbacks_modules[self.name][key].append(batch[key])
                else:
                    # Not a shared queue (e.g., different queue_length) - always append
                    pl_module._callbacks_modules[self.name][key].append(batch[key])

    def on_validation_epoch_start(self, trainer, pl_module):
        """Snapshot training queue contents at validation start for other callbacks to access.

        This creates a temporary snapshot of the training data stored in queues,
        making it available in pl_module._callbacks_queue_snapshots[self.name].
        The snapshot is cleared at validation end.

        If gather_distributed is True and world_size > 1, automatically gathers
        data across all distributed processes.
        """
        logging.info(f"{self.name}: validation epoch start, caching queue(s)")

        if not hasattr(pl_module, "_callbacks_queue_snapshots"):
            pl_module._callbacks_queue_snapshots = {}

        pl_module._callbacks_queue_snapshots[self.name] = {}
        should_gather = self.gather_distributed and trainer.world_size > 1

        if should_gather:
            logging.info(
                f"{self.name}: Caching and gathering distributed data across {trainer.world_size} processes"
            )

        for key in self.to_save:
            queue = pl_module._callbacks_modules[self.name][key]
            tensor = queue.get()

            # Gather if distributed and requested
            if should_gather:
                gathered_tensor = pl_module.all_gather(tensor).flatten(0, 1)
                pl_module._callbacks_queue_snapshots[self.name][key] = gathered_tensor
                logging.info(
                    f"\t- {key}: {tensor.shape} -> {gathered_tensor.shape} (gathered)"
                )
            else:
                pl_module._callbacks_queue_snapshots[self.name][key] = tensor
                logging.info(f"\t- {key}: {tensor.shape}, {tensor.dtype}")

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Clean up the queue snapshot at the end of validation epoch.

        This removes the temporary snapshot of training queue contents that was created
        at validation start. Only the snapshot in pl_module._callbacks_queue_snapshots
        is deleted; the actual training queues remain intact.
        """
        logging.info(f"{self.name}: validation epoch end, cleaning up cache")
        if (
            hasattr(pl_module, "_callbacks_queue_snapshots")
            and self.name in pl_module._callbacks_queue_snapshots
        ):
            del pl_module._callbacks_queue_snapshots[self.name]

    def get_queue_snapshot(
        self, pl_module: LightningModule
    ) -> Union[Dict[str, torch.Tensor], None]:
        """Safely retrieve the queue snapshot for this callback.

        This method handles all the validation and logging for accessing queue snapshots,
        making it easy for child classes to get their cached data.

        Args:
            pl_module: The Lightning module

        Returns:
            Dictionary containing the queue snapshots if available, None otherwise.
            Returns None with appropriate warning logs if snapshots are not available.
        """
        if not hasattr(pl_module, "_callbacks_queue_snapshots"):
            logging.warning(
                f"{self.name}: No queue snapshots found on pl_module. "
                "Ensure OnlineQueue callbacks run before this callback."
            )
            return None

        if self.name not in pl_module._callbacks_queue_snapshots:
            logging.warning(
                f"{self.name}: No queue snapshot found for this callback. "
                f"Available snapshots: {list(pl_module._callbacks_queue_snapshots.keys())}"
            )
            return None

        return pl_module._callbacks_queue_snapshots[self.name]
