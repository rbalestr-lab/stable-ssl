"""RankMe callback using the new queue discovery architecture."""

from typing import Iterable, Union

import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from loguru import logger as logging

from .queue import find_or_create_queue_callback


class RankMe(Callback):
    """RankMe (effective rank) monitor using queue discovery.

    RankMe measures the effective rank of feature representations by computing
    the exponential of the entropy of normalized singular values. This metric
    helps detect dimensional collapse in self-supervised learning.

    Args:
        name: Unique name for this callback instance
        target: Key in batch dict containing the feature embeddings to monitor
        queue_length: Required queue length
        target_shape: Shape of the target embeddings (e.g., 768 for 768-dim features)
    """

    def __init__(
        self,
        name: str,
        target: str,
        queue_length: int,
        target_shape: Union[int, Iterable[int]],
    ) -> None:
        super().__init__()

        # Convert shape to int if it's a single-element tuple/list
        if isinstance(target_shape, (list, tuple)):
            if len(target_shape) == 1:
                target_shape = target_shape[0]
            else:
                target_shape = int(torch.prod(torch.tensor(target_shape)))

        self.name = name
        self.target = target
        self.queue_length = queue_length
        self.target_shape = target_shape

        # Queue reference will be set in setup
        self._target_queue = None

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Find or create the queue callback for target features."""
        # Setup is needed for all stages, not just fit
        if self._target_queue is None:
            self._target_queue = find_or_create_queue_callback(
                trainer,
                self.target,
                self.queue_length,
                self.target_shape,
                torch.float32,  # RankMe typically uses float features
                gather_distributed=True,  # RankMe typically needs gathering
                create_if_missing=True,
            )
            logging.info(f"{self.name}: Using queue for target '{self.target}'")

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict,
        batch: dict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Compute RankMe metric on the first validation batch only."""
        # Only compute on first batch
        if batch_idx > 0:
            return

        logging.info(f"{self.name}: Computing RankMe on first validation batch")

        # Get cached features from queue
        embeddings = self._target_queue.data

        if embeddings is None:
            logging.warning(
                f"{self.name}: Queue data not available (not in validation?)"
            )
            return

        # Compute RankMe on rank 0 only
        if trainer.global_rank == 0:
            with torch.no_grad():
                # Compute singular values
                s = torch.linalg.svdvals(embeddings)

                # Normalize to get probability distribution
                p = (s / torch.sum(s, axis=0)) + 1e-5

                # Compute entropy
                entropy = -torch.sum(p * torch.log(p))

                # RankMe = exp(entropy)
                rankme = torch.exp(entropy)

                # Log the metric
                pl_module.log(self.name, rankme.item())
