from typing import Dict, Iterable

import torch
from lightning.pytorch import LightningModule, Trainer
from loguru import logger as logging

from .queue import OnlineQueue


class RankMe(OnlineQueue):
    """RankMe (effective rank) monitor from :cite:`garrido2023rankme`.

    RankMe measures the effective rank of feature representations by computing
    the exponential of the entropy of normalized singular values. This metric
    helps detect dimensional collapse in self-supervised learning, where the
    model might only use a subset of available dimensions.

    The metric is computed as:
        1. Compute SVD of the feature matrix to get singular values
        2. Normalize singular values to get a probability distribution
        3. Compute entropy of this distribution
        4. RankMe = exp(entropy)

    Higher RankMe values indicate more dimensions are being effectively used,
    while lower values suggest dimensional collapse.

    Args:
        pl_module: PyTorch Lightning module to attach the callback to.
        name: Unique name for this callback instance.
        target: Key in batch dict containing the feature embeddings to monitor.
        queue_length: Maximum number of samples to store in the queue.
        target_shape: Shape of the target embeddings (e.g., [768] for 768-dim features).
    """

    def __init__(
        self,
        pl_module,
        name: str,
        target: str,
        queue_length: int,
        target_shape: Iterable[int],
    ) -> None:
        super().__init__(
            pl_module,
            name=name,
            to_save=[target],
            queue_length=queue_length,
            dims=[target_shape],
            dtypes=[torch.float],
        )

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Dict,
        batch: Dict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Compute RankMe metric on the first validation batch only.

        RankMe (effective rank) is computed as exp(entropy) of the normalized
        singular values of the feature matrix. This metric helps monitor the
        dimensional collapse in self-supervised learning representations.
        """
        # Only compute on first batch (not possible to accumulate ranks across batches)
        if batch_idx > 0:
            return

        logging.info(f"{self.name}: batch 0 of validation step, computing RankMe")

        # Get cached embeddings from parent's validation cache
        if not hasattr(pl_module, "_callbacks_validation_cache"):
            logging.warning(f"{self.name}: No validation cache found")
            return

        if self.name not in pl_module._callbacks_validation_cache:
            logging.warning(f"{self.name}: No cached data found in validation cache")
            return

        embeddings = list(pl_module._callbacks_validation_cache[self.name].values())[0]

        # Gather embeddings from all processes
        embeddings = pl_module.all_gather(embeddings).flatten(0, 1)

        # Compute RankMe on rank 0 only
        if trainer.global_rank == 0:
            s = torch.linalg.svdvals(embeddings)
            p = (s / torch.sum(s, axis=0)) + 1e-5
            entropy = -torch.sum(p * torch.log(p))
            rankme = torch.exp(entropy)
            pl_module.log(self.name, rankme.item())
