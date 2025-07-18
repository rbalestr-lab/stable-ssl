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
            gather_distributed=True,
        )
        self.target = target

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Dict,
        batch: Dict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Compute RankMe metric on the first validation batch only."""
        # Only compute on first batch - the queue snapshot already contains all accumulated training data
        if batch_idx > 0:
            return

        logging.info(f"{self.name}: batch 0 of validation step, computing RankMe.")

        # Get validation cache using parent's method
        cached_data = self.get_queue_snapshot(pl_module)
        if cached_data is None:
            return
        embeddings = cached_data[self.target]

        # Compute RankMe on rank 0 only
        if trainer.global_rank == 0:
            s = torch.linalg.svdvals(embeddings)
            p = (s / torch.sum(s, axis=0)) + 1e-5
            entropy = -torch.sum(p * torch.log(p))
            rankme = torch.exp(entropy)
            pl_module.log(self.name, rankme.item())
