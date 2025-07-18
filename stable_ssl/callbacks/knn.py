from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule, Trainer
from loguru import logger as logging
from torch import Tensor

from stable_ssl.utils.distance_metrics import compute_pairwise_distances_chunked

from .queue import OnlineQueue
from .utils import format_metrics_as_dict


class OnlineKNN(OnlineQueue):
    """Weighted KNN online evaluator for self-supervised learning.

    The weighted KNN classifier matches sec 3.4 of https://arxiv.org/pdf/1805.01978.pdf.
    The implementation follows:
        1. https://github.com/zhirongw/lemniscate.pytorch/blob/master/test.py
        2. https://github.com/leftthomas/SimCLR
        3. https://github.com/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb

    Args:
        name: Unique identifier for this callback instance
        input: Key in batch dict containing input features
        target: Key in batch dict containing target labels
        queue_length: Maximum number of samples to store in feature bank
        metrics: Dictionary of metrics to compute during validation
        input_dim: Dimensionality of input features (can be int or shape tuple/list). If None, inferred from data.
        target_dim: Should be 1 for classification (single label per sample). If None, defaults to 1.
        k: Number of nearest neighbors to consider (must be positive)
        temperature: Temperature parameter for distance weighting (must be > 0)
        chunk_size: Batch size for memory-efficient distance computation. Use -1 to process all data at once
        distance_metric: Distance metric to use for KNN computation. Options:
            - "euclidean": L2 distance (default)
            - "squared_euclidean": Squared L2 distance
            - "cosine": Cosine distance (1 - cosine_similarity)
            - "manhattan": L1 distance
    """

    NAME = "OnlineKNN"

    def __init__(
        self,
        pl_module: LightningModule,
        name: str,
        input: str,
        target: str,
        queue_length: int,
        metrics: Dict,
        input_dim: Union[Tuple[int, ...], List[int], int, None] = None,
        target_dim: Union[int, None] = 1,
        k: int = 5,
        temperature: float = 0.07,
        chunk_size: int = -1,
        distance_metric: Literal[
            "euclidean", "squared_euclidean", "cosine", "manhattan"
        ] = "euclidean",
    ) -> None:
        # Validate inputs
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if chunk_size == 0 or chunk_size < -1:
            raise ValueError(
                f"chunk_size must be positive or -1 (no chunking), got {chunk_size}"
            )

        # Process input_dim
        if input_dim is not None and isinstance(input_dim, (list, tuple)):
            input_dim = int(np.prod(input_dim))

        # Initialize parent OnlineQueue with both input and target
        super().__init__(
            pl_module=pl_module,
            name=name,
            to_save=[input, target],
            queue_length=queue_length,
            dims=[input_dim, target_dim],  # Both can be None for inference
            dtypes=[torch.float, torch.long],  # Features are float32, labels are long
            gather_distributed=True,  # Enable automatic distributed gathering
        )

        self.input = input
        self.target = target
        self.k = k
        self.temperature = temperature
        self.chunk_size = chunk_size
        self.distance_metric = distance_metric

        # Add metrics
        logging.info("\t- caching metrics into `_callbacks_metrics`")
        pl_module._callbacks_metrics[name] = format_metrics_as_dict(metrics)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Dict,
        batch: Dict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Compute KNN predictions during validation."""
        # Get validation cache using parent's method
        cached_data = self.get_queue_snapshot(pl_module)
        if cached_data is None:
            return
        cached_X = cached_data.get(self.input)
        cached_y = cached_data.get(self.target)

        if cached_X is None or cached_X.size(0) == 0:
            logging.warning(
                f"{self.name}: No cached data found for input key '{self.input}' "
                f"or cache is empty. Available keys: {list(cached_data.keys())}"
            )
            return

        # Check if all required keys are present in the validation batch
        for key in self.to_save:
            if key not in batch:
                logging.warning(
                    f"{self.name}: Required key '{key}' not found in validation batch. "
                    f"Available keys: {list(batch.keys())}"
                )
                return

        predictions = self._compute_knn_predictions(
            batch, pl_module, cached_X, cached_y
        )

        if predictions is not None:
            prediction_key = f"{self.name}_preds"
            if prediction_key in batch:
                msg = (
                    f"Asking to save predictions for callback `{self.name}` "
                    f"but `{prediction_key}` already exists in the batch dict."
                )
                logging.error(msg)
                raise ValueError(msg)
            batch[prediction_key] = predictions

            self._log_metrics(pl_module, predictions, batch[self.target])

    @torch.no_grad()
    def _compute_knn_predictions(
        self,
        batch: Dict,
        pl_module: LightningModule,
        cached_X: Tensor,
        cached_y: Tensor,
    ) -> Optional[Tensor]:
        """Compute KNN predictions with memory-efficient chunked processing."""
        features = batch[self.input]
        batch_size = features.size(0)
        num_classes = cached_y.max().item() + 1

        predictions = torch.zeros(
            batch_size, num_classes, device=features.device, dtype=torch.float32
        )

        # Ensure tensors are on the same device for distance computation
        if cached_X.device != features.device:
            cached_X = cached_X.to(features.device)
            cached_y = cached_y.to(features.device)

        k_actual = min(self.k, cached_X.size(0))

        chunk_size = batch_size if self.chunk_size == -1 else self.chunk_size
        dist_matrix = compute_pairwise_distances_chunked(
            cached_X, features, metric=self.distance_metric, chunk_size=chunk_size
        )
        dist_weight, sim_indices = dist_matrix.topk(k=k_actual, dim=0, largest=False)
        # 1/(d+T) is a monotonic proxy for exp(-d/T)
        dist_weight = 1 / dist_weight.add_(self.temperature)

        one_hot_labels = F.one_hot(cached_y[sim_indices], num_classes=num_classes)

        # Weighted voting
        predictions = (dist_weight.unsqueeze(-1) * one_hot_labels).sum(0)
        return predictions

    def _log_metrics(
        self, pl_module: LightningModule, predictions: Tensor, targets: Tensor
    ) -> None:
        """Compute and log validation metrics."""
        logs = {}
        for metric_name, metric in pl_module._callbacks_metrics[self.name][
            "_val"
        ].items():
            metric(predictions, targets)
            logs[f"eval/{self.name}_{metric_name}"] = metric

        pl_module.log_dict(logs, on_step=False, on_epoch=True)

    # on_validation_epoch_end is handled by parent OnlineQueue
