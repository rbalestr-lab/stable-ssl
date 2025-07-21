"""KNN callback."""

from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch import Callback, LightningModule, Trainer
from loguru import logger as logging
from torch import Tensor

from stable_ssl.utils import get_data_from_batch_or_outputs
from stable_ssl.utils.distance_metrics import compute_pairwise_distances_chunked

from .queue import find_or_create_queue_callback
from .utils import format_metrics_as_dict


class OnlineKNN(Callback):
    """Weighted KNN online evaluator using queue discovery.

    This callback finds OnlineQueue callbacks that track the required features
    and labels, then uses that data for KNN evaluation during validation.

    Args:
        name: Unique identifier for this callback instance
        input: Key in batch dict containing input features
        target: Key in batch dict containing target labels
        queue_length: Required queue length for both input and target
        metrics: Dictionary of metrics to compute during validation
        input_dim: Dimensionality of input features (None to accept any)
        target_dim: Dimensionality of targets (None to accept any)
        k: Number of nearest neighbors to consider
        temperature: Temperature parameter for distance weighting
        chunk_size: Batch size for memory-efficient distance computation (-1 for no chunking)
        distance_metric: Distance metric to use for KNN computation
    """

    def __init__(
        self,
        name: str,
        input: str,
        target: str,
        queue_length: int,
        metrics: Dict,
        input_dim: Optional[Union[Tuple[int, ...], List[int], int]] = None,
        target_dim: Optional[int] = None,
        k: int = 5,
        temperature: float = 0.07,
        chunk_size: int = -1,
        distance_metric: Literal[
            "euclidean", "squared_euclidean", "cosine", "manhattan"
        ] = "euclidean",
    ) -> None:
        super().__init__()

        # Validate inputs
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if chunk_size == 0 or chunk_size < -1:
            raise ValueError(f"chunk_size must be positive or -1, got {chunk_size}")

        # Process input_dim
        if input_dim is not None and isinstance(input_dim, (list, tuple)):
            input_dim = int(np.prod(input_dim))

        self.name = name
        self.input = input
        self.target = target
        self.queue_length = queue_length
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.k = k
        self.temperature = temperature
        self.chunk_size = chunk_size
        self.distance_metric = distance_metric
        self.metrics = metrics

        # Queue references will be set in setup
        self._input_queue = None
        self._target_queue = None

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Find or create queue callbacks and setup metrics."""
        # Setup is needed for all stages, not just fit
        if self._input_queue is None or self._target_queue is None:
            # Find or create queue for input features
            self._input_queue = find_or_create_queue_callback(
                trainer,
                self.input,
                self.queue_length,
                self.input_dim,
                torch.float32 if self.input_dim is not None else None,
                gather_distributed=True,  # KNN typically needs gathering
                create_if_missing=True,
            )
            logging.info(f"{self.name}: Using queue for input '{self.input}'")

            # Find or create queue for target labels
            self._target_queue = find_or_create_queue_callback(
                trainer,
                self.target,
                self.queue_length,
                self.target_dim,
                torch.long if self.target_dim is not None else None,
                gather_distributed=True,  # KNN typically needs gathering
                create_if_missing=True,
            )
            logging.info(f"{self.name}: Using queue for target '{self.target}'")

            # Setup metrics
            logging.info(f"{self.name}: Setting up metrics")
            if not hasattr(pl_module, "_callbacks_metrics"):
                pl_module._callbacks_metrics = {}
            pl_module._callbacks_metrics[self.name] = format_metrics_as_dict(
                self.metrics
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
        """Compute KNN predictions during validation."""
        # Get input and target data from batch or outputs
        input_data = get_data_from_batch_or_outputs(
            self.input, batch, outputs, caller_name=self.name
        )
        if input_data is None:
            return

        target_data = get_data_from_batch_or_outputs(
            self.target, batch, outputs, caller_name=self.name
        )
        if target_data is None:
            return

        # Get cached data from queues
        cached_features = self._input_queue.data
        cached_labels = self._target_queue.data

        if cached_features is None or cached_labels is None:
            logging.warning(
                f"{self.name}: Queue data not available (not in validation?)"
            )
            return

        # Check if cached data is empty
        if cached_features.numel() == 0 or cached_labels.numel() == 0:
            logging.warning(
                f"{self.name}: Queue data is empty, skipping KNN computation"
            )
            return

        # Compute predictions
        predictions = self._compute_knn_predictions(
            input_data, cached_features, cached_labels
        )

        if predictions is not None:
            # Store predictions
            prediction_key = f"{self.name}_preds"
            if prediction_key in batch:
                raise ValueError(f"Key '{prediction_key}' already exists in batch")
            batch[prediction_key] = predictions

            # Log metrics
            self._log_metrics(pl_module, predictions, batch[self.target])

    @torch.no_grad()
    def _compute_knn_predictions(
        self,
        features: Tensor,
        cached_features: Tensor,
        cached_labels: Tensor,
    ) -> Optional[Tensor]:
        """Compute KNN predictions."""
        batch_size = features.size(0)
        num_classes = int(cached_labels.max().item()) + 1

        predictions = torch.zeros(
            batch_size, num_classes, device=features.device, dtype=torch.float32
        )

        # Ensure tensors are on same device
        if cached_features.device != features.device:
            cached_features = cached_features.to(features.device)
            cached_labels = cached_labels.to(features.device)

        k_actual = min(self.k, cached_features.size(0))

        # Ensure both tensors have the same dtype for distance computation
        if cached_features.dtype != features.dtype:
            # Convert both to float32 for accurate distance computation
            cached_features = cached_features.float()
            features = features.float()

        # Compute distances
        chunk_size = batch_size if self.chunk_size == -1 else self.chunk_size
        dist_matrix = compute_pairwise_distances_chunked(
            cached_features,
            features,
            metric=self.distance_metric,
            chunk_size=chunk_size,
        )

        # Get k nearest neighbors
        dist_weight, sim_indices = dist_matrix.topk(k=k_actual, dim=0, largest=False)

        # Weight by inverse distance
        dist_weight = 1 / dist_weight.add_(self.temperature)

        # One-hot encode labels
        # sim_indices has shape [k_actual, batch_size], cached_labels has shape [N, 1]
        # We need to squeeze cached_labels to 1D before indexing
        labels_1d = (
            cached_labels.squeeze(-1) if cached_labels.dim() > 1 else cached_labels
        )
        # Ensure labels are LongTensor for one_hot
        selected_labels = labels_1d[sim_indices].long()
        one_hot_labels = F.one_hot(selected_labels, num_classes=num_classes)

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
