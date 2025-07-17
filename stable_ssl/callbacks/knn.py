from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch import Callback, LightningModule, Trainer
from loguru import logger as logging
from torch import Tensor

from ..utils import UnsortedQueue
from .utils import format_metrics_as_dict


class OnlineKNN(Callback):
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
        features_dim: Dimensionality of input features (can be int or shape tuple/list)
        k: Number of nearest neighbors to consider (must be positive)
        temperature: Temperature parameter for distance weighting (must be > 0)
        normalizer: Type of normalization to apply ('batch_norm' or 'layer_norm')
        chunk_size: Batch size for memory-efficient distance computation
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
        features_dim: Union[Tuple[int, ...], List[int], int],
        k: int = 5,
        temperature: float = 0.07,
        normalizer: str = "batch_norm",
        chunk_size: int = 1000,
    ) -> None:
        super().__init__()

        # Validate inputs
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if normalizer not in ["batch_norm", "layer_norm"]:
            raise ValueError(
                f"normalizer must be 'batch_norm' or 'layer_norm', got '{normalizer}'"
            )

        logging.info(f"Setting up callback ({self.NAME})")
        logging.info(f"\t- {input=}")
        logging.info(f"\t- {target=}")
        logging.info(f"\t- {k=}, {temperature=}")
        logging.info(f"\t- {queue_length=}, {chunk_size=}")
        logging.info("\t- caching modules into `_callbacks_modules`")

        if name in pl_module._callbacks_modules:
            raise ValueError(f"{name=} already used in callbacks")

        self.name = name
        self.input = input
        self.target = target
        self.k = k
        self.temperature = temperature
        self.chunk_size = chunk_size

        if isinstance(features_dim, (list, tuple)):
            features_dim = int(np.prod(features_dim))
        self.features_dim = features_dim

        normalizer_module = self._create_normalizer(normalizer, features_dim)

        pl_module._callbacks_modules[name] = torch.nn.ModuleDict(
            {
                "normalizer": normalizer_module,
                "queue_X": UnsortedQueue(queue_length, features_dim),
                "queue_y": UnsortedQueue(queue_length),
            }
        )

        logging.info(
            f"`_callbacks_modules` now contains ({list(pl_module._callbacks_modules.keys())})"
        )
        logging.info("\t- caching metrics into `_callbacks_metrics`")
        pl_module._callbacks_metrics[name] = format_metrics_as_dict(metrics)

        # Note: Cached feature banks are stored on pl_module during validation
        # as _cached_{name}_X and _cached_{name}_y for potential cross-callback access

    def _create_normalizer(self, normalizer: str, features_dim: int) -> torch.nn.Module:
        """Create the appropriate normalizer module."""
        if normalizer == "batch_norm":
            return torch.nn.BatchNorm1d(features_dim, affine=False)
        elif normalizer == "layer_norm":
            return torch.nn.LayerNorm(
                features_dim, elementwise_affine=False, bias=False
            )
        else:
            return torch.nn.Identity()

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Dict,
        batch: Dict,
        batch_idx: int,
    ) -> None:
        """Store features and labels during training."""
        if self.input not in batch:
            logging.warning(f"Input key '{self.input}' not found in batch")
            return
        if self.target not in batch:
            logging.warning(f"Target key '{self.target}' not found in batch")
            return

        with torch.no_grad():
            features = batch[self.input]
            normalizer = pl_module._callbacks_modules[self.name]["normalizer"]
            normalizer = normalizer.to(features.device)
            normalizer.train()  # to update running statistics
            normalized = normalizer(features)
            pl_module._callbacks_modules[self.name]["queue_X"].append(normalized)
            pl_module._callbacks_modules[self.name]["queue_y"].append(
                batch[self.target]
            )

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Gather features from all processes at validation start."""
        logging.info(
            f"(Validation epoch start, {self.name}) gather queue from all processes"
        )

        qx = pl_module._callbacks_modules[self.name]["queue_X"]
        qy = pl_module._callbacks_modules[self.name]["queue_y"]

        X = qx.get()
        y = qy.get()

        # Handle distributed training
        if trainer.world_size > 1:
            X = pl_module.all_gather(X).flatten(0, 1)
            y = pl_module.all_gather(y).flatten(0, 1)

        # Store on pl_module for potential cross-callback access
        setattr(pl_module, f"_cached_{self.name}_X", X)
        setattr(pl_module, f"_cached_{self.name}_y", y)

        if X.size(0) > 0:
            logging.info(
                f"(Validation epoch start, {self.name}) X cache: {X.shape}, y cache: {y.shape}"
            )
        else:
            logging.warning(f"(Validation epoch start, {self.name}) Empty feature bank")

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
        cached_X = getattr(pl_module, f"_cached_{self.name}_X", None)
        cached_y = getattr(pl_module, f"_cached_{self.name}_y", None)

        if cached_X is None or cached_X.size(0) == 0:
            return

        if self.input not in batch:
            logging.warning(f"Input key '{self.input}' not found in batch")
            return
        if self.target not in batch:
            logging.warning(f"Target key '{self.target}' not found in batch")
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

    def _compute_knn_predictions(
        self,
        batch: Dict,
        pl_module: LightningModule,
        cached_X: Tensor,
        cached_y: Tensor,
    ) -> Optional[Tensor]:
        """Compute KNN predictions with memory-efficient chunked processing."""
        with torch.no_grad():
            features = batch[self.input]
            normalizer = pl_module._callbacks_modules[self.name]["normalizer"]

            normalizer = normalizer.to(features.device)
            normalizer.eval()
            normalized = normalizer(features)
            batch_size = normalized.size(0)
            num_classes = cached_y.max().item() + 1

            predictions = torch.zeros(
                batch_size, num_classes, device=features.device, dtype=torch.float32
            )

            if cached_X.device != features.device:
                cached_X = cached_X.to(features.device)
                cached_y = cached_y.to(features.device)

            # Process in chunks to save memory
            for i in range(0, batch_size, self.chunk_size):
                end_idx = min(i + self.chunk_size, batch_size)
                chunk = normalized[i:end_idx]

                dist_matrix = torch.cdist(cached_X, chunk)

                # Use min to handle case where we have fewer samples than k
                k_actual = min(self.k, cached_X.size(0))
                dist_weight, sim_indices = dist_matrix.topk(
                    k=k_actual, dim=0, largest=False
                )

                # Avoid division by zero with proper temperature handling
                dist_weight = 1 / dist_weight.add_(self.temperature)

                one_hot_labels = F.one_hot(
                    cached_y[sim_indices], num_classes=num_classes
                )

                # Weighted voting
                chunk_preds = (dist_weight.unsqueeze(-1) * one_hot_labels).sum(0)
                predictions[i:end_idx] = chunk_preds

            return predictions.detach()

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

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Clean up cached data after validation."""
        logging.info(f"(Validation epoch end, {self.name}) cleanup")
        # Remove cached attributes from pl_module
        if hasattr(pl_module, f"_cached_{self.name}_X"):
            delattr(pl_module, f"_cached_{self.name}_X")
        if hasattr(pl_module, f"_cached_{self.name}_y"):
            delattr(pl_module, f"_cached_{self.name}_y")
