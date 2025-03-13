"""Training/evaluation metrics that are computed at the end of each step."""

#
# Author: @sami-bg
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
from collections import deque

import torch
import torch.distributed as dist

from stable_ssl import (
    BaseTrainer,
    JointEmbeddingPredictiveTrainer,
    JointEmbeddingTrainer,
)
from stable_ssl.utils import broadcast, gather, reduce, warn_once


def get_num_devices() -> int:
    """Return the number of devices used in this run."""
    num_devices = 1
    if dist.is_available() and dist.is_initialized():
        num_devices = dist.get_world_size()
    return num_devices


class Monitor:
    """Base class for metrics that are monitored at the end of each step.

    Inheritors must implement a `compute` method, that calculates the metric,
    and a `name` attribute for logging.
    """

    name: str = "monitor"

    def compute(self, trainer: BaseTrainer):
        """Abstract method that calculates a score given a model."""
        pass


class RankMe(Monitor):
    """RankMe (effective rank) monitor from :cite:`garrido2023rankme`."""

    name = "rankme"

    def __init__(self, limit: int = 8, epsilon: float = 1e-7):
        super().__init__()
        self.global_limit = limit

        self.num_devices = get_num_devices()

        assert self.global_limit % self.num_devices == 0, (
            f"RankMe {limit=} must be divisible by {self.num_devices=}"
        )
        self.device_limit = self.global_limit // self.num_devices

        self.epsilon = epsilon
        self.bounded_queue = deque(maxlen=self.device_limit)

    def rankme(self, encoding: torch.Tensor, epsilon: float) -> float:
        (batch_size, *_), device = encoding.shape, encoding.device
        encoding = encoding.reshape(batch_size, -1)

        self.bounded_queue.append(encoding)

        encoding = torch.cat(list(self.bounded_queue), dim=0)
        encoding = gather(encoding, rank=0)

        # NOTE torch.linalg.svd only supports torch.float32 for now
        if encoding.dtype != torch.float32:
            warn_once(
                f"RankMe expected tensors of type {torch.float32}, "
                f"but received {encoding.dtype}, will convert "
                f"{encoding.dtype}->{torch.float32}"
            )
            encoding = encoding.to(torch.float32)

        _u, s, _vh = torch.linalg.svd(encoding, full_matrices=False)
        p = (s / torch.sum(s, axis=0)) + epsilon
        entropy = -torch.sum(p * torch.log(p))

        rankme = float(torch.exp(entropy))
        return broadcast(torch.tensor([rankme], device=device), src_rank=0).item()

    def compute(self, trainer: BaseTrainer) -> float:
        if not isinstance(
            trainer, (JointEmbeddingTrainer, JointEmbeddingPredictiveTrainer)
        ):
            raise NotImplementedError(
                f"RankMe only implemented for JointEmbeddingTrainer, JointEmbeddingPredictiveTrainer "
                f"and not yet implemented for type {type(trainer)}"
            )

        encoding: list | torch.Tensor = trainer.latest_representations
        if isinstance(encoding, list):
            # assume a list is of views, where each view is batch_size on the 0th dim
            # (as per JointEmbeddng)
            return [self.compute(batch) for batch in encoding][-1]

        return self.rankme(encoding, self.epsilon)


class LiDAR(Monitor):
    """LiDAR (Linear Discriminant Analysis Rank) monitor from :cite`thilak2023lidar`."""

    name = "LiDAR"

    def __init__(self, n: int = 1000, epsilon: float = 1e-7, delta: float = 1e-3):
        super().__init__()
        self.n = n
        self.global_limit = n
        self.epsilon = epsilon
        self.delta = delta

        self.num_devices = get_num_devices()

        self.queue = None
        self.device_limit = self.global_limit // self.num_devices

    def _init_bounded_queue(self, batch_size: int) -> None:
        # NOTE Dynamically create queue, rounding it's capacity to
        # the nearest batch size.
        self.device_limit = (self.device_limit // batch_size) * batch_size
        self.global_limit = self.device_limit * self.num_devices
        self.queue = deque(maxlen=self.device_limit)

        if self.global_limit != self.n:
            warn_once(
                f"Received n={self.n} but rounded to {self.global_limit}. "
                f"To avoid this, make sure n={self.n} and your batch size "
                f"({batch_size}) are divisible."
            )
        logging.info(f"Initialized LiDAR with n={self.global_limit}")
        return

    def lidar(self, batch_embeddings: list[torch.Tensor]) -> float:
        if not self.queue:
            batch_size = len(batch_embeddings)
            self._init_bounded_queue(batch_size)

        self.queue.extend(batch_embeddings)

        embeddings: torch.Tensor = torch.stack(list(self.queue), dim=0)
        (local_n, q, d), device = embeddings.shape, embeddings.device

        n_total_tensor = torch.tensor([local_n], device=device)
        n_total_tensor = reduce(n_total_tensor, rank=0, op=dist.ReduceOp.SUM)

        n_total_tensor = broadcast(n_total_tensor, src_rank=0)

        if (n_total := n_total_tensor.item()) == 1:
            warn_once("LiDAR cannot compute within-class scatter with only one class!")
            return 1.0

        class_means = embeddings.mean(dim=1)
        grand_mean_local = class_means.mean(dim=0)

        local_Sb = torch.zeros(d, d, device=device)
        local_Sw = torch.zeros(d, d, device=device)

        for i in range(local_n):
            diff_b = (class_means[i] - grand_mean_local).unsqueeze(1)
            local_Sb += diff_b @ diff_b.T
            for j in range(q):
                diff_w = (embeddings[i, j] - class_means[i]).unsqueeze(1)
                local_Sw += diff_w @ diff_w.T
        S_b = reduce(local_Sb, rank=0, op=dist.ReduceOp.SUM) / (n_total - 1)
        S_w = reduce(local_Sw, rank=0, op=dist.ReduceOp.SUM) / (n_total * (q - 1))
        S_w += self.delta * torch.eye(d, device=device)

        eigvals_w, eigvecs_w = torch.linalg.eigh(S_w)
        eigvals_w = torch.clamp(eigvals_w, min=self.epsilon)

        invsqrt_w = (eigvecs_w * (1.0 / torch.sqrt(eigvals_w))) @ eigvecs_w.transpose(
            -1, -2
        )
        Sigma_lidar = invsqrt_w @ S_b @ invsqrt_w

        lam, _ = torch.linalg.eigh(Sigma_lidar)
        lam = torch.clamp(lam, min=0.0)

        lam_sum = lam.sum() + self.epsilon
        p = lam / lam_sum

        p_log_p = p * torch.log(p + self.epsilon)

        lidar = float(torch.exp(-p_log_p.sum()))
        return broadcast(torch.tensor([lidar], device=device), src_rank=0).item()

    def compute(self, trainer: BaseTrainer) -> float:
        if not isinstance(
            trainer, (JointEmbeddingTrainer, JointEmbeddingPredictiveTrainer)
        ):
            raise NotImplementedError(
                f"LiDAR only implemented for JointEmbeddingTrainer, JointEmbeddingPredictiveTrainer "
                f"and not yet implemented for type {type(trainer)}"
            )

        trainer: JointEmbeddingTrainer | JointEmbeddingPredictiveTrainer
        embeddings: list[torch.Tensor] = trainer.latest_embeddings
        return self.lidar(embeddings)
