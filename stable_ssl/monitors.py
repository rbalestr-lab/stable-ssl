"""Training/evaluation metrics that are computed at the end of each step."""
#
# Author: @sami-bg
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import deque
from typing import Union

import torch
import torch.distributed as dist

from stable_ssl.utils import gather, warn_once


class Monitor:
    """Base class for metrics that are monitored at the end of each step.

    Inheritors must implement a `compute` method, that calculates the metric,
    and a `name` attribute for logging.
    """

    name: str = "monitor"

    def compute(self, x):
        """Abstract method that calculates a score given a model."""
        pass


class RankMe(Monitor):
    """RankMe (effective rank) monitor from :cite:`garrido2023rankme`."""

    name = "rankme"

    def __init__(self, limit: int = 8, epsilon: float = 1e-7):
        super().__init__()
        self.global_limit = limit

        num_devices = 1
        if dist.is_available() and dist.is_initialized():
            num_devices = dist.get_world_size()

        assert (
            self.global_limit % num_devices == 0
        ), f"RankMe {limit=} must be divisible by {num_devices=}"
        assert (
            self.global_limit % num_devices == 0
        ), f"RankMe {limit=} must be divisible by {num_devices=}"
        self.device_limit = self.global_limit // num_devices

        self.epsilon = epsilon
        self.bounded_queue = deque(maxlen=self.device_limit)

    @staticmethod
    def _calculate_rankme(x: torch.Tensor, epsilon: float) -> torch.Tensor:
        # NOTE torch.linalg.svd only supports torch.float32 for now
        if x.dtype != torch.float32:
            warn_once(
                f"RankMe expected tensors of type {torch.float32}, "
                f"but received {x.dtype}, will convert "
                f"{x.dtype}->{torch.float32}."
            )
            x = x.to(torch.float32)

        _u, s, _vh = torch.linalg.svd(x, full_matrices=False)
        p = (s / torch.sum(s, axis=0)) + epsilon
        entropy = -torch.sum(p * torch.log(p))
        return torch.exp(entropy)

    def compute(self, encoding: Union[list, torch.Tensor]) -> float:
        if isinstance(encoding, list):
            # assume a list is of views, where each view is batch_size on the 0th dim
            # (as per JointEmbeddng)
            return [self.compute(batch) for batch in encoding][-1]

        batch_size, *_ = encoding.shape
        encoding = encoding.reshape(batch_size, -1)

        self.bounded_queue.append(encoding)

        encoding = torch.cat(list(self.bounded_queue), dim=0)
        encoding = gather(encoding)

        return RankMe._calculate_rankme(encoding, self.epsilon)
