# -*- coding: utf-8 -*-
"""BarlowTwins model."""
#
# Author: Randall Balestriero <randallbalestriero@gmail.com>
#         Hugues Van Assel <vanasselhugues@gmail.com>
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from stable_ssl.utils import off_diagonal, gather_processes


class BarlowTwinsLoss(torch.nn.Module):
    """BarlowTwins model from [ZJM+21]_.

    Reference
    ---------
    .. [ZJM+21] Zbontar, J., Jing, L., Misra, I., LeCun, Y., & Deny, S. (2021).
            Barlow Twins: Self-Supervised Learning via Redundancy Reduction.
            In International conference on machine learning (pp. 12310-12320). PMLR.
    """

    def __init__(self, lamb: 0.1):
        self.lamb = lamb
        self.bn = torch.nn.LazyBatchNorm1d()

    @gather_processes
    def compute_ssl_loss(self, z_i, z_j):
        # Empirical cross-correlation matrix.
        c = self.bn(z_i).T @ self.bn(z_j)

        # Sum the cross-correlation matrix between all gpus.
        c.div_(self.config.data.train_dataset.batch_size)
        torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.config.model.lambd * off_diag
        return loss
