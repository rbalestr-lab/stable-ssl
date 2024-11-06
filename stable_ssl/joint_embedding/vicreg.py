# -*- coding: utf-8 -*-
"""VICReg model."""
#
# Author: Randall Balestriero <randallbalestriero@gmail.com>
#         Hugues Van Assel <vanasselhugues@gmail.com>
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
import torch

from stable_ssl.utils import off_diagonal, gather_processes
from .base import JointEmbeddingConfig, JointEmbeddingModel


class VICReg(JointEmbeddingModel):
    """VICReg model from [BPL21]_.

    Reference
    ---------
    .. [BPL21] Bardes, A., Ponce, J., & LeCun, Y. (2021).
            VICReg: Variance-Invariance-Covariance Regularization
            For Self-Supervised Learning.
            International Conference on Learning Representations (ICLR).
    """

    @gather_processes
    def compute_ssl_loss(self, z_i, z_j):
        repr_loss = torch.nn.functional.mse_loss(z_i, z_j)

        x = z_i - z_i.mean(dim=0)
        y = z_j - z_j.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + self.config.model.epsilon)
        std_y = torch.sqrt(y.var(dim=0) + self.config.model.epsilon)
        std_loss = (
            torch.mean(torch.nn.functional.relu(1 - std_x)) / 2
            + torch.mean(torch.nn.functional.relu(1 - std_y)) / 2
        )

        cov_x = (x.T @ x) / (x.size(0) - 1)
        cov_y = (y.T @ y) / (x.size(0) - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(x.size(1)) + off_diagonal(
            cov_y
        ).pow_(2).sum().div(x.size(1))

        loss = (
            self.config.model.sim_coeff * repr_loss
            + self.config.model.std_coeff * std_loss
            + self.config.model.cov_coeff * cov_loss
        )
        return loss


@dataclass
class VICRegConfig(JointEmbeddingConfig):
    """Configuration for the VICreg model parameters."""

    sim_coeff: float = 25
    std_coeff: float = 25
    cov_coeff: float = 1
    epsilon: float = 0.0001

    def trainer(self):
        return VICReg
