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

from stable_ssl.utils import FullGatherLayer, off_diagonal
from .base import JEConfig, JETrainer


class VICReg(JETrainer):
    """VICReg model from [BPL21]_.

    Reference
    ---------
    .. [BPL21] Bardes, A., Ponce, J., & LeCun, Y. (2021).
            VICReg: Variance-Invariance-Covariance Regularization
            For Self-Supervised Learning.
            International Conference on Learning Representations (ICLR).
    """

    def compute_ssl_loss(self, z1, z2):

        repr_loss = torch.nn.functional.mse_loss(z1, z2)

        # if self.config.hardware.world_size > 1:
        #     x = torch.cat(FullGatherLayer.apply(z1), dim=0)
        #     y = torch.cat(FullGatherLayer.apply(z2), dim=0)
        # else:
        #     x = z1
        #     y = z2
        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)

        std_z1 = torch.sqrt(z1.var(dim=0) + self.config.model.epsilon)
        std_z2 = torch.sqrt(z2.var(dim=0) + self.config.model.epsilon)
        std_loss = (
            torch.mean(torch.nn.functional.relu(1 - std_z1)) / 2
            + torch.mean(torch.nn.functional.relu(1 - std_z2)) / 2
        )

        cov_z1 = (z1.T @ z1) / (z1.size(0) - 1)
        cov_y = (z2.T @ z2) / (z2.size(0) - 1)
        cov_loss = off_diagonal(cov_z1).pow_(2).sum().div(z1.size(1)) + off_diagonal(
            cov_z2
        ).pow_(2).sum().div(z1.size(1))

        loss = (
            self.config.model.sim_coeff * repr_loss
            + self.config.model.std_coeff * std_loss
            + self.config.model.cov_coeff * cov_loss
        )
        return loss


@dataclass
class VICRegConfig(JEConfig):
    """Configuration for the VICreg model parameters."""

    sim_coeff: float = 25
    std_coeff: float = 25
    cov_coeff: float = 1
    epsilon: float = 0.0001

    def trainer(self):
        return VICReg
