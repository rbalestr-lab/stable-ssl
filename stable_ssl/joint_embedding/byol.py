# -*- coding: utf-8 -*-
"""BYOL model."""
#
# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#         Randall Balestriero <randallbalestriero@gmail.com>
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch

from .base import SelfDistillationModel, SelfDistillationConfig


class BYOL(SelfDistillationModel):
    """BYOL model from [GSA+20].

    Reference
    ---------
    .. [GSA+20] Grill, J. B., Strub, F., Altché, ... & Valko, M. (2020).
            Bootstrap Your Own Latent-A New Approach To Self-Supervised Learning.
            Advances in neural information processing systems, 33, 21271-21284.
    """

    def compute_ssl_loss(self, projections, projections_target):
        """Compute the loss of the BYOL model.

        Parameters
        ----------
        projections : list of torch.Tensor
            Projections of the different augmented views from the online network.
        projections_target : list of torch.Tensor
            Projections of the corresponding augmented views from the target network.

        Returns
        -------
        float
            The computed loss.
        """
        if len(projections) > 2 or len(projections_target) > 2:
            logging.warning(
                "BYOL only supports two views. Only the first two views will be used."
            )

        criterion = torch.nn.CosineSimilarity(dim=1)
        return 0.5 * (
            criterion(projections[0], projections_target[1])
            + criterion(projections[1], projections_target[0])
        )


class BYOLConfig(SelfDistillationConfig):
    """Configuration for the BYOL model parameters."""

    pass
