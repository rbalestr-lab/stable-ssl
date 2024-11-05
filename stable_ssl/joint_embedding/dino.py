# -*- coding: utf-8 -*-
"""DINO model."""
#
# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#         Randall Balestriero <randallbalestriero@gmail.com>
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
import torch
import torch.nn.functional as F

from .base import SelfDistillationConfig, SelfDistillationModel


class DINO(SelfDistillationModel):
    """DINO model from [CTM+21]_.

    Reference
    ---------
    .. [CTM+21] Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J.,
        Bojanowski, P., & Joulin, A. (2021).
        Emerging Properties in Self-Supervised Vision Transformers.
        International Conference on Computer Vision.
    """

    def ssl_loss(self, z_i, z_j):
        """Compute the loss of the DINO model.

        Parameters
        ----------
        z_i : torch.Tensor
            Latent representation of the first augmented view of the batch.
        z_j : torch.Tensor
            Latent representation of the second augmented view of the batch.

        Returns
        -------
        float
            The computed loss.
        """
