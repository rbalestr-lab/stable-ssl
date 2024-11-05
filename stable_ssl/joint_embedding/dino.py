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

from .base import JointEmbeddingConfig, JointEmbeddingModel


class DINO(JointEmbeddingModel):
    """DINO model from [CTM+21]_.

    Reference
    ---------
    .. [CTM+21] Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J.,
        Bojanowski, P., & Joulin, A. (2021).
        Emerging Properties in Self-Supervised Vision Transformers.
        International Conference on Computer Vision.
    """

    def compute_ssl_loss(self, z_i, z_j):
        """Compute the contrastive loss for SimCLR.

        Parameters
        ----------
        z_i : torch.Tensor
            Latent representation of the first augmented view of the batch.
        z_j : torch.Tensor
            Latent representation of the second augmented view of the batch.

        Returns
        -------
        float
            The computed contrastive loss.
        """
        z = torch.cat([z_i, z_j], 0)
        N = z.size(0)

        features = F.normalize(z, dim=1)
        sim = torch.matmul(features, features.T) / self.config.model.temperature

        sim_i_j = torch.diag(sim, N // 2)
        sim_j_i = torch.diag(sim, -N // 2)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0)  # shape (N)

        mask = torch.eye(N, dtype=bool).to(self.this_device)
        negative_samples = sim[~mask].reshape(N, -1)  # shape (N, N-1)

        attraction = -positive_samples.mean()
        repulsion = torch.logsumexp(negative_samples, dim=1).mean()

        return attraction + repulsion
