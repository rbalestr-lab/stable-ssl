# -*- coding: utf-8 -*-
"""SimCLR model."""
#
# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#         Randall Balestriero <randallbalestriero@gmail.com>
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F


class NTXEnt(torch.nn.Module):
    """Normalized temperature-scaled cross entropy loss.

    Introduced in the SimCLR paper [CKNH20]_. Also used in MoCo [HFW+20]_.

    Reference
    ---------
    .. [CKNH20] Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020).
            A Simple Framework for Contrastive Learning of Visual Representations.
            In International Conference on Machine Learning (pp. 1597-1607). PMLR.
    .. [HFW+20] He, K., Fan, H., Wu, Y., Xie, S., & Girshick, R. (2020).
            Momentum Contrast for Unsupervised Visual Representation Learning.
            IEEE/CVF Conference on Computer Vision and Pattern Recognition.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """Compute the NT-Xent loss.

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
        sim = torch.matmul(features, features.T) / self.temperature

        sim_i_j = torch.diag(sim, N // 2)
        sim_j_i = torch.diag(sim, -N // 2)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0)  # shape (N)

        mask = torch.eye(N, dtype=bool).to(z_i.device)
        negative_samples = sim[~mask].reshape(N, -1)  # shape (N, N-1)

        attraction = -positive_samples.mean()
        repulsion = torch.logsumexp(negative_samples, dim=1).mean()

        return attraction + repulsion


class BYOLLoss(torch.nn.Module):
    """SSL objective used in BYOL [GSA+20]_.

    Reference
    ---------
    .. [GSA+20] Grill, J. B., Strub, F., Altché, ... & Valko, M. (2020).
            Bootstrap Your Own Latent-A New Approach To Self-Supervised Learning.
            Advances in neural information processing systems, 33, 21271-21284.
    """

    def forward(self, predictions, projections_target):
        """Compute the loss of the BYOL model.

        Parameters
        ----------
        predictions : list of torch.Tensor
            Predictions of the different augmented views from the online network.
        projections_target : list of torch.Tensor
            Projections of the corresponding augmented views from the target network.

        Returns
        -------
        float
            The computed loss.
        """
        if len(predictions) > 2 or len(projections_target) > 2:
            logging.warning(
                "BYOL only supports two views. Only the first two views will be used."
            )

        sim = torch.nn.CosineSimilarity(dim=1)
        return -0.5 * (
            sim(predictions[0], projections_target[1]).mean()
            + sim(predictions[1], projections_target[0]).mean()
        )
