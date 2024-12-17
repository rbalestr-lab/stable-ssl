# -*- coding: utf-8 -*-
"""SSL losses."""
#
# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#         Randall Balestriero <randallbalestriero@gmail.com>
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn.functional as F

from stable_ssl.utils import gather, off_diagonal, all_reduce


class NTXEntLoss(torch.nn.Module):
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
        z_i = gather(z_i)
        z_j = gather(z_j)

        z = torch.cat([z_i, z_j], 0)
        N = z.size(0)

        features = F.normalize(z, dim=1)
        sim = torch.matmul(features, features.T) / self.temperature

        sim_i_j = torch.diag(sim, N // 2)
        sim_j_i = torch.diag(sim, -N // 2)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0)

        mask = torch.eye(N, dtype=bool).to(z_i.device)
        negative_samples = sim[~mask].reshape(N, -1)

        attraction = -positive_samples.mean()
        repulsion = torch.logsumexp(negative_samples, dim=1).mean()

        return attraction + repulsion


class NegativeCosineSimilarity(torch.nn.Module):
    """Negative cosine similarity objective.

    This objective is used for instance in BYOL [GSA+20]_ or SimSiam [CH21]_.

    Reference
    ---------
    .. [GSA+20] Grill, J. B., Strub, F., Altché, ... & Valko, M. (2020).
            Bootstrap Your Own Latent-A New Approach To Self-Supervised Learning.
            Advances in neural information processing systems, 33, 21271-21284.
    .. [CH21] Chen, X., & He, K. (2021).
            Exploring simple siamese representation learning.
            IEEE/CVF conference on Computer Vision and Pattern Recognition.
    """

    def forward(self, z_i, z_j):
        """Compute the loss of the BYOL model.

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
        sim = torch.nn.CosineSimilarity(dim=1)
        return -sim(z_i, z_j).mean()


class VICRegLoss(torch.nn.Module):
    """SSL objective used in VICReg [BPL21]_.

    Parameters
    ----------
    sim_coeff : float, optional
        The weight of the similarity loss (attractive term).
        Default is 25.
    std_coeff : float, optional
        The weight of the standard deviation loss.
        Default is 25.
    cov_coeff : float, optional
        The weight of the covariance loss.
        Default is 1.
    epsilon : float, optional
        Small value to avoid division by zero.
        Default is 1e-4.

    Reference
    ---------
    .. [BPL21] Bardes, A., Ponce, J., & LeCun, Y. (2021).
            VICReg: Variance-Invariance-Covariance Regularization
            For Self-Supervised Learning.
            International Conference on Learning Representations (ICLR).
    """

    def __init__(
        self,
        sim_coeff: float = 25,
        std_coeff: float = 25,
        cov_coeff: float = 1,
        epsilon: float = 1e-4,
    ):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.epsilon = epsilon

    def forward(self, z_i, z_j):
        """Compute the loss of the VICReg model.

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
        repr_loss = F.mse_loss(z_i, z_j)

        z_i = gather(z_i)
        z_j = gather(z_j)

        z_i = z_i - z_i.mean(dim=0)
        z_j = z_j - z_j.mean(dim=0)

        std_i = torch.sqrt(z_i.var(dim=0) + self.epsilon)
        std_j = torch.sqrt(z_j.var(dim=0) + self.epsilon)
        std_loss = torch.mean(F.relu(1 - std_i)) / 2 + torch.mean(F.relu(1 - std_j)) / 2

        cov_i = (z_i.T @ z_i) / (z_i.size(0) - 1)
        cov_j = (z_j.T @ z_j) / (z_i.size(0) - 1)
        cov_loss = off_diagonal(cov_i).pow_(2).sum().div(z_i.size(1)) + off_diagonal(
            cov_j
        ).pow_(2).sum().div(z_i.size(1))

        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )
        return loss


class BarlowTwinsLoss(torch.nn.Module):
    """SSL objective used in [ZJM+21]_.

    Parameters
    ----------
    lambd : float, optional
        The weight of the off-diagonal terms in the loss.
        Default is 5e-3.

    Reference
    ---------
    .. [ZJM+21] Zbontar, J., Jing, L., Misra, I., LeCun, Y., & Deny, S. (2021).
            Barlow Twins: Self-Supervised Learning via Redundancy Reduction.
            In International conference on machine learning (pp. 12310-12320). PMLR.
    """

    def __init__(self, lambd: float = 5e-3):
        super().__init__()
        self.lambd = lambd
        self.bn = torch.nn.LazyBatchNorm1d()

    def forward(self, z_i, z_j):
        """Compute the loss of the Barlow Twins model.

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
        c = self.bn(z_i).T @ self.bn(z_j)  # normalize along the batch dimension
        c = c / z_i.size(0)
        all_reduce(c)

        on_diag = (torch.diagonal(c) - 1).pow(2).sum()
        off_diag = off_diagonal(c).pow(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss
