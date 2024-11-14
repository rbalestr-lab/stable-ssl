# -*- coding: utf-8 -*-
"""Base class for joint embedding models."""
#
# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#         Randall Balestriero <randallbalestriero@gmail.com>
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import torch
import torch.nn.functional as F
from stable_ssl.utils import gather_processes

from stable_ssl.utils import deactivate_requires_grad, update_momentum
from stable_ssl.base import BaseModel


class JointEmbedding(BaseModel):
    """Joint embedding version of Base Model.

    Parameters
    ----------
    cfg: dict
        the config
    """

    def forward(self, x):
        return self.networks["backbone_classifier"](self.networks["backbone"](x))

    def compute_loss(self):
        embeddings = [self.networks["backbone"](view) for view in self.batch[0]]
        loss_backbone = self._compute_backbone_classifier_loss(*embeddings)

        projections = [self.networks["projector"](embed) for embed in embeddings]
        loss_proj = self._compute_projector_classifier_loss(*projections)
        loss_ssl = self.compute_ssl_loss(*projections)

        if self.global_step % self.logger["every_step"] == 0:
            self.log(
                {
                    "train/loss_ssl": loss_ssl.item(),
                    "train/loss_backbone_classifier": loss_backbone.item(),
                    "train/loss_projector_classifier": loss_proj.item(),
                },
                commit=False,
            )

        return loss_ssl + loss_proj + loss_backbone

    @gather_processes
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
        return self.objective(z_i, z_j)

    def _compute_backbone_classifier_loss(self, *embeddings):
        losses = [
            F.cross_entropy(
                self.networks["backbone_classifier"](embed.detach()), self.batch[1]
            )
            for embed in embeddings
        ]
        return sum(losses)

    def _compute_projector_classifier_loss(self, *projections):
        losses = [
            F.cross_entropy(
                self.networks["projector_classifier"](proj.detach()), self.batch[1]
            )
            for proj in projections
        ]
        return sum(losses)


class SelfDistillationModel(JointEmbedding):
    r"""Base class for training a self-distillation SSL model.

    Parameters
    ----------
    config : GlobalConfig
        Parameters organized in groups.
        For details, see the `GlobalConfig` class in `config.py`.
    """

    def initialize_modules(self):
        super().initialize_modules()
        self.backbone_target = copy.deepcopy(self.backbone)
        self.projector_target = copy.deepcopy(self.projector)

        deactivate_requires_grad(self.backbone_target)
        deactivate_requires_grad(self.projector_target)

    def compute_loss(self):
        embeddings = [self.backbone(view) for view in self.data[0]]
        loss_backbone = self._compute_backbone_classifier_loss(*embeddings)

        projections = [self.projector(embed) for embed in embeddings]
        loss_proj = self._compute_projector_classifier_loss(*projections)

        with torch.no_grad():
            projections_target = [
                self.projector_target(self.backbone_target(view))
                for view in self.data[0]
            ]
        loss_ssl = self.compute_ssl_loss(projections, projections_target)

        self.log(
            {
                "train/loss_ssl": loss_ssl.item(),
                "train/loss_backbone_classifier": loss_backbone.item(),
                "train/loss_projector_classifier": loss_proj.item(),
            },
            commit=False,
        )

        return loss_ssl + loss_proj + loss_backbone

    def before_train_step(self):
        # Update the target parameters as EMA of the online model parameters.
        update_momentum(
            self.backbone, self.backbone_target, m=self.config.model.momentum
        )
        update_momentum(
            self.projector, self.projector_target, m=self.config.model.momentum
        )


# @dataclass
# class SelfDistillationConfig(JointEmbeddingConfig):
#     """Configuration for the self-distillation model parameters.

#     Parameters
#     ----------
#     momentum : float
#         Momentum for the EMA of the target parameters. Default is 0.999.
#     """

#     momentum: float = 0.999
