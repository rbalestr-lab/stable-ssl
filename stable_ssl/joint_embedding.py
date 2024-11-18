# -*- coding: utf-8 -*-
"""Base class for joint embedding models."""
#
# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#         Randall Balestriero <randallbalestriero@gmail.com>
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import torch.nn.functional as F

from stable_ssl.utils import deactivate_requires_grad, update_momentum
from stable_ssl.base import BaseModel


class JointEmbedding(BaseModel):
    """Joint embedding version of Base Model.

    Parameters
    ----------
    cfg: dict
        the config
    """

    def forward(self):
        return self.networks["backbone_classifier"](
            self.networks["backbone"](self.batch[0])
        )

    def compute_loss(self):
        embeddings = [self.networks["backbone"](view) for view in self.batch[0]]
        loss_backbone_classifier = sum(
            [
                F.cross_entropy(
                    self.networks["backbone_classifier"](embed.detach()), self.batch[1]
                )
                for embed in embeddings
            ]
        )

        projections = [self.networks["projector"](embed) for embed in embeddings]
        loss_proj_classifier = sum(
            [
                F.cross_entropy(
                    self.networks["projector_classifier"](proj.detach()), self.batch[1]
                )
                for proj in projections
            ]
        )

        loss_ssl = self.compute_ssl_loss(*projections)

        if self.global_step % self.logger["every_step"] == 0:
            self.log(
                {
                    "train/loss_ssl": loss_ssl.item(),
                    "train/loss_backbone_classifier": loss_backbone_classifier.item(),
                    "train/loss_projector_classifier": loss_proj_classifier.item(),
                },
                commit=False,
            )

        return loss_ssl + loss_proj_classifier + loss_backbone_classifier


class SelfDistillation(JointEmbedding):
    r"""Base class for training a self-distillation SSL model.

    Parameters
    ----------
    config : GlobalConfig
        Parameters organized in groups.
        For details, see the `GlobalConfig` class in `config.py`.
    """

    def initialize_modules(self):
        super().initialize_modules()
        self.networks["backbone_target"] = copy.deepcopy(self.networks["backbone"])
        self.networks["projector_target"] = copy.deepcopy(self.networks["projector"])

        deactivate_requires_grad(self.networks["backbone_target"])
        deactivate_requires_grad(self.networks["projector_target"])

    def before_fit_step(self):
        """Update the target parameters as EMA of the online model parameters."""
        update_momentum(
            self.backbone, self.backbone_target, m=self.config.model.momentum
        )
        update_momentum(
            self.projector, self.projector_target, m=self.config.model.momentum
        )
