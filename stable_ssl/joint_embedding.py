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
    r"""Base class for training a joint-embedding SSL model."""

    def predict(self):
        return self.network["backbone_classifier"](self.forward())

    def compute_loss(self):
        embeddings = [self.network["backbone"](view) for view in self.batch[0]]
        loss_backbone_classifier = sum(
            [
                F.cross_entropy(
                    self.network["backbone_classifier"](embed.detach()), self.batch[1]
                )
                for embed in embeddings
            ]
        )

        projections = [self.network["projector"](embed) for embed in embeddings]
        loss_proj_classifier = sum(
            [
                F.cross_entropy(
                    self.network["projector_classifier"](proj.detach()), self.batch[1]
                )
                for proj in projections
            ]
        )

        loss_ssl = self.objective(*projections)

        return {
            "train/loss_ssl": loss_ssl,
            "train/loss_backbone_classifier": loss_backbone_classifier,
            "train/loss_projector_classifier": loss_proj_classifier,
        }


class SelfDistillation(JointEmbedding):
    r"""Base class for training a self-distillation SSL model."""

    def initialize_modules(self):
        super().initialize_modules()
        self.network["backbone_target"] = copy.deepcopy(self.network["backbone"])
        self.network["projector_target"] = copy.deepcopy(self.network["projector"])

        deactivate_requires_grad(self.network["backbone_target"])
        deactivate_requires_grad(self.network["projector_target"])

    def before_fit_step(self):
        """Update the target parameters as EMA of the online model parameters."""
        update_momentum(
            self.backbone, self.backbone_target, m=self.config.model.momentum
        )
        update_momentum(
            self.projector, self.projector_target, m=self.config.model.momentum
        )
