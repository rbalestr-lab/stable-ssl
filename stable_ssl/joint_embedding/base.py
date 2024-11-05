# -*- coding: utf-8 -*-
"""Base class for joint embedding models."""
#
# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#         Randall Balestriero <randallbalestriero@gmail.com>
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
from abc import abstractmethod
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
from torch import nn

from stable_ssl.utils import load_nn, mlp, deactivate_requires_grad
from stable_ssl.base import BaseModel, ModelConfig


@dataclass
class JointEmbeddingConfig(ModelConfig):
    """Configuration for the joint-embedding model parameters.

    Parameters
    ----------
    projector : str
        Architecture of the projector head. Default is "2048-128".
    """

    projector: list[int] = field(default_factory=lambda: [2048, 128])

    def __post_init__(self):
        """Convert projector string to a list of integers if necessary."""
        if isinstance(self.projector, str):
            self.projector = [int(i) for i in self.projector.split("-")]


class JointEmbeddingModel(BaseModel):
    r"""Base class for training a joint-embedding SSL model.

    Parameters
    ----------
    config : TrainerConfig
        Parameters for Trainer organized in groups.
        For details, see the `TrainerConfig` class in `config.py`.
    """

    def initialize_modules(self):
        # backbone
        backbone, fan_in = load_nn(
            backbone_model=self.config.model.backbone_model,
            pretrained=False,
            dataset=self.config.data.train_dataset.name,
        )
        self.backbone = backbone.train()

        # projector
        sizes = [fan_in] + self.config.model.projector
        self.projector = mlp(sizes)

        # linear probes
        self.backbone_classifier = torch.nn.Linear(
            fan_in, self.config.data.train_dataset.num_classes
        )
        self.projector_classifier = torch.nn.Linear(
            self.config.model.projector[-1],
            self.config.data.train_dataset.num_classes,
        )

    def forward(self, x):
        return self.backbone(x)

    def compute_loss(self):
        embed_i = self.backbone(self.data[0][0])
        embed_j = self.backbone(self.data[0][1])

        loss_backbone = self._backbone_classifier_loss(embed_i, embed_j)

        z_i = self.projector(embed_i)
        z_j = self.projector(embed_j)

        loss_proj = self._projector_classifier_loss(z_i, z_j)
        loss_ssl = self._ssl_loss(z_i, z_j)

        self.log(
            {
                "train/loss_ssl": loss_ssl.item(),
                "train/loss_backbone_classifier": loss_backbone.item(),
                "train/loss_projector_classifier": loss_proj.item(),
            },
            commit=False,
        )

        return loss_ssl + loss_proj + loss_backbone

    @abstractmethod
    def ssl_loss(self, z_i, z_j):
        raise NotImplementedError

    def _backbone_classifier_loss(self, embed_i, embed_j):
        loss_backbone_i = F.cross_entropy(
            self.backbone_classifier(embed_i.detach()), self.data[1]
        )
        loss_backbone_j = F.cross_entropy(
            self.backbone_classifier(embed_j.detach()), self.data[1]
        )
        return loss_backbone_i + loss_backbone_j

    def _projector_classifier_loss(self, z_i, z_j):
        loss_proj_i = F.cross_entropy(
            self.projector_classifier(z_i.detach()), self.data[1]
        )
        loss_proj_j = F.cross_entropy(
            self.projector_classifier(z_j.detach()), self.data[1]
        )
        return loss_proj_i + loss_proj_j


class SelfDistillationModel(JointEmbeddingModel):
    def initialize_modules(self):
        super().initialize_modules()
        self.backbone_target = copy.deepcopy(self.backbone)
        self.projector_target = copy.deepcopy(self.projector)

        deactivate_requires_grad(self.backbone_target)
        deactivate_requires_grad(self.projector_target)
