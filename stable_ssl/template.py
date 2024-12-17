# -*- coding: utf-8 -*-
"""Template classes to easily instanciate SSL models."""
#
# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#         Randall Balestriero <randallbalestriero@gmail.com>
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging

import torch
import torch.nn.functional as F

from .base import BaseModel
from .utils import update_momentum


class JointEmbedding(BaseModel):
    r"""Base class for training a joint-embedding SSL model."""

    def format_views_labels(self):
        if (
            len(self.batch) == 2
            and torch.is_tensor(self.batch[1])
            and not torch.is_tensor(self.batch[0])
        ):
            # we assume the second element are the labels
            views, labels = self.batch
        elif (
            len(self.batch) > 1
            and all([torch.is_tensor(b) for b in self.batch])
            and len(set([b.ndim for b in self.batch])) == 1
        ):
            # we assume all elements are views
            views = self.batch
            labels = None
        else:
            msg = """You are using the JointEmbedding class with only 1 view!
            Make sure to double check your config and datasets definition.
            Most methods expect 2 views, some can use more."""
            log_and_raise(ValueError, msg)
        return views, labels

    def predict(self):
        return self.module["backbone_classifier"](self.forward())

    def compute_loss(self):
        views, labels = self.format_views_labels()
        embeddings = [self.module["backbone"](view) for view in views]
        projections = [self.module["projector"](embed) for embed in embeddings]

        if "predictor" in self.module:

            if len(projections_target) > 2:
                logging.warning(
                    "Only the first two views are used when there is a predictor."
                )

                predictions = [self.module["predictor"](proj) for proj in projections]
                detached_projections = [proj.detach() for proj in projections]

                loss_ssl = 0.5 * (
                    self.objective(predictions[0], detached_projections[1])
                    + self.objective(predictions[1], detached_projections[0])
                )
        else:
            loss_ssl = self.objective(*projections)

        classifier_losses = self.compute_loss_classifiers(
            embeddings, projections, labels
        )

        return {"train/loss_ssl": loss_ssl, **classifier_losses}

    def compute_loss_classifiers(self, embeddings, projections, labels):
        loss_backbone_classifier = 0
        loss_projector_classifier = 0

        if labels is not None:
            for embed, proj in zip(embeddings, projections):
                loss_backbone_classifier += F.cross_entropy(
                    self.module["backbone_classifier"](embed.detach()), labels
                )
                loss_projector_classifier += F.cross_entropy(
                    self.module["projector_classifier"](proj.detach()), labels
                )

        return {
            "train/loss_backbone_classifier": loss_backbone_classifier,
            "train/loss_projector_classifier": loss_projector_classifier,
        }


class SelfDistillation(JointEmbedding):
    r"""Base class for training a self-distillation SSL model."""

    def setup(self):
        logging.getLogger().setLevel(self._logger["level"])
        logging.info(f"=> SETUP OF {self.__class__.__name__} STARTED.")
        self._instanciate()
        self.module["backbone_target"] = copy.deepcopy(self.module["backbone"])
        self.module["projector_target"] = copy.deepcopy(self.module["projector"])

        self.module["backbone_target"].requires_grad_(False)
        self.module["projector_target"].requires_grad_(False)
        self._load_checkpoint()
        logging.info(f"=> SETUP OF {self.__class__.__name__} COMPLETED.")

    def before_fit_step(self):
        """Update the target parameters as EMA of the online model parameters."""
        update_momentum(
            self.backbone, self.backbone_target, m=self.config.model.momentum
        )
        update_momentum(
            self.projector, self.projector_target, m=self.config.model.momentum
        )

    def compute_loss(self):
        views, labels = self.format_views_labels()
        embeddings = [self.module["backbone"](view) for view in views]
        projections = [self.module["projector"](embed) for embed in embeddings]

        # If a predictor is used, it is generally applied to the online projections.
        if "predictor" in self.module:
            projections = [self.module["predictor"](proj) for proj in projections]

        projections_target = [
            self.module["projector_target"](self.module["backbone_target"](view))
            for view in views
        ]

        loss_ssl = 0.5 * (
            self.objective(projections[0], projections_target[1])
            + self.objective(projections[1], projections_target[0])
        )

        classifier_losses = self.compute_loss_classifiers(
            embeddings, projections, labels
        )

        return {"train/loss_ssl": loss_ssl, **classifier_losses}
