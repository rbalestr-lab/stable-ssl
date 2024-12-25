# -*- coding: utf-8 -*-
"""Template classes to easily instanciate Supervised or SSL trainers."""
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

from .base import BaseTrainer
from .utils import update_momentum, log_and_raise


class SupervisedTrainer(BaseTrainer):
    r"""Base class for training a supervised model."""

    def forward(self, x):
        return self.module["backbone"](x)

    def predict(self):
        return self.forward(self.batch[0])

    def compute_loss(self):
        loss = self.loss(self.predict(), self.batch[1])
        return {"loss": loss}


class JointEmbeddingTrainer(BaseTrainer):
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

    def forward(self, x):
        return self.module["backbone"](x)

    def predict(self):
        return self.module["backbone_classifier"](self.forward(self.batch[0]))

    def compute_loss(self):
        views, labels = self.format_views_labels()
        embeddings = [self.module["backbone"](view) for view in views]
        self.latest_forward = embeddings
        projections = [self.module["projector"](embed) for embed in embeddings]

        loss_ssl = self.loss(*projections)

        classifier_losses = self.compute_loss_classifiers(
            embeddings, projections, labels
        )

        return {"loss_ssl": loss_ssl, **classifier_losses}

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
            "loss_backbone_classifier": loss_backbone_classifier,
            "loss_projector_classifier": loss_projector_classifier,
        }


class SelfDistillationTrainer(JointEmbeddingTrainer):
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
            self.loss(projections[0], projections_target[1])
            + self.loss(projections[1], projections_target[0])
        )

        classifier_losses = self.compute_loss_classifiers(
            embeddings, projections, labels
        )

        return {"train/loss_ssl": loss_ssl, **classifier_losses}


class SimSiamTrainer(JointEmbeddingTrainer):
    r"""Base class for training a SimSiam SSL model."""

    def compute_loss(self):
        views, labels = self.format_views_labels()
        embeddings = [self.module["backbone"](view) for view in views]
        projections = [self.module["projector"](embed) for embed in embeddings]

        if len(projections) > 2:
            logging.warning("Only the first two views are used when using SimSiam.")

        if not hasattr(self.module, "predictor"):
            log_and_raise(ValueError, "SimSiam requires a predictor module.")

        predictions = [self.module["predictor"](proj) for proj in projections]
        detached_projections = [proj.detach() for proj in projections]

        loss_ssl = 0.5 * (
            self.loss(predictions[0], detached_projections[1])
            + self.loss(predictions[1], detached_projections[0])
        )

        classifier_losses = self.compute_loss_classifiers(
            embeddings, projections, labels
        )

        return {"train/loss_ssl": loss_ssl, **classifier_losses}
