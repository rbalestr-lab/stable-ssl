# -*- coding: utf-8 -*-
"""Template classes to easily instanciate Supervised or SSL trainers."""
#
# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#         Randall Balestriero <randallbalestriero@gmail.com>
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from .base import BaseTrainer
from .utils import log_and_raise
from .modules import TeacherModule


class SupervisedTrainer(BaseTrainer):
    r"""Base class for training a supervised model."""

    def check_module(self):
        """Check if a 'backbone' module is defined."""
        if "backbone" not in self.module:
            log_and_raise(
                ValueError,
                "A `backbone` module needs to be defined when "
                f"using the {self.__class__.__name__} trainer.",
            )

    def forward(self, x):
        """Forward pass. By default, it simply calls the 'backbone' module."""
        return self.module["backbone"](x)

    def predict(self):
        """Call the forward pass of current batch."""
        return self.forward(self.batch[0])

    def compute_loss(self):
        """Compute the loss of the model using the `loss` provided in the config."""
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

    def check_module(self):
        """Check if 'backbone', 'projector' and their classifiers are defined."""
        required_modules = [
            "backbone",
            "projector",
            "backbone_classifier",
            "projector_classifier",
        ]

        for module_name in required_modules:
            if module_name not in self.module:
                log_and_raise(
                    ValueError,
                    f"A `{module_name}` module needs to be defined when "
                    f"using the {self.__class__.__name__} trainer.",
                )

    def forward(self, x):
        """Forward pass. By default, it simply calls the 'backbone' module."""
        return self.module["backbone"](x)

    def predict(self):
        """Call the backbone classifier on the forward pass of current batch."""
        return self.module["backbone_classifier"](self.forward(self.batch[0]))

    def compute_loss(self):
        """Compute final loss as sum of SSL loss and classifier losses."""
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
        """Compute the classifier loss for both backbone and projector."""
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

    def check_module(self):
        """Check if 'backbone', 'projector', classifiers and teachers are defined."""
        super().check_module()  # check 'backbone', 'projector' and classifiers

        # Find the TeacherModule that wraps the 'projector' and 'backbone' modules.
        for name, mod in self.module.items():
            if isinstance(mod, TeacherModule):
                if mod.student is self.module["projector"]:
                    self.projector_teacher = mod
                elif mod.student is self.module["backbone"]:
                    self.backbone_teacher = mod

        if not hasattr(self, "projector_teacher") or not hasattr(
            self, "backbone_teacher"
        ):
            log_and_raise(
                ValueError,
                "A `TeacherModule` needs to be defined for both "
                "the 'backbone' and 'projector' modules.",
            )

    def compute_loss(self):
        """Compute final loss as sum of SSL loss and classifier losses."""
        views, labels = self.format_views_labels()
        embeddings = [self.module["backbone"](view) for view in views]
        projections = [self.module["projector"](embed) for embed in embeddings]

        # If a predictor is used, it is applied to the student projections.
        if "predictor" in self.module:
            projections = [self.module["predictor"](proj) for proj in projections]

        projections_teacher = [
            self.projector_teacher(self.backbone_teacher(view)) for view in views
        ]

        loss_ssl = 0.5 * (
            self.loss(projections[0], projections_teacher[1])
            + self.loss(projections[1], projections_teacher[0])
        )

        classifier_losses = self.compute_loss_classifiers(
            embeddings, projections, labels
        )

        return {"loss_ssl": loss_ssl, **classifier_losses}


# @torch.no_grad()
# def center_mean(x: Tensor, dim: Tuple[int, ...]) -> Tensor:
#     """Returns the center of the input tensor by calculating the mean."""
#     batch_center = torch.mean(x, dim=dim, keepdim=True)
#     if dist.is_available() and dist.is_initialized():
#         dist.all_reduce(batch_center)
#         batch_center = batch_center / dist.get_world_size()
#     return batch_center
