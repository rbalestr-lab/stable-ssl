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
from .utils import log_and_raise, compute_global_mean
from .modules import TeacherStudentModule


# ==========================================
# Base trainers that require a loss function
# ==========================================


class SupervisedTrainer(BaseTrainer):
    r"""Base class for training a supervised model."""

    required_modules = {"backbone": torch.nn.Module}

    def forward(self, *args, **kwargs):
        """Forward pass. By default, it simply calls the 'backbone' module."""
        return self.module["backbone"](*args, **kwargs)

    def predict(self):
        """Call the forward pass of current batch."""
        return self.forward(self.batch[0])

    def compute_loss(self):
        """Compute the loss of the model using the `loss` provided in the config."""
        loss = self.loss(self.predict(), self.batch[1])
        return {"loss": loss}


class JointEmbeddingTrainer(BaseTrainer):
    r"""Base class for training a joint-embedding SSL model."""

    required_modules = {
        "backbone": torch.nn.Module,
        "projector": torch.nn.Module,
        "backbone_classifier": torch.nn.Module,
        "projector_classifier": torch.nn.Module,
    }

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

    def forward(self, *args, **kwargs):
        """Forward pass. By default, it simply calls the 'backbone' module."""
        return self.module["backbone"](*args, **kwargs)

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

    required_modules = {
        "backbone": TeacherStudentModule,
        "projector": TeacherStudentModule,
        "backbone_classifier": torch.nn.Module,
        "projector_classifier": torch.nn.Module,
    }

    def compute_loss(self):
        """Compute final loss as sum of SSL loss and classifier losses."""
        views, labels = self.format_views_labels()

        embeddings_student = [
            self.module["backbone"].forward_student(view) for view in views
        ]
        projections_student = [
            self.module["projector"].forward_student(embed)
            for embed in embeddings_student
        ]

        # If a predictor is used, it is applied to the student projections.
        if "predictor" in self.module:
            projections_student = [
                self.module["predictor"](proj) for proj in projections_student
            ]

        embeddings_teacher = [
            self.module["backbone"].forward_teacher(view) for view in views
        ]
        self.latest_forward = embeddings_teacher
        projections_teacher = [
            self.module["projector"].forward_teacher(embed)
            for embed in embeddings_teacher
        ]

        loss_ssl = 0.5 * (
            self.loss(projections_student[0], projections_teacher[1])
            + self.loss(projections_student[1], projections_teacher[0])
        )

        classifier_losses = self.compute_loss_classifiers(
            embeddings_teacher, projections_teacher, labels
        )

        return {"loss_ssl": loss_ssl, **classifier_losses}


# ===============================
# Trainers with Specific Losses
# ===============================


class DINOTrainer(SelfDistillationTrainer):
    r"""DINO SSL model by :cite:`caron2021emerging`.

    Parameters
    ----------
    warmup_temperature_teacher : float, optional
        The initial temperature for the teacher output.
        Default is 0.04.
    temperature_teacher : float, optional
        The temperature for the teacher output.
        Default is 0.04.
    warmup_epochs_temperature_teacher : int, optional
        The number of epochs to warm up the teacher temperature.
        Default is 30.
    temperature_student : float, optional
        The temperature for the student output.
        Default is 0.1.
    center_momentum : float, optional
        The momentum used to update the center.
        Default is 0.9.
    **kwargs
        Additional arguments passed to the base class.
    """

    def __init__(
        self,
        warmup_temperature_teacher: float = 0.04,
        temperature_teacher: float = 0.04,
        warmup_epochs_temperature_teacher: int = 30,
        temperature_student: float = 0.1,
        center_momentum: float = 0.9,
        **kwargs,
    ):
        super().__init__(
            warmup_temperature_teacher=warmup_temperature_teacher,
            temperature_teacher=temperature_teacher,
            warmup_epochs_temperature_teacher=warmup_epochs_temperature_teacher,
            temperature_student=temperature_student,
            center_momentum=center_momentum,
            **kwargs,
        )

        self.temperature_teacher_schedule = torch.linspace(
            start=warmup_temperature_teacher,
            end=temperature_teacher,
            steps=warmup_epochs_temperature_teacher,
        )

    def compute_loss(self):
        """Compute the DINO loss."""
        views, labels = self.format_views_labels()

        embeddings_student = [
            self.module["backbone"].forward_student(view) for view in views
        ]
        projections_student = [
            self.module["projector"].forward_student(embed)
            for embed in embeddings_student
        ]

        # Construct target *from global views only* with the target ('teacher') network.
        with torch.no_grad():
            global_views = self.batch[0][:2]  # First two views should be global views.
            embeddings_teacher = [
                self.module["backbone"].forward_teacher(view) for view in global_views
            ]
            self.latest_forward = embeddings_teacher
            projections_teacher = [
                self.module["projector"].forward_teacher(embed)
                for embed in embeddings_teacher
            ]

        if self.epoch < self.warmup_epochs_temperature_teacher:
            temperature_teacher = self.temperature_teacher_schedule[self.epoch]
        else:
            temperature_teacher = self.temperature_teacher

        stacked_projections_teacher = torch.stack(projections_teacher)
        if hasattr(self, "center"):
            probs_teacher = F.softmax(
                (stacked_projections_teacher - self.center) / temperature_teacher,
                dim=-1,
            )
        else:
            probs_teacher = F.softmax(
                stacked_projections_teacher / temperature_teacher, dim=-1
            )

        stacked_projections_student = torch.stack(projections_student)
        log_probs_student = F.log_softmax(
            stacked_projections_student / self.temperature_student, dim=-1
        )

        # Compute the cross entropy loss between the student and teacher probabilities.
        probs_teacher_flat = probs_teacher.flatten(start_dim=1)
        log_probs_student_flat = log_probs_student.flatten(start_dim=1)
        loss_ssl = probs_teacher_flat @ log_probs_student_flat.T
        loss_ssl.fill_diagonal_(0)

        # Normalize the loss.
        N = loss_ssl.size(0)
        n_terms = N * (N - 1)
        batch_size = stacked_projections_teacher.shape[1]
        loss_ssl = loss_ssl.sum() / (n_terms * batch_size)

        # Update the center of the teacher network.
        with torch.no_grad():
            batch_center = compute_global_mean(stacked_projections_teacher, dim=(0, 1))
            if not hasattr(self, "center"):
                self.center = batch_center
            else:
                self.center = self.center * self.center_momentum + batch_center * (
                    1 - self.center_momentum
                )

        classifier_losses = self.compute_loss_classifiers(
            embeddings_teacher, projections_teacher, labels
        )

        return {"loss_ssl": loss_ssl, **classifier_losses}
