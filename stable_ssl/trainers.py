"""Template classes to easily instantiate Supervised or SSL trainers."""

#
# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#         Randall Balestriero <randallbalestriero@gmail.com>
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from .base import BaseTrainer
from .modules import TeacherStudentModule
from .utils import compute_global_mean, log_and_raise

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
        if self.loss is None:
            log_and_raise(
                ValueError,
                f"When using the trainer {self.__class__.__name__}, "
                "one needs to either provide a loss function in the config "
                "or implement a custom `compute_loss` method.",
            )
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
            and len({b.ndim for b in self.batch}) == 1
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
        if self.loss is None:
            log_and_raise(
                ValueError,
                f"When using the trainer {self.__class__.__name__}, "
                "one needs to either provide a loss function in the config "
                "or implement a custom `compute_loss` method.",
            )

        views, labels = self.format_views_labels()
        representations = [self.module["backbone"](view) for view in views]
        self._latest_representations = representations

        embeddings = [self.module["projector"](rep) for rep in representations]
        self._latest_embeddings = embeddings

        loss_ssl = self.loss(*embeddings)

        classifier_losses = self.compute_loss_classifiers(
            representations, embeddings, labels
        )

        return {"loss_ssl": loss_ssl, **classifier_losses}

    def compute_loss_classifiers(self, representations, embeddings, labels):
        """Compute the classifier loss for both backbone and projector."""
        loss_backbone_classifier = 0
        loss_projector_classifier = 0

        # Inputs are detached to avoid backprop through backbone and projector.
        if labels is not None:
            for rep, embed in zip(representations, embeddings):
                loss_backbone_classifier += F.cross_entropy(
                    self.module["backbone_classifier"](rep.detach()), labels
                )
                loss_projector_classifier += F.cross_entropy(
                    self.module["projector_classifier"](embed.detach()), labels
                )

        return {
            "loss_backbone_classifier": loss_backbone_classifier,
            "loss_projector_classifier": loss_projector_classifier,
        }

    @property
    def latest_embeddings(self):
        if not hasattr(self, "_latest_embeddings"):
            return None
        return self._latest_embeddings

    @latest_embeddings.setter
    def latest_embeddings(self, value):
        self._latest_embeddings = value

    @property
    def latest_representations(self):
        if not hasattr(self, "_latest_representations"):
            return None
        return self._latest_representations

    @latest_representations.setter
    def latest_representations(self, value):
        self._latest_representations = value


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
        if self.loss is None:
            log_and_raise(
                ValueError,
                f"When using the trainer {self.__class__.__name__}, "
                "one needs to either provide a loss function in the config "
                "or implement a custom `compute_loss` method.",
            )

        views, labels = self.format_views_labels()

        representations_student = [
            self.module["backbone"].forward_student(view) for view in views
        ]
        embeddings_student = [
            self.module["projector"].forward_student(rep)
            for rep in representations_student
        ]

        # If a predictor is used, it is applied to the student embeddings.
        if "predictor" in self.module:
            embeddings_student = [
                self.module["predictor"](embed) for embed in embeddings_student
            ]

        representations_teacher = [
            self.module["backbone"].forward_teacher(view) for view in views
        ]
        self.latest_representations = representations_teacher
        embeddings_teacher = [
            self.module["projector"].forward_teacher(rep)
            for rep in representations_teacher
        ]
        self.latest_embeddings = embeddings_teacher

        loss_ssl = 0.5 * (
            self.loss(embeddings_student[0], embeddings_teacher[1])
            + self.loss(embeddings_student[1], embeddings_teacher[0])
        )

        classifier_losses = self.compute_loss_classifiers(
            representations_teacher, embeddings_teacher, labels
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
        Default is 0.07.
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
        temperature_teacher: float = 0.07,
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

        representations_student = [
            self.module["backbone"].forward_student(view) for view in views
        ]
        embeddings_student = [
            self.module["projector"].forward_student(rep)
            for rep in representations_student
        ]

        # Construct target *from global views only* with the target ('teacher') network.
        with torch.no_grad():
            global_views = self.batch[0][:2]  # First two views should be global views.
            representations_teacher = [
                self.module["backbone"].forward_teacher(view) for view in global_views
            ]
            self.latest_representations = representations_teacher
            embeddings_teacher = [
                self.module["projector"].forward_teacher(rep)
                for rep in representations_teacher
            ]
            self.latest_embeddings = embeddings_teacher

        if self.epoch < self.warmup_epochs_temperature_teacher:
            temperature_teacher = self.temperature_teacher_schedule[self.epoch]
        else:
            temperature_teacher = self.temperature_teacher

        stacked_embeddings_teacher = torch.stack(embeddings_teacher)
        if hasattr(self, "center"):
            probs_teacher = F.softmax(
                (stacked_embeddings_teacher - self.center) / temperature_teacher,
                dim=-1,
            )
        else:
            probs_teacher = F.softmax(
                stacked_embeddings_teacher / temperature_teacher, dim=-1
            )

        stacked_embeddings_student = torch.stack(embeddings_student)
        log_probs_student = F.log_softmax(
            stacked_embeddings_student / self.temperature_student, dim=-1
        )

        # Compute the cross entropy loss between the student and teacher probabilities.
        probs_teacher_flat = probs_teacher.flatten(start_dim=1)
        log_probs_student_flat = log_probs_student.flatten(start_dim=1)
        loss_ssl = -probs_teacher_flat @ log_probs_student_flat.T
        loss_ssl.fill_diagonal_(0)

        # Normalize the loss.
        n_terms = loss_ssl.numel() - loss_ssl.diagonal().numel()
        batch_size = stacked_embeddings_teacher.shape[1]
        loss_ssl = loss_ssl.sum() / (n_terms * batch_size)

        # Update the center of the teacher network.
        with torch.no_grad():
            batch_center = compute_global_mean(stacked_embeddings_teacher, dim=(0, 1))
            if not hasattr(self, "center"):
                self.center = batch_center
            else:
                self.center = self.center * self.center_momentum + batch_center * (
                    1 - self.center_momentum
                )

        classifier_losses = self.compute_loss_classifiers(
            representations_teacher, embeddings_teacher, labels
        )

        return {"loss_ssl": loss_ssl, **classifier_losses}
