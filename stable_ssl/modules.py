# -*- coding: utf-8 -*-
"""Neural network modules."""
#
# Author: Randall Balestriero <randallbalestriero@gmail.com>
#         Hugues Van Assel <vanasselhugues@gmail.com>
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torchvision
import torch
import torch.nn as nn

import logging
import copy

from .utils import log_and_raise


def load_backbone(
    name,
    num_classes,
    weights=None,
    low_resolution=False,
    return_feature_dim=False,
    **kwargs,
):
    """Load a backbone model.

    If num_classes is provided, the last layer is replaced by a linear layer of
    output size num_classes. Otherwise, the last layer is replaced by an identity layer.

    Parameters
    ----------
    name : str
        Name of the backbone model. Supported models are:
        - Any model from torchvision.models
        - "Resnet9"
        - "ConvMixer"
    num_classes : int
        Number of classes in the dataset.
        If None, the model is loaded without the classifier.
    weights : bool, optional
        Whether to load a weights model, by default False.
    low_resolution : bool, optional
        Whether to adapt the resolution of the model (for CIFAR typically).
        By default False.
    return_feature_dim : bool, optional
        Whether to return the feature dimension of the model.
    **kwargs: dict
        Additional keyword arguments for the model.

    Returns
    -------
    torch.nn.Module
        The neural network model.
    """
    # Load the name.
    if name == "resnet9":
        model = Resnet9(**kwargs)
    elif name == "ConvMixer":
        model = ConvMixer(**kwargs)
    else:
        try:
            model = torchvision.models.__dict__[name](weights=weights, **kwargs)
        except KeyError:
            raise ValueError(f"Unknown model: {name}.")

    # Adapt the last layer, either linear or identity.
    def last_layer(num_classes, in_features):
        if num_classes is not None:
            return nn.Linear(in_features, num_classes)
        else:
            return nn.Identity()

    # For models like ResNet.
    if hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = last_layer(num_classes, in_features)
    # For models like VGG or AlexNet.
    elif hasattr(model, "classifier"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = last_layer(num_classes, in_features)
    # For models like ViT.
    elif hasattr(model, "heads"):
        in_features = model.heads.head.in_features
        model.heads.head = last_layer(num_classes, in_features)
    # For models like Swin Transformer.
    elif hasattr(model, "head"):
        in_features = model.head.in_features
        model.head = last_layer(num_classes, in_features)
    else:
        raise ValueError(f"Unknown model structure for : '{name}'.")

    if low_resolution:  # reduce resolution, for instance for CIFAR
        if hasattr(model, "conv1"):
            model.conv1 = nn.Conv2d(
                3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            )
            model.maxpool = nn.Identity()
        else:
            logging.warning(f"Cannot adapt resolution for model: {name}.")

    if return_feature_dim:
        return model, in_features
    else:
        return model


class TeacherModule(nn.Module):
    """Teacher network updated as an EMA of the student network.

    The teacher model is updated by taking a running average of the student’s
    parameters and buffers. When `ema_coefficient == 0.0`, the teacher and student
    are literally the same object, saving memory but forward passes through the teacher
    will not produce any gradients.

    Parameters
    ----------
    student : str
        The name of the student model to be used as the teacher. It is only a str to
        indicate the student model to be used. The actual student model is set using
        the `set_student` method.
    warm_init : bool, optional
        If True, performs an initialization step to match the student’s parameters
        immediately. Default is True.
    ema_coefficient : float, optional
        The EMA decay factor in [0, 1]. A value of 0.0 means the teacher is fully
        updated to the student’s parameters on every step, while a value of 1.0 means
        the teacher remains unchanged. Default is 0.99.

    Raises
    ------
    ValueError
        If `ema_coefficient` is not in [0.0, 1.0].
    """

    def __init__(
        self,
        student: str = None,
        warm_init: bool = True,
        ema_coefficient: float = 0.99,
    ):
        if not (0.0 <= ema_coefficient <= 1.0):
            log_and_raise(
                ValueError,
                f"ema_coefficient must be in [0, 1]. Found: {ema_coefficient}.",
            )

        super().__init__()
        self._student = student
        self.warm_init = warm_init
        self.ema_coefficient = torch.Tensor([ema_coefficient])[0]

    def set_student(self, student: nn.Module):
        self.student = student

        if self.ema_coefficient.item() == 0.0:
            # No need to create a teacher network if the EMA coefficient is 0.0.
            self.teacher = student
        else:
            # Create a teacher network with the same architecture as the student.
            self.teacher = copy.deepcopy(student)
            self.teacher.requires_grad_(False)

            if self.warm_init:  # Initialization step to match the student’s parameters.
                original_ema_coefficient = self.ema_coefficient.item()
                self.ema_coefficient = torch.zeros(())
                self.step()
                self.ema_coefficient = torch.Tensor([original_ema_coefficient])[0]

    @torch.no_grad
    def step(self):
        """Perform one EMA update step on the teacher’s parameters.

        The update rule is:
            teacher_param = ema_coefficient * teacher_param
            + (1 - ema_coefficient) * student_param

        This is done in a `no_grad` context to ensure the teacher’s parameters do
        not accumulate gradients, but the student remains fully trainable.

        Everything is updated, including buffers (e.g. batch norm running averages).
        """
        if (
            self.ema_coefficient.item() == 0.0
        ):  # Nothing to update when the teacher is the student.
            return

        for teacher_group, student_group in [
            (self.teacher.parameters(), self.student.parameters()),
            (self.teacher.buffers(), self.student.buffers()),
        ]:
            for t, s in zip(teacher_group, student_group):
                # c = self.ema_coefficient.to(dtype=t.dtype)
                # t.mul_(c).add_((1.0 - c) * s)
                ty = t.dtype
                t.mul_(self.ema_coefficient.to(dtype=ty))
                t.add_((1.0 - self.ema_coefficient).to(dtype=ty) * s)

    def forward(self, *args, **kwargs):
        """Forward pass through the teacher.

        If ema_coefficient == 0.0, we wrap the forward in torch.no_grad() so that
        no gradients flow, even though teacher == student in memory.
        """
        if self.ema_coefficient.item() == 0.0:
            with torch.no_grad():
                return self.teacher(*args, **kwargs)
        else:
            return self.teacher(*args, **kwargs)


class MLP(nn.Module):
    """Multi-layer perceptron."""

    def __init__(self, sizes, activation="ReLU", batch_norm=True):
        super().__init__()
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=not batch_norm))
            if batch_norm:
                layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.__dict__[activation]())
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.layers = nn.Sequential(layers)

    def forward(self, x):
        """Forward pass."""
        return self.layers(x)


class Resnet9(nn.Module):
    """ResNet-9 model."""

    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()

        self.conv1 = self.conv_block(in_channels, 64)
        self.conv2 = self.conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(self.conv_block(128, 128), self.conv_block(128, 128))

        self.conv3 = self.conv_block(128, 256, pool=True)
        self.conv4 = self.conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(self.conv_block(512, 512), self.conv_block(512, 512))
        self.conv5 = self.conv_block(512, 1028, pool=True)
        self.res3 = nn.Sequential(
            self.conv_block(1028, 1028), self.conv_block(1028, 1028)
        )
        self.pool = nn.Sequential(nn.AdaptiveMaxPool2d(1), nn.Flatten())

        self.fc = nn.Linear(1028, num_classes)

    @staticmethod
    def conv_block(in_channels, out_channels, pool=False):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)

    def forward(self, xb):
        """Forward pass."""
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.conv5(out)
        out = self.res3(out) + out
        out = self.fc(self.pool(out))
        return out


class ConvMixer(nn.Module):
    """ConvMixer model from :cite:`trockman2022patches`."""

    def __init__(
        self,
        in_channels=3,
        num_classes=10,
        dim=64,
        depth=6,
        kernel_size=9,
        patch_size=7,
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
        )

        self.blocks_a = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.BatchNorm2d(dim),
                    nn.ReLU(),
                )
                for _ in range(depth)
            ]
        )
        self.blocks_b = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=1), nn.BatchNorm2d(dim), nn.ReLU()
                )
                for _ in range(depth)
            ]
        )

        self.pool = nn.Sequential(nn.AdaptiveMaxPool2d(1), nn.Flatten())
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, xb):
        """Forward pass."""
        out = self.conv1(xb)
        for a, b in zip(self.blocks_a, self.blocks_b):
            out = out + a(out)
            out = b(out)
        out = self.fc(self.pool(out))
        return out
