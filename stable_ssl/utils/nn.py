# -*- coding: utf-8 -*-
"""Neural network models."""
#
# Author: Randall Balestriero <randallbalestriero@gmail.com>
#         Hugues Van Assel <vanasselhugues@gmail.com>
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torchvision
import torch.nn as nn


def load_nn(
    backbone_model,
    n_classes=None,
    pretrained=False,
    dataset="CIFAR10",
    **kwargs,
):
    """Load a neural network model with a given backbone.

    Parameters
    ----------
    backbone_model : str
        Name of the backbone model.
    n_classes : int
        Number of classes in the dataset.
        If None, the model is loaded without the classifier.
        Default is None.
    pretrained : bool, optional
        Whether to load a pretrained model, by default False.
    dataset : str, optional
        Name of the dataset, by default "CIFAR10".
    **kwargs: dict
        Additional keyword arguments for the model.

    Returns
    -------
    torch.nn.Module
        The neural network model.
    int
        The number of features in the last layer.
    """
    # Load the backbone_model.
    if backbone_model == "resnet9":
        model = resnet9(**kwargs)
    elif backbone_model == "ConvMixer":
        model = ConvMixer(**kwargs)
    else:
        try:
            model = torchvision.models.__dict__[backbone_model](
                pretrained=pretrained, **kwargs
            )
        except KeyError:
            raise ValueError(f"Unknown model: {backbone_model}.")

    # Adapt the last layer, either linear or identity.
    def last_layer(n_classes, in_features):
        if n_classes is not None:
            return nn.Linear(in_features, n_classes)
        else:
            return nn.Identity()

    # For models like ResNet.
    if hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = last_layer(n_classes, in_features)
    # For models like VGG or AlexNet.
    elif hasattr(model, "classifier"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = last_layer(n_classes, in_features)
    # For models like ViT.
    elif hasattr(model, "heads"):
        in_features = model.heads.head.in_features
        model.heads.head = last_layer(n_classes, in_features)
    # For models like Swin Transformer.
    elif hasattr(model, "head"):
        in_features = model.head.in_features
        model.head = last_layer(n_classes, in_features)
    else:
        raise ValueError(f"Unknown model structure for : '{backbone_model}'.")

    # TODO: enhance flexibility, this is too hardcoded.
    # Adapt the resolution of the model if using CIFAR with resnet.
    if (
        "CIFAR" in dataset
        and "resnet" in backbone_model
        and backbone_model != "resnet9"
    ):
        model.conv1 = nn.Conv2d(
            3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        model.maxpool = nn.Identity()

    return model, in_features


class resnet9(nn.Module):
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
    """ConvMixer model from [TK22]_.

    References
    ----------
    .. [TK22]  Trockman, A., & Kolter, J. Z. (2022).
            Patches are all you need?. arXiv preprint arXiv:2201.09792.
    """

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
        out = self.conv1(xb)
        for a, b in zip(self.blocks_a, self.blocks_b):
            out = out + a(out)
            out = b(out)
        out = self.fc(self.pool(out))
        return out
