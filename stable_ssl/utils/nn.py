import torchvision
import torch.nn as nn


def load_nn(
    name, n_classes, with_classifier=True, pretrained=False, dataset="CIFAR10", **kwargs
):
    if name == "resnet9":
        model = resnet9(**kwargs)
    elif name == "ConvMixer":
        model = ConvMixer(**kwargs)

    else:
        try:
            model = torchvision.models.__dict__[name](pretrained=pretrained, **kwargs)
        except KeyError:
            raise ValueError(f"Unknown model {name}.")

    if hasattr(model, "fc"):  # For models like ResNet
        in_features = model.fc.in_features
        model.fc = last_layer(n_classes, with_classifier, in_features)
    elif hasattr(model, "classifier"):  # For models like VGG, AlexNet
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = last_layer(n_classes, with_classifier, in_features)
    elif hasattr(model, "heads"):  # For models like ViT
        in_features = model.heads.head.in_features
        model.heads.head = last_layer(n_classes, with_classifier, in_features)
    elif hasattr(model, "head"):  # For models like Swin Transformer
        in_features = model.head.in_features
        model.head = last_layer(n_classes, with_classifier, in_features)
    else:
        raise ValueError(f"Unknown model structure for {name}.")

    model = adapt_resolution(model, dataset=dataset, backbone_model=name)

    return model, in_features


def adapt_resolution(model, dataset, backbone_model):
    if (
        "CIFAR" in dataset
        and "resnet" in backbone_model
        and backbone_model != "resnet9"
    ):
        model.conv1 = nn.Conv2d(
            3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        model.maxpool = nn.Identity()
    return model


def last_layer(n_classes, with_classifier, in_features):
    if with_classifier:
        return nn.Linear(in_features, n_classes)
    else:
        return nn.Identity()


class resnet9(nn.Module):
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
    # https://openreview.net/forum?id=TVHS5Y4dNvM
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
