from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics.classification import MulticlassAccuracy

from ...utils import load_nn
from ..base import BaseModel, BaseModelConfig


@dataclass
class SSLConfig(BaseModelConfig):
    """
    Configuration for the SSL model parameters.

    Parameters:
    -----------
    projector : str
        Architecture of the projector head. Default is "2048-128".
    """

    projector: list[int] = field(default_factory=lambda: [2048, 128])

    def __post_init__(self):
        if isinstance(self.projector, str):
            self.projector = [int(i) for i in self.projector.split("-")]


class SSLTrainer(BaseModel):
    r"""Base class for training a Self-Supervised Learning (SSL) model.

    Parameters:
    -----------
    config : TrainerConfig
        Parameters for Trainer organized in groups.
        For details, see the `TrainerConfig` class in `config.py`.
    """

    def initialize_modules(self):
        # backbone
        model, fan_in = load_nn(
            name=self.config.model.backbone_model,
            n_classes=self.config.data.datasets[self.config.data.train_on].num_classes,
            with_classifier=False,
            pretrained=False,
            dataset=self.config.data.datasets[self.config.data.train_on].name,
        )
        self.backbone = model.train()

        # projector
        sizes = [fan_in] + self.config.model.projector
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # linear probes
        self.backbone_classifier = torch.nn.Linear(
            fan_in, self.config.data.datasets[self.config.data.train_on].num_classes
        )
        self.projector_classifier = torch.nn.Linear(
            self.config.model.projector[-1],
            self.config.data.datasets[self.config.data.train_on].num_classes,
        )

    def forward(self, x):
        return self.backbone_classifier(self.backbone(x))

    def compute_loss(self):
        embed_i = self.backbone(self.data[0][0])
        embed_j = self.backbone(self.data[0][1])

        h_i = self.projector(embed_i)
        h_j = self.projector(embed_j)
        loss_ssl = self.compute_ssl_loss(h_i, h_j)
        loss_backbone = F.cross_entropy(
            self.backbone_classifier(embed_i.detach()), self.data[1]
        )
        loss_backbone += F.cross_entropy(
            self.backbone_classifier(embed_j.detach()), self.data[1]
        )
        loss_proj = F.cross_entropy(
            self.projector_classifier(h_i.detach()), self.data[1]
        )
        loss_proj += F.cross_entropy(
            self.projector_classifier(h_j.detach()), self.data[1]
        )

        self.log(
            {
                "train/loss_ssl": loss_ssl,
                "train/loss_backbone_classifier": loss_backbone,
                "train/loss_projector_classifier": loss_proj,
            },
            commit=False,
        )
        return loss_ssl + loss_proj + loss_backbone

    def compute_ssl_loss(self, embeds):
        raise NotImplementedError

    def initialize_metrics(self):

        nc = self.config.data.datasets[self.config.data.train_on].num_classes
        train_acc1 = MulticlassAccuracy(num_classes=nc, top_k=1)
        acc1 = MulticlassAccuracy(num_classes=nc, top_k=1)
        acc5 = MulticlassAccuracy(num_classes=nc, top_k=5)
        acc1_by_class = MulticlassAccuracy(num_classes=nc, average="none", top_k=1)
        acc5_by_class = MulticlassAccuracy(num_classes=nc, average="none", top_k=5)
        self.metrics = torch.nn.ModuleDict(
            {
                "train/step/acc1": train_acc1,
                "eval/epoch/acc1": acc1,
                "eval/epoch/acc5": acc5,
                "eval/epoch/acc1_by_class": acc1_by_class,
                "eval/epoch/acc5_by_class": acc5_by_class,
            }
        )
