import torch
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy

from .utils import load_nn
from .base import BaseModel


class Supervised(BaseModel):
    r"""Base class for training a supervised model.

    Parameters
    ----------
    config : BaseModelConfig
        Parameters for BaseModel organized in groups.
        For details, see the `BaseModelConfig` class in `config.py`.
    """

    def initialize_modules(self):
        model, _ = load_nn(
            backbone_model=self.config.model.backbone_model,
            n_classes=self.config.data.num_classes,
            with_classifier=self.config.model.with_classifier,
            pretrained=self.config.model.pretrained,
        )
        self.model = model.train()

    def forward(self, x):
        return self.model(x)

    def compute_loss(self):
        preds = self.forward(self.data[0])
        self.log(
            {"train/step/acc1": self.metrics["train/step/acc1"](preds, self.data[1])},
            commit=False,
        )
        return F.cross_entropy(preds, self.data[1])
