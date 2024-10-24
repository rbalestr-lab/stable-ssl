import torch
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy

from stable_ssl.utils import load_nn
from stable_ssl import BaseModel


class Supervised(BaseModel):
    r"""Base class for training a supervised model.

    Parameters:
    -----------
    config : BaseModelConfig
        Parameters for BaseModel organized in groups.
        For details, see the `BaseModelConfig` class in `config.py`.
    """

    def initialize_modules(self):
        model, _ = load_nn(
            name=self.config.model.backbone_model,
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

    def initialize_metrics(self):

        nc = self.config.data.num_classes
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
