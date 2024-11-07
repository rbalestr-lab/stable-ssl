import torch.nn.functional as F

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
        backbone, fan_in = load_nn(
            backbone_model=self.config.model.backbone_model,
            n_classes=self.config.data.train_dataset.num_classes,
            pretrained=False,
            dataset=self.config.data.train_dataset.name,
        )
        self.backbone = backbone.train()

    def forward(self, x):
        return self.backbone(x)

    def compute_loss(self):
        preds = self.forward(self.data[0])

        if self.batch_idx % self.config.log.log_every_step == 0:
            self.log(
                {"train/acc1": self.metrics["train/acc1"](preds, self.data[1])},
                commit=False,
            )

        return F.cross_entropy(preds, self.data[1])
