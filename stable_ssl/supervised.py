import torch.nn.functional as F

from .base import BaseModel


class Supervised(BaseModel):
    r"""Base class for training a supervised model.

    Parameters
    ----------
    config : BaseModelConfig
        Parameters for BaseModel organized in groups.
        For details, see the `BaseModelConfig` class in `config.py`.
    """

    def forward(self, x):
        return self.config.networks["backbone"](x)

    def compute_loss(self):
        predictions = [self.forward(view) for view in self.data[0]]
        loss = sum([F.cross_entropy(pred, self.data[1]) for pred in predictions])

        if self.global_step % self.config.log.log_every_step == 0:
            self.log(
                {
                    "train/loss": loss.item(),
                },
                commit=False,
            )

        return loss
