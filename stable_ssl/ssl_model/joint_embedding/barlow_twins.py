from dataclasses import dataclass
import torch

from .base import JEConfig, JETrainer


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(JETrainer):
    def compute_ssl_loss(self, z1, z2):
        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.config.model.lambd * off_diag
        return loss


@dataclass
class BarlowTwinsConfig(JEConfig):
    """
    Configuration for the SSL model parameters.

    Parameters:
    -----------
    lambd : str
        Lambda parameter for the off-diagonal loss. Default is 0.1.
    """

    lambd: str = 0.1

    def trainer(self):
        return BarlowTwins
