import torch
from .base import SSLConfig, SSLTrainer
from dataclasses import dataclass
import torch.distributed as dist


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class VICReg(SSLTrainer):
    def compute_ssl_loss(self, z1, z2):

        repr_loss = torch.nn.functional.mse_loss(z1, z2)

        if self.config.hardware.world_size > 1:
            x = torch.cat(FullGatherLayer.apply(z1), dim=0)
            y = torch.cat(FullGatherLayer.apply(z2), dim=0)
        else:
            x = z1
            y = z2
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + self.config.model.epsilon)
        std_y = torch.sqrt(y.var(dim=0) + self.config.model.epsilon)
        std_loss = (
            torch.mean(torch.nn.functional.relu(1 - std_x)) / 2
            + torch.mean(torch.nn.functional.relu(1 - std_y)) / 2
        )

        cov_x = (x.T @ x) / (x.size(0) - 1)
        cov_y = (y.T @ y) / (x.size(0) - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(x.size(1)) + off_diagonal(
            cov_y
        ).pow_(2).sum().div(x.size(1))

        loss = (
            self.config.model.sim_coeff * repr_loss
            + self.config.model.std_coeff * std_loss
            + self.config.model.cov_coeff * cov_loss
        )
        return loss


class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


@dataclass
class VICRegConfig(SSLConfig):
    """
    Configuration for the VICreg model parameters.
    """

    sim_coeff: float = 25
    std_coeff: float = 25
    cov_coeff: float = 1
    epsilon: float = 0.0001

    def trainer(self):
        return VICReg
