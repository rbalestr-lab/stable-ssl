from dataclasses import dataclass
import torch
import torch.nn.functional as F

from .base import JEConfig, JETrainer


class SimCLR(JETrainer):
    """SimCLR model for self-supervised learning."""

    def compute_ssl_loss(self, h_i, h_j):
        """Compute the contrastive loss for SimCLR.

        Parameters
        ----------
        h_i : torch.Tensor
            Latent representation of the first augmented view of the batch.
        h_j : torch.Tensor
            Latent representation of the second augmented view of the batch.

        Returns
        -------
        float
            The computed contrastive loss.
        """
        z = torch.cat([h_i, h_j], 0)

        N = z.size(0) * self.config.hardware.world_size

        features = F.normalize(z, dim=1)
        sim = torch.matmul(features, features.T) / self.config.model.temperature

        sim_i_j = torch.diag(sim, N // 2)
        sim_j_i = torch.diag(sim, -N // 2)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0)  # shape (N)

        mask = torch.eye(N, dtype=bool).to(self.this_device)
        negative_samples = sim[~mask].reshape(N, -1)  # shape (N, N-1)

        attraction = -positive_samples.mean()
        repulsion = torch.logsumexp(negative_samples, dim=1).mean()

        return attraction + repulsion


@dataclass
class SimCLRConfig(JEConfig):
    """Configuration for the SSL model parameters.

    Parameters
    ----------
    temperature : float
        Temperature parameter for the contrastive loss. Default is 0.15.
    """

    temperature: float = 0.15

    def trainer(self):
        """Return the corresponding trainer for the SimCLR configuration.

        Returns
        -------
        SimCLR
            A SimCLR trainer instance.
        """
        return SimCLR
