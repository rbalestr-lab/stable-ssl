import torch
import torch.nn.functional as F

from .base import SSLTrainer
from stable_ssl.config import TrainerConfig


class SimCLR(SSLTrainer):
    """
    SimCLR Loss:
    When using a batch size of 2048, use LARS as optimizer
    with a base learning rate of 0.5, weight decay of 1e-6 and a temperature of 0.15.
    When using a batch size of 256, use LARS as optimizer
    with base learning rate of 1.0, weight decay of 1e-6 and a temperature of 0.15.
    """

    def __init__(self, config: TrainerConfig):
        super().__init__(config)

    def initialize_modules(self):
        # backbone
        model, fan_in = load_model(
            name=self.config.model.backbone_model,
            n_classes=10,
            with_classifier=False,
            pretrained=False,
        )
        self.backbone = model.train()

        # projector
        sizes = [fan_in] + list(map(int, self.config.model.projector.split("-")))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # linear probes
        self.classifier = torch.nn.Linear(fan_in, 10)

    def forward(self, x):
        return self.backbone(x)

    def compute_ssl_loss(self, embeds):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the
        other 2(N-1) augmented examples within a minibatch as negative examples.
        """
        projs = self.projector(embeds)

        z_i, z_j = torch.chunk(projs, 2, dim=0)
        batch_size = z_i.size(0)
        N = 2 * batch_size * self.config.hardware.world_size

        mask = self._mask_correlated_samples(
            batch_size, self.config.hardware.world_size
        ).to(self.this_device)

        if self.config.hardware.world_size > 1:
            z_i = torch.cat(self.gather(z_i), dim=0)
            z_j = torch.cat(self.gather(z_j), dim=0)

        z = torch.cat((z_i, z_j), dim=0)

        features = F.normalize(z, dim=1)
        sim = torch.matmul(features, features.T) / self.config.model.temperature

        sim_i_j = torch.diag(sim, batch_size * self.config.hardware.world_size)
        sim_j_i = torch.diag(sim, -batch_size * self.config.hardware.world_size)

        # We have 2N samples, but with Distributed training every GPU gets
        # N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[mask].reshape(N, -1)
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        logits_num = logits
        logits_denum = torch.logsumexp(logits, dim=1, keepdim=True)
        num_sim = (-logits_num[:, 0]).sum() / N
        num_entropy = logits_denum[:, 0].sum() / N
        return num_sim + num_entropy

    @staticmethod
    def _mask_correlated_samples(batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size * world_size + i] = 0
            mask[batch_size * world_size + i, i] = 0
        return mask
