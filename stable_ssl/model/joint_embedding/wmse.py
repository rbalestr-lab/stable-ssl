import torch
from .base import SSLConfig, SSLTrainer
from dataclasses import dataclass
import torch.distributed as dist


class WMSE(SSLTrainer):
    def initialize_modules(self):
        super().initialize_modules()
        self.whitening = Whitening2d(
            self.config.model.projector[-1],
            eps=self.config.model.w_eps,
            track_running_stats=False,
        )

    def compute_ssl_loss(self, embeds):
        n_views = 2
        h = self.projector(embeds)
        bs = h.size(0) // n_views
        loss = 0
        for _ in range(self.w_iter):
            z = torch.empty_like(h)
            perm = torch.randperm(bs).view(-1, bs)
            for idx in perm:
                for i in range(n_views):
                    z[idx + i * bs] = self.whitening(h[idx + i * bs])
            for i in range(n_views - 1):
                for j in range(i + 1, n_views):
                    x0 = z[i * bs : (i + 1) * bs]
                    x1 = z[j * bs : (j + 1) * bs]
                    loss += self.loss_f(x0, x1)
        loss /= self.w_iter * self.num_pairs

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


class Whitening2d(torch.nn.Module):
    def __init__(self, num_features, momentum=0.01, track_running_stats=True, eps=0):
        super(Whitening2d, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        self.eps = eps

        if self.track_running_stats:
            self.register_buffer(
                "running_mean", torch.zeros([1, self.num_features, 1, 1])
            )
            self.register_buffer("running_variance", torch.eye(self.num_features))

    def forward(self, x):
        x = x.unsqueeze(2).unsqueeze(3)
        m = x.mean(0).view(self.num_features, -1).mean(-1).view(1, -1, 1, 1)
        if not self.training and self.track_running_stats:  # for inference
            m = self.running_mean
        xn = x - m

        T = xn.permute(1, 0, 2, 3).contiguous().view(self.num_features, -1)
        f_cov = torch.mm(T, T.permute(1, 0)) / (T.shape[-1] - 1)

        eye = torch.eye(self.num_features).type(f_cov.type())

        if not self.training and self.track_running_stats:  # for inference
            f_cov = self.running_variance

        f_cov_shrinked = (1 - self.eps) * f_cov + self.eps * eye

        inv_sqrt = torch.linalg.solve_triangular(
            torch.linalg.cholesky(f_cov_shrinked), eye, upper=False
        )

        inv_sqrt = inv_sqrt.contiguous().view(
            self.num_features, self.num_features, 1, 1
        )

        decorrelated = torch.nn.functional.conv2d(xn, inv_sqrt)

        if self.training and self.track_running_stats:
            self.running_mean = torch.add(
                self.momentum * m.detach(),
                (1 - self.momentum) * self.running_mean,
                out=self.running_mean,
            )
            self.running_variance = torch.add(
                self.momentum * f_cov.detach(),
                (1 - self.momentum) * self.running_variance,
                out=self.running_variance,
            )

        return decorrelated.squeeze(2).squeeze(2)

    def extra_repr(self):
        return "features={}, eps={}, momentum={}".format(
            self.num_features, self.eps, self.momentum
        )


@dataclass
class WMSEConfig(SSLConfig):
    """
    Configuration for the VICreg model parameters.
    """

    w_iter: float = 1
    w_eps: float = 0

    def trainer(self):
        return WMSE
