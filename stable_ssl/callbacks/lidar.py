from typing import Iterable
import torch
from loguru import logger as logging
from .queue import OnlineQueue
import types


def wrap_validation_step(fn, name):
    def ffn(self, batch, batch_idx, fn=fn, name=name):
        raise NotImplementedError
        batch = fn(batch, batch_idx)
        if batch_idx > 0:
            return batch
        embeddings = getattr(self, f"_cached_{name}_X")
        encoding = self.all_gather(embeddings).flatten(0, 1)
        if self.trainer.global_rank == 0:

            class_means = embeddings.mean(dim=1)
            grand_mean_local = class_means.mean(dim=0)

            local_Sb = torch.zeros(d, d, device=device)
            local_Sw = torch.zeros(d, d, device=device)

            for i in range(local_n):
                diff_b = (class_means[i] - grand_mean_local).unsqueeze(1)
                local_Sb += diff_b @ diff_b.T
                for j in range(q):
                    diff_w = (embeddings[i, j] - class_means[i]).unsqueeze(1)
                    local_Sw += diff_w @ diff_w.T
            S_b = reduce(local_Sb, rank=0, op=dist.ReduceOp.SUM) / (n_total - 1)
            S_w = reduce(local_Sw, rank=0, op=dist.ReduceOp.SUM) / (n_total * (q - 1))
            S_w += self.delta * torch.eye(d, device=device)

            eigvals_w, eigvecs_w = torch.linalg.eigh(S_w)
            eigvals_w = torch.clamp(eigvals_w, min=self.epsilon)

            invsqrt_w = (
                eigvecs_w * (1.0 / torch.sqrt(eigvals_w))
            ) @ eigvecs_w.transpose(-1, -2)
            Sigma_lidar = invsqrt_w @ S_b @ invsqrt_w

            lam, _ = torch.linalg.eigh(Sigma_lidar)
            lam = torch.clamp(lam, min=0.0)

            lam_sum = lam.sum() + self.epsilon
            p = lam / lam_sum

            p_log_p = p * torch.log(p + self.epsilon)

            lidar = float(torch.exp(-p_log_p.sum()))
            return broadcast(torch.tensor([lidar], device=device), src_rank=0).item()

        return batch

    return ffn


class LiDAR(OnlineQueue):
    """LiDAR (Linear Discriminant Analysis Rank) monitor from :cite`thilak2023lidar`."""

    def __init__(
        self,
        pl_module,
        name: str,
        target: str,
        queue_length: int,
        target_shape: Iterable[int],
    ) -> None:
        super().__init__(
            pl_module,
            name=name,
            to_save=[target],
            queue_length=queue_length,
            shapes=[target_shape],
            dtypes=[torch.float],
        )
        logging.info(f"\t- wrapping the `validation_step`")
        fn = wrap_validation_step(pl_module.validation_step, target, input, name)
        pl_module.validation_step = types.MethodType(fn, pl_module)
