import types
from typing import Iterable

import torch
from loguru import logger as logging

from .queue import OnlineQueue


def wrap_validation_step(fn, name):
    def ffn(self, batch, batch_idx, fn=fn, name=name):
        batch = fn(batch, batch_idx)
        if batch_idx > 0:
            return batch
        logging.info(f"{name}: batch 0 of validation step, computing RankMe")
        embeddings = list(getattr(self, "_callbacks_validation_cache")[name].values())[
            0
        ]
        encoding = self.all_gather(embeddings).flatten(0, 1)
        if self.trainer.global_rank == 0:
            s = torch.linalg.svdvals(encoding)
            p = (s / torch.sum(s, axis=0)) + 1e-5
            entropy = -torch.sum(p * torch.log(p))
            rankme = torch.exp(entropy)
            self.log(name, rankme.item())
        return batch

    return ffn


class RankMe(OnlineQueue):
    """RankMe (effective rank) monitor from :cite:`garrido2023rankme`."""

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
            dims=[target_shape],
            dtypes=[torch.float],
        )
        logging.info("\t- wrapping the `validation_step`")
        fn = wrap_validation_step(pl_module.validation_step, name)
        pl_module.validation_step = types.MethodType(fn, pl_module)
