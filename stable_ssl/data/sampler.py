import math
from typing import Iterable, Iterator, List, Union

import numpy as np
import torch
import torch.distributed as dist


def _check_integer(v, name):
    if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
        raise ValueError(
            f"{name} should be a positive integer value, but got {name}={v}"
        )


class RepeatedRandomSampler(torch.utils.data.DistributedSampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.

    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        n_views (int): number of views to repeat each sample, default=1
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
    """

    def __init__(
        self,
        data_source_or_len: Union[int, Iterable],
        n_views: int = 1,
        replacement: bool = False,
        seed: int = 0,
    ):
        if type(data_source_or_len) is int:
            self._data_source_len = data_source_or_len
        else:
            self._data_source_len = len(data_source_or_len)
        self.replacement = replacement
        self.n_views = n_views
        self.seed = seed
        self.epoch = 0

        if dist.is_available() and dist.is_initialized():
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
            if self.rank >= self.num_replicas or self.rank < 0:
                raise ValueError(
                    f"Invalid rank {self.rank}, rank should be in the interval [0, {self.num_replicas - 1}]"
                )
        else:
            self.num_replicas = 1
            self.rank = 0
        if self._data_source_len % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (self._data_source_len - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = self._data_source_len

        if not isinstance(self.replacement, bool):
            raise TypeError(
                f"replacement should be a boolean value, but got replacement={self.replacement}"
            )

    def __len__(self):
        return self.num_samples * self.n_views

    # @property
    # def num_samples(self) -> int:
    #     return (len(self.data_source) * self.n_views) // self.num_replicas

    def __iter__(self) -> Iterator[int]:
        """Iterate over the indices of the dataset.

        Note: need to append the worker_id to the seed to ensure that each worker
        gets a different random sequence. worker_init_fn is called only at the beginning

        https://pytorch-lightning.readthedocs.io/en/1.7.7/api/pytorch_lightning.utilities.seed.html#pytorch_lightning.utilities.seed.seed_everything
        """
        n = self._data_source_len
        g = torch.Generator()
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        g.manual_seed(self.seed + self.epoch + worker_id)

        if self.replacement:
            raise NotImplementedError()
            for _ in range(self.num_samples // 32):
                yield from torch.randint(
                    high=n, size=(32,), dtype=torch.int64, generator=g
                ).tolist()
            yield from torch.randint(
                high=n,
                size=(self.num_samples % 32,),
                dtype=torch.int64,
                generator=g,
            ).tolist()
        else:
            overall_slice = torch.randperm(n, generator=g)
            rank_slice = overall_slice[
                self.rank * self.num_samples : (self.rank + 1) * self.num_samples
            ]
            yield from rank_slice.repeat_interleave(self.n_views).tolist()


# class RepeatedSampler(torch.utils.data.BatchSampler):
#     def __init__(
#         self,
#         num_samples_or_dataset: Union[torch.utils.data.Dataset, HFDataset, int],
#         batch_size: Optional[int] = 1,
#         n_views: Optional[int] = 1,
#         shuffle: bool = True,
#         replacement: bool = False,
#         sampler=None,
#         drop_last: bool = False,
#     ) -> None:
#         # if hasattr(num_samples_or_dataset, "__len__"):
#         #     num_samples_or_dataset = len(num_samples_or_dataset)
#         if sampler is None and shuffle:
#             sampler = torch.utils.data.RandomSampler(
#                 num_samples_or_dataset, replacement=replacement
#             )
#         elif sampler is None and not shuffle:
#             sampler = torch.utils.data.SequentialSampler(num_samples_or_dataset)
#         super().__init__(sampler=sampler, batch_size=batch_size, drop_last=drop_last)
#         _check_integer(n_views, "n_views")
#         self.batch_size = batch_size
#         self.n_views = n_views

#     def __iter__(self) -> Iterator[List[int]]:
#         # Create multiple references to the same iterator
#         sampler_iter = iter(self.sampler)
#         args = [sampler_iter] * (self.batch_size // self.n_views)
#         for batch_droplast in zip(*args):
#             indices = [*batch_droplast]
#             yield [int(i) for i in np.repeat(indices, self.n_views)]

#     def __len__(self) -> int:
#         # Can only be called if self.sampler has __len__ implemented
#         # We cannot enforce this condition, so we turn off typechecking for the
#         # implementation below.
#         return len(self.sampler) // (self.batch_size // self.n_views)


class SupervisedBatchSampler(torch.utils.data.Sampler[List[int]]):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(
        ...     BatchSampler(
        ...         SequentialSampler(range(10)), batch_size=3, drop_last=False
        ...     )
        ... )
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(
        ...     BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True)
        ... )
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(
        self,
        batch_size: int,
        n_views: int,
        targets_or_dataset: Union[torch.utils.data.Dataset, list],
        *args,
        **kwargs,
    ) -> None:
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError(
                f"batch_size should be a positive integer value, but got batch_size={batch_size}"
            )
        if not isinstance(n_views, int) or isinstance(n_views, bool) or n_views <= 0:
            raise ValueError(
                f"n_views should be a positive integer value, but got n_views={n_views}"
            )
        self.batch_size = batch_size
        self.n_views = n_views
        if isinstance(targets_or_dataset, torch.utils.data.Dataset):
            targets = targets_or_dataset.targets
        else:
            targets = targets_or_dataset
        self._length = len(targets)

        self.batches = {}
        unique_targets, counts = np.unique(targets, return_counts=True)
        self.prior = counts / counts.sum()
        for label in unique_targets:
            self.batches[label.item()] = np.flatnonzero(targets == label)

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(len(self)):
            n_parents = self.batch_size // self.n_views
            parents = np.random.choice(
                list(self.batches.keys()), size=n_parents, replace=True, p=self.prior
            )
            indices = []
            for p in parents:
                indices.extend(
                    np.random.choice(self.batches[p], size=self.n_views, replace=False)
                )
            indices = np.asarray(indices).astype(int)
            yield indices

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        return self._length // self.batch_size // self.n_views


class RandomBatchSampler(torch.utils.data.Sampler[List[int]]):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(
        ...     BatchSampler(
        ...         SequentialSampler(range(10)), batch_size=3, drop_last=False
        ...     )
        ... )
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(
        ...     BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True)
        ... )
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(
        self,
        batch_size: int,
        length_or_dataset: Union[torch.utils.data.Dataset, int],
        *args,
        **kwargs,
    ) -> None:
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError(
                f"batch_size should be a positive integer value, but got batch_size={batch_size}"
            )
        self.batch_size = batch_size
        if isinstance(length_or_dataset, torch.utils.data.Dataset):
            length_or_dataset = len(length_or_dataset)
        self._length = length_or_dataset

    def __iter__(self) -> Iterator[List[int]]:
        perm = np.random.permutation(self._length).astype(int)
        for i in range(len(self)):
            yield perm[i * self.batch_size : (i + 1) * self.batch_size]

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        return len(self.sampler) // self.batch_size // self.n_views
