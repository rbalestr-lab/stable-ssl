import itertools
import math
import warnings
from collections.abc import Sequence

# UP006 wants 'Iterable' to be imported from collections.abc but it needs to
# stay from typing for now due to BC concerns. In particular several internal
# targets fail to typecheck with:
#     TypeError: Cannot create a consistent method resolution order (MRO) for
#     bases Iterable, Generic
from typing import Optional, Union, cast  # noqa: UP035

import lightning as pl
import numpy as np
import torch
from loguru import logger as logging

# No 'default_generator' in torch/__init__.pyi
from torch import Generator, default_generator, randperm


class Dataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self._trainer = None

    def set_pl_trainer(self, trainer: pl.Trainer):
        self._trainer = trainer

    def process_sample(self, sample):
        if self._trainer is not None:
            if "global_step" in sample:
                raise ValueError("Can't use that keywords")
            if "current_epoch" in sample:
                raise ValueError("Can't use that keywords")
            sample["global_step"] = self._trainer.global_step
            sample["current_epoch"] = self._trainer.current_epoch
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class Subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    dataset: Dataset
    indices: Sequence[int]

    def __init__(self, dataset: Dataset, indices: Sequence[int]) -> None:
        super().__init__()
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __getitems__(self, indices: list[int]) -> list:
        # add batched sampling support when parent dataset supports it.
        # see torch.utils.data._utils.fetch._MapDatasetFetcher
        if callable(getattr(self.dataset, "__getitems__", None)):
            return self.dataset.__getitems__([self.indices[idx] for idx in indices])  # type: ignore[attr-defined]
        else:
            return [self.dataset[self.indices[idx]] for idx in indices]

    def __len__(self):
        return len(self.indices)

    @property
    def column_names(self):
        return self.dataset.column_names


class FromTorchDataset(Dataset):
    def __init__(self, dataset, names, transform):
        super().__init__(transform)
        self.dataset = dataset
        self.names = names

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        sample = {k: v for k, v in zip(self.names, sample)}
        return self.process_sample(sample)

    def __len__(self):
        return len(self.dataset)

    @property
    def column_names(self):
        return self.names


class MinariStepsDataset(Dataset):
    NAMES = ["observations", "actions", "rewards", "terminations", "truncations"]

    def __init__(self, dataset, num_steps=2, transform=None):
        super().__init__(transform)
        self.num_steps = num_steps
        self.dataset = dataset
        self.bounds = self.dataset.episode_indices
        self.bounds -= np.arange(self.dataset.total_episodes) * (num_steps - 1)
        self._length = (
            self.dataset.total_steps - (num_steps - 1) * self.dataset.total_episodes
        )
        logging.info("Minari Dataset setup")
        logging.info(f"\t- {self.dataset.total_episodes} episodes")
        logging.info(f"\t- {len(self)} steps")

    def nested_step(self, value, idx):
        if type(value) is dict:
            return {k: self.nested_step(v, idx) for k, v in value.items()}
        return value[idx : idx + self.num_steps]

    def __getitem__(self, idx):
        ep_idx = np.searchsorted(self.bounds, idx, side="right") - 1
        frame_idx = idx - self.bounds[ep_idx]
        episode = self.dataset[ep_idx]
        sample = {
            name: self.nested_step(getattr(episode, name), frame_idx)
            for name in self.NAMES
        }
        return self.process_sample(sample)

    def __len__(self):
        return self._length

    @property
    def column_names(self):
        return self.names


class MinariEpisodeDataset(torch.utils.data.Dataset):
    NAMES = ["observations", "actions", "rewards", "terminations", "truncations"]

    def __init__(self, dataset):
        self.dataset = dataset
        self.bounds = self.dataset.episode_indices
        self._trainer = None

        logging.info("Minari Dataset setup")
        logging.info(f"\t- {self.dataset.total_episodes} episodes")
        logging.info(f"\t- {len(self)} steps")

    def set_pl_trainer(self, trainer: pl.Trainer):
        self._trainer = trainer

    def nested_step(self, value, idx):
        if type(value) is dict:
            return {k: self.nested_step(v, idx) for k, v in value.items()}
        return value[idx]

    def __getitem__(self, idx):
        ep_idx = np.searchsorted(self.bounds, idx, side="right") - 1
        frame_idx = idx - self.bounds[ep_idx]
        print(ep_idx, frame_idx)
        episode = self.dataset[ep_idx]
        sample = {
            name: self.nested_step(getattr(episode, name), frame_idx)
            for name in self.NAMES
        }
        if self._trainer is not None:
            if "global_step" in sample:
                raise ValueError("Can't use that keywords")
            if "current_epoch" in sample:
                raise ValueError("Can't use that keywords")
            sample["global_step"] = self._trainer.global_step
            sample["current_epoch"] = self._trainer.current_epoch
        return sample

    def __len__(self):
        return self.dataset.total_steps

    @property
    def column_names(self):
        return self.names


class HFDataset(Dataset):
    def __init__(
        self, *args, transform=None, rename_columns=None, remove_columns=None, **kwargs
    ):
        super().__init__(transform)
        import datasets

        if (
            torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            import time

            s = int(torch.distributed.get_rank()) * 2
            logging.info(
                f"Sleeping for {s}s to avoid race condition of dataset cache"
                " see https://github.com/huggingface/transformers/issues/15976)"
            )
            time.sleep(s)
        if "storage_options" not in kwargs:
            logging.warning(
                "You didn't pass a storage optionwe are adding one to avoid timeout"
            )
            from aiohttp import ClientTimeout

            kwargs["storage_options"] = {
                "client_kwargs": {"timeout": ClientTimeout(total=3600)}
            }
        dataset = datasets.load_dataset(*args, **kwargs)
        dataset = dataset.add_column("sample_idx", list(range(dataset.num_rows)))
        if rename_columns is not None:
            for k, v in rename_columns.items():
                dataset = dataset.rename_column(k, v)
        if remove_columns is not None:
            dataset = dataset.remove_columns(remove_columns)
        self.dataset = dataset

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return self.process_sample(sample)

    def __len__(self):
        return self.dataset.num_rows

    @property
    def column_names(self):
        return self.dataset.column_names


class Categorical(torch.nn.Module):
    def __init__(
        self,
        values: Union[list, torch.Tensor],
        probabilities: Union[list, torch.Tensor],
    ):
        super().__init__()
        self.mix = torch.distributions.Categorical(torch.Tensor(probabilities))
        self.values = torch.Tensor(values)
        print(self.mix, self.values)

    def __call__(self):
        return self.values[self.mix.sample()]

    def sample(self, *args, **kwargs):
        return self.values[self.mix.sample(*args, **kwargs)]


class ExponentialMixtureNoiseModel(torch.nn.Module):
    def __init__(self, rates, prior, upper_bound=torch.inf):
        super().__init__()
        mix = torch.distributions.Categorical(torch.Tensor(prior))
        comp = torch.distributions.Exponential(torch.Tensor(rates))
        self.mm = torch.distributions.MixtureSameFamily(mix, comp)
        self.upper_bound = upper_bound

    def __call__(self):
        return self.mm.sample().clip_(min=0, max=self.upper_bound)

    def sample(self, *args, **kwargs):
        return self.mm.sample(*args, **kwargs).clip_(min=0, max=self.upper_bound)


class ExponentialNormalNoiseModel(torch.nn.Module):
    def __init__(self, rate, mean, std, prior, upper_bound=torch.inf):
        super().__init__()
        self.mix = torch.distributions.Categorical(torch.Tensor(prior))
        self.exp = torch.distributions.Exponential(rate)
        self.gauss = torch.distributions.Normal(mean, std)
        self.upper_bound = upper_bound

    def __call__(self):
        mix = self.mix.sample()
        if mix == 0:
            return self.exp.sample().clip_(min=0, max=self.upper_bound)
        return self.gauss.sample().clip_(min=0, max=self.upper_bound)

    def sample(self, *args, **kwargs):
        mix = self.mix.sample(*args, **kwargs)
        exp = self.exp.sample(*args, **kwargs)
        gauss = self.gauss.sample(*args, **kwargs)
        return torch.where(mix.bool(), gauss, exp).clip_(min=0, max=self.upper_bound)


def fold_views(tensor, idx):
    _, counts = torch.unique_consecutive(idx, return_counts=True)
    if not counts.min().eq(counts.max()):
        raise RuntimeError("counts are not the same for all samples!")
    n_views = counts[0].item()
    fold_shape = (tensor.size(0) // n_views, n_views)
    t = tensor.view(*fold_shape, *tensor.shape[1:])
    return t.unbind(dim=1)


def random_split(
    dataset: Dataset,
    lengths: Sequence[Union[int, float]],
    generator: Optional[Generator] = default_generator,
) -> list[Subset]:
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.

    Optionally fix the generator for reproducible results, e.g.:

    Example:
        >>> # xdoctest: +SKIP
        >>> generator1 = torch.Generator().manual_seed(42)
        >>> generator2 = torch.Generator().manual_seed(42)
        >>> random_split(range(10), [3, 7], generator=generator1)
        >>> random_split(range(30), [0.3, 0.3, 0.4], generator=generator2)

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: list[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(
                    f"Length of split at index {i} is 0. "
                    f"This might result in an empty dataset."
                )

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):  # type: ignore[arg-type]
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )

    indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[arg-type, call-overload]
    lengths = cast(Sequence[int], lengths)
    return [
        Subset(dataset, indices[offset - length : offset])
        for offset, length in zip(itertools.accumulate(lengths), lengths)
    ]
