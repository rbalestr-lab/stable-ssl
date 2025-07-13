from loguru import logger as logging
from typing import Union

import lightning as pl
import torch
import numpy as np


class FromTorchDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, names):
        self.dataset = dataset
        self.names = names

    def set_pl_trainer(self, trainer: pl.Trainer):
        self._trainer = trainer

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        sample = {k: v for k, v in zip(self.names, sample)}
        if self._trainer is not None:
            if "global_step" in sample:
                raise ValueError("Can't use that keywords")
            if "current_epoch" in sample:
                raise ValueError("Can't use that keywords")
            sample["global_step"] = self._trainer.global_step
            sample["current_epoch"] = self._trainer.current_epoch
        return sample

    def __len__(self):
        return len(self.dataset)

    @property
    def column_names(self):
        return self.names


class MinariStepsDataset(torch.utils.data.Dataset):
    NAMES = ["observations", "actions", "rewards", "terminations", "truncations"]

    def __init__(self, dataset, num_steps=2):
        self.num_steps = num_steps
        self.dataset = dataset
        self.bounds = self.dataset.episode_indices
        self.bounds -= np.arange(self.dataset.total_episodes) * (num_steps - 1)
        self._length = (
            self.dataset.total_steps - (num_steps - 1) * self.dataset.total_episodes
        )
        self._trainer = None
        logging.info("Minari Dataset setup")
        logging.info(f"\t- {self.dataset.total_episodes} episodes")
        logging.info(f"\t- {len(self)} steps")

    def set_pl_trainer(self, trainer: pl.Trainer):
        self._trainer = trainer

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
        if self._trainer is not None:
            if "global_step" in sample:
                raise ValueError("Can't use that keywords")
            if "current_epoch" in sample:
                raise ValueError("Can't use that keywords")
            sample["global_step"] = self._trainer.global_step
            sample["current_epoch"] = self._trainer.current_epoch
        return sample

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


class HFDataset(torch.utils.data.Dataset):
    def __init__(
        self, *args, transform=None, rename_columns=None, remove_columns=None, **kwargs
    ):
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
        self.transform = transform
        self.dataset = dataset
        self._trainer = None

    def set_pl_trainer(self, trainer: pl.Trainer):
        self._trainer = trainer

    def __getitem__(self, idx):
        sample = self.dataset[idx]
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
