"""Data utilities for stable-ssl."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#         Randall Balestriero <randallbalestriero@gmail.com>
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Callable, Iterable, Optional, Union

import torch
from datasets import load_dataset
from typing_extensions import override


class _DatasetSamplerWrapper(torch.utils.data.Dataset):
    """Dataset to create indexes from `Sampler` or `Iterable`."""

    def __init__(self, sampler) -> None:
        self._sampler = sampler
        # defer materializing an iterator until it is necessary
        self._sampler_list = None

    @override
    def __getitem__(self, index: int):
        if self._sampler_list is None:
            self._sampler_list = list(self._sampler)
        return self._sampler_list[index]

    def __len__(self) -> int:
        return len(self._sampler)

    def reset(self) -> None:
        """Reset the sampler list in order to get new sampling."""
        self._sampler_list = list(self._sampler)


class DistributedSamplerWrapper(torch.utils.data.DistributedSampler):
    """Wrap a dataloader for DDP.

    Parameters
    ----------
    sampler: iterable
        The original dataset sampler.
    """

    def __init__(self, sampler, *args, **kwargs) -> None:
        super().__init__(_DatasetSamplerWrapper(sampler), *args, **kwargs)

    @override
    def __iter__(self) -> Iterable:
        """Iterate over DDP dataset.

        Returns
        -------
            Iterable: minibatch
        """
        self.dataset.reset()
        return (self.dataset[index] for index in super().__iter__())


class MultiViewSampler:
    """Apply a list of transforms to an input and return all outputs."""

    def __init__(self, transforms: list):
        logging.info(f"MultiViewSampler initialized with {len(transforms)} views.")
        self.transforms = transforms

    def __call__(self, x):
        views = []
        for t in self.transforms:
            views.append(t(x))
        if len(self.transforms) == 1:
            return views[0]
        return views


class HuggingFaceDataset(torch.utils.data.Dataset):
    """Load a HuggingFace dataset.

    Parameters
    ----------
    *args: list
        Additional arguments to pass to `datasets.load_dataset`.
    rename_columns: dict
        A mapping of names from the HF dataset to what the dict should contain in this dataset.
        For example `{"x":"image", "y":"label"}
    remove_columns: list
        A mapping of names from the HF dataset to what the dict should contain in this dataset.
        For example `{"x":"image", "y":"label"}
    transform: dict[str: callable]
        Which key to transform
    add_index: bool
        Whether to add a key "index" with the datum index
    **kwargs: dict
        Additional keyword arguments to pass to `datasets.load_dataset`.
    """

    def __init__(
        self,
        *args: list,
        rename_columns: dict = None,
        remove_columns: dict = None,
        transform: dict = None,
        add_index: bool = False,
        **kwargs: dict,
    ):
        self.add_index = add_index
        self.transform = transform or {}
        dataset = load_dataset(*args, **kwargs)
        if remove_columns is not None:
            dataset = dataset.remove_columns(remove_columns)

        if rename_columns is not None:
            dataset = dataset.rename_columns(rename_columns)

        self.dataset = dataset

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: Union[int, torch.Tensor]) -> tuple:
        """Get a sample from the dataset.

        Parameters
        ----------
        idx: int or torch.Tensor
            Index to sample from the dataset.

        Returns
        -------
        dict: (str, data)
            A dict containing the data sample.
        """
        if isinstance(idx, torch.Tensor) and idx.dim() == 0:
            idx = idx.item()
        idx = int(idx)

        sample = self.dataset[idx]
        for k, t in self.transform.items():
            sample[k] = t(sample[k])
        assert type(sample) is dict
        if self.add_index:
            if "index" in sample:
                raise ValueError("Tried to add index in data but already present")
            sample["index"] = idx
        return sample
