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
    path: str
        Path to the dataset (can be a Hugging Face dataset name or a local path).
    x: str
        Name of the column to treat as x (input).
    y: str
        Name of the column to treat as y (label).
    transform (optional): callable, default=None
        Transform to apply on x. By default, no transform is applied (identity transform).
    *args: list
        Additional arguments to pass to `datasets.load_dataset`.
    **kwargs: dict
        Additional keyword arguments to pass to `datasets.load_dataset`.
    """

    def __init__(
        self,
        path: str,
        x: str,
        y: str,
        transform: Optional[Callable] = None,
        *args: list,
        **kwargs: dict,
    ):
        self.dataset = load_dataset(path, *args, **kwargs)

        assert x in self.dataset.column_names, f"Column '{x}' not found in the dataset."
        assert y in self.dataset.column_names, f"Column '{y}' not found in the dataset."

        self.x = x
        self.y = y
        self.transform = transform if transform else lambda x: x

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
        tuple: (transformed x, y)
            A tuple containing the transformed input (x) and the label (y).
        """
        if isinstance(idx, torch.Tensor) and idx.dim() == 0:
            idx = idx.item()

        x = self.dataset[idx][self.x]
        y = self.dataset[idx][self.y]

        x_transformed = self.transform(x)

        return x_transformed, y
