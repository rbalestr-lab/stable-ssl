import logging
from typing import Iterable, Optional, Union

import numpy as np
import torch
from typing_extensions import override

from stable_ssl.utils import log_and_raise


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
        the original dataloader sampler
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
        logging.info(f"MultiViewSampler initialized with {len(transforms)} views")
        self.transforms = transforms

    def __call__(self, x):
        views = []
        for t in self.transforms:
            views.append(t(x))
        if len(self.transforms) == 1:
            return views[0]
        return views


# def load_dataset(dataset_name, data_path, train=True):
#     """
#     Load a dataset from torchvision.datasets.
#     Uses PositivePairSampler for training and ValSampler for validation.
#     If coeff_imbalance is not None, create an imbalanced version of the dataset with
#     the specified coefficient (exponential imbalance).
#     """

#     if not hasattr(torchvision.datasets, dataset_name):
#         raise ValueError(f"Dataset {dataset_name} not found in torchvision.datasets.")

#     torchvision_dataset = getattr(torchvision.datasets, dataset_name)

#     if train:
#         return torchvision_dataset(
#             root=data_path,
#             train=True,
#             download=True,
#             transform=Sampler(dataset=dataset_name),
#         )

#     return torchvision_dataset(
#         root=data_path,
#         train=False,
#         download=True,
#         transform=ValSampler(dataset=dataset_name),
#     )


# def imbalance_torchvision_dataset(
#     data_path, dataset, dataset_name, coeff_imbalance=2.0
# ):
#     save_path = os.path.join(data_path, f"imbalanced_coeff_{coeff_imbalance}.pt")

#     if not os.path.exists(save_path):
#         data, labels = from_torchvision(data_path=data_path, dataset=dataset)
#         imbalanced_data, imbalanced_labels = resample_classes(
#             data, labels, coeff_imbalance=coeff_imbalance
#         )
#       imbalanced_dataset = {"features": imbalanced_data, "labels": imbalanced_labels}
#         save_path = os.path.join(data_path, f"imbalanced_coeff_{coeff_imbalance}.pt")
#         torch.save(imbalanced_dataset, save_path)

#         print(f"[stable-SSL] Subsampling : imbalanced dataset saved to {save_path}.")

#     return CustomTorchvisionDataset(
#         root=save_path, transform=PositivePairSampler(dataset=dataset_name)
#     )


def resample_classes(dataset, samples_or_freq, random_seed=None):
    """Create an exponential class imbalance.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The input dataset.
    samples_or_freq : iterable
        Number of samples or frequency for each class in the new dataset.
    random_seed : int, optional
        The random seed for reproducibility. Default is None.

    Returns
    -------
    torch.utils.data.Subset
        Subset of the dataset with the resampled classes.

    Raises
    ------
    ValueError
        If the dataset does not have 'labels' or 'targets' attributes.
    """
    if hasattr(dataset, "labels"):
        labels = dataset.labels
    elif hasattr(dataset, "targets"):
        labels = dataset.targets
    else:
        log_and_raise(ValueError, "Dataset does not have `labels`.")

    classes, class_inverse, class_counts = np.unique(
        labels, return_counts=True, return_inverse=True
    )

    logging.info(f"Subsampling : original class counts: {list(class_counts)}")

    if np.min(samples_or_freq) < 0:
        log_and_raise(
            ValueError,
            "There can't be any negative values in `samples_or_freq`, "
            "got {samples_or_freq}.",
        )
    elif np.sum(samples_or_freq) <= 1:
        target_class_counts = np.array(samples_or_freq) * len(dataset)
    elif np.sum(samples_or_freq) == len(dataset):
        freq = np.array(samples_or_freq) / np.sum(samples_or_freq)
        target_class_counts = freq * len(dataset)
        if (target_class_counts / class_counts).max() > 1:
            log_and_raise(
                ValueError, "Specified more samples per class than available."
            )
    else:
        log_and_raise(
            ValueError,
            "Samples_or_freq needs to sum to <= 1 or len(dataset) "
            f"({len(dataset)}), got {np.sum(samples_or_freq)}.",
        )

    target_class_counts = (
        target_class_counts / (target_class_counts / class_counts).max()
    ).astype(int)

    logging.info(f"Subsampling : target class counts: {list(target_class_counts)}")

    keep_indices = []
    generator = np.random.Generator(np.random.PCG64(seed=random_seed))
    for cl, count in zip(classes, target_class_counts):
        cl_indices = np.flatnonzero(class_inverse == cl)
        cl_indices = generator.choice(cl_indices, size=count, replace=False)
        keep_indices.extend(cl_indices)

    return torch.utils.data.Subset(dataset, indices=keep_indices)


class HuggingFace(torch.utils.data.Dataset):
    """Load a HuggingFace dataset.

    Parameters
    ----------
    x: str
        name of the column to treat as x (input)
    y: str
        name of the column to treat as y (label)
    transform (optional): callable
        transform to apply on x
    *args: list
        args to pass to datasets.load_dataset
    **kwargs: dict
        kwargs to pass to datasets.load_dataset
    """

    def __init__(
        self, x: str, y: str, transform: Optional[callable], *args: list, **kwargs: dict
    ):
        from datasets import load_dataset

        self.dataset = load_dataset(*args, **kwargs)
        assert x in self.dataset.column_names
        assert y in self.dataset.column_names
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self) -> int:
        """Get length of dataset.

        Returns
        -------
            int: length
        """
        return len(self.dataset)

    def __getitem__(self, i: Union[int, torch.Tensor]) -> tuple:
        """Get an sample.

        Parameters
        ----------
        i: int
            index to sample from the dataset

        Returns
        -------
            tuple: (transform(x), y)
        """
        if torch.is_tensor(i) and i.dim() == 0:
            i = i.item()
        x = self.dataset[i][self.x]
        if self.transform is not None:
            xt = self.transform(x)
        else:
            xt = x
        y = self.dataset[i][self.y]
        return xt, y
