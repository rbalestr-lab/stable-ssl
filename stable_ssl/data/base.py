import os
from dataclasses import dataclass
import logging
import numpy as np
import hydra

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image

from .augmentations import TransformsConfig


@dataclass
class DatasetConfig:
    """Configuration for the data used for training the model.

    Parameters
    ----------
    dir : str, optional
        Path to the directory containing the training data.
        Default is "data".
    name : str, optional
        Name of the dataset to use (e.g., "CIFAR10", "CIFAR100").
        Default is "CIFAR10".
    split : str, optional
        Name of the dataset split to use (e.g., "train", "test").
        Default is "train".
    num_workers : int, optional
        Number of workers to use for data loading.
        Default is -1 (use all available CPUs).
    batch_size : int, optional
        Batch size for training. Default is 256.
    transforms : dict, optional
        Dictionary of transformations to apply to the data. Default is None.
    drop_last : bool, optional
        Whether to drop the last incomplete batch. Default is False.
    shuffle : bool, optional
        Whether to shuffle the data. Default is False.
    """

    dir: str = "data"
    name: str = "CIFAR10"
    split: str = "train"
    num_workers: int = -1
    batch_size: int = 256
    transforms: list[TransformsConfig] = None
    drop_last: bool = False
    shuffle: bool = False

    def __post_init__(self):
        """Initialize transforms if not provided."""
        if self.transforms is None:
            self.transforms = [TransformsConfig("None")]
        else:
            self.transforms = [
                TransformsConfig(name, t) for name, t in self.transforms.items()
            ]

    @property
    def num_classes(self):
        """Return the number of classes in the dataset."""
        if self.name == "CIFAR10":
            return 10
        elif self.name == "CIFAR100":
            return 100

    @property
    def resolution(self):
        """Return the resolution of the images in the dataset."""
        if self.name in ["CIFAR10", "CIFAR100"]:
            return 32

    @property
    def data_path(self):
        """Return the path to the dataset."""
        return os.path.join(hydra.utils.get_original_cwd(), self.dir, self.name)

    def get_dataset(self):
        """Load a dataset from torchvision.datasets.

        Raises
        ------
        ValueError
            If the dataset is not found in torchvision.datasets.
        """
        if not hasattr(torchvision.datasets, self.name):
            raise ValueError(f"Dataset {self.name} not found in torchvision.datasets.")

        torchvision_dataset = getattr(torchvision.datasets, self.name)

        return torchvision_dataset(
            root=self.data_path,
            train=self.split == "train",
            download=True,
            transform=Sampler(self.transforms),
        )

    def get_dataloader(self, world_size=1):
        """Return a DataLoader for the dataset.

        Returns
        -------
        torch.utils.data.DataLoader
            DataLoader object for the dataset.
        """
        dataset = self.get_dataset()

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.sampler = torch.utils.data.distributed.DistributedSampler(dataset)

            if self.batch_size % world_size != 0:
                logging.warning(
                    f"Batch size ({self.batch_size}) is not divisible by world size "
                    f"({world_size}). Setting per-device batch size to "
                    f"{self.batch_size // world_size}."
                )
            per_device_batch_size = max(self.batch_size // world_size, 1)

        else:
            self.sampler = None
            per_device_batch_size = self.batch_size

        # Use all available CPUs if num_workers is set to -1.
        if self.num_workers == -1:
            if os.environ.get("SLURM_JOB_ID"):
                num_workers = int(os.environ.get("SLURM_JOB_CPUS_PER_NODE", 1))
            else:
                num_workers = os.cpu_count()
            logging.info(
                f"Using {num_workers} workers (maximum available) for data loading."
            )
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                num_workers = max(num_workers // world_size, 1)  # workers per GPU

        else:
            num_workers = self.num_workers

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=per_device_batch_size,
            num_workers=num_workers,
            pin_memory=True,
            sampler=self.sampler,
            shuffle=self.shuffle and sampler is None,
            drop_last=self.drop_last and sampler is None,
        )

        return loader

    @property
    def sampler(self):
        """Return the sampler for the dataset."""
        return self.sampler


@dataclass
class DataConfig:
    """Configuration for multiple datasets used for training the model.

    Parameters
    ----------
    train_on : str
        The dataset to train on.
    datasets : dict[str, DatasetConfig]
        A dictionary of dataset configurations.
    """

    train_on: str
    datasets: dict[str, DatasetConfig]

    def __init__(self, train_on, *args, **datasets):
        """Initialize DataConfig.

        Parameters
        ----------
        train_on : str
            The dataset to train on.
        datasets : dict
            A dictionary of dataset configurations.
        """
        assert len(args) == 0
        self.train_on = train_on
        self.datasets = {name: DatasetConfig(**d) for name, d in datasets.items()}

    def get_datasets(self):
        """Get datasets for training and validation.

        Returns
        -------
        dict
            A dictionary containing datasets.
        """
        return {name: d.get_dataset() for name, d in self.datasets.items()}

    def get_dataloaders(self, world_size=1):
        """Get dataloaders for the datasets.

        Returns
        -------
        dict
            A dictionary containing dataloaders.
        """
        return {
            name: d.get_dataloader(world_size=world_size)
            for name, d in self.datasets.items()
        }

    @property
    def train_dataset(self):
        """Return the batch size for training."""
        return self.datasets[self.train_on]

    def set_epoch_train_sampler(self, epoch):
        """Set the epoch for the training sampler."""
        if self.train_dataset.sampler is not None:
            self.train_dataset.sampler.set_epoch(epoch)


class Sampler:
    """Apply a list of transforms to an input and return all outputs."""

    def __init__(self, transforms: list):
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


def from_torchvision(data_path, dataset):
    """Load dataset features and labels from torchvision.

    Parameters
    ----------
    data_path : str
        Path to the dataset.
    dataset : torch.utils.data.Dataset
        The dataset class from torchvision.

    Returns
    -------
    tuple
        Tuple of features and labels.
    """
    dataset = dataset(
        root=data_path, train=True, download=True, transform=transforms.ToTensor()
    )
    features = torch.stack([dataset[i][0] for i in range(len(dataset))])
    labels = torch.tensor([dataset[i][1] for i in range(len(dataset))])
    return features, labels


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
        raise ValueError("dataset does not have `labels`")
    classes, class_inverse, class_counts = np.unique(
        labels, return_counts=True, return_inverse=True
    )

    logging.info(f"Subsampling : original class counts: {list(class_counts)}")

    if np.min(samples_or_freq) < 0:
        raise ValueError(
            "You can't have negative values in `samples_or_freq`, "
            f"got {samples_or_freq}."
        )
    elif np.sum(samples_or_freq) <= 1:
        target_class_counts = np.array(samples_or_freq) * len(dataset)
    elif np.sum(samples_or_freq) == len(dataset):
        freq = np.array(samples_or_freq) / np.sum(samples_or_freq)
        target_class_counts = freq * len(dataset)
        if (target_class_counts / class_counts).max() > 1:
            raise ValueError("specified more samples per class than available")
    else:
        raise ValueError(
            f"samples_or_freq needs to sum to <= 1 or len(dataset) ({len(dataset)}), "
            f"got {np.sum(samples_or_freq)}."
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


class CustomTorchvisionDataset(Dataset):
    """A custom dataset class for loading torchvision datasets.

    Parameters
    ----------
    root : str
        Path to the dataset.
    transform : callable, optional
        Transformation function to apply to the data. Default is None.
    """

    def __init__(self, root, transform=None):
        """Initialize the dataset with the given root path and transform."""
        self.transform = transform

        # Load the dataset from the .pt file
        data = torch.load(root)
        self.features = data["features"]
        self.labels = data["labels"]

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.features)

    def __getitem__(self, idx):
        """Get a sample from the dataset.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple
            The feature and label of the sample.
        """
        feature = self.features[idx]
        feature = to_pil_image(feature)

        label = self.labels[idx]

        if self.transform:
            feature = self.transform(feature)

        return feature, label
