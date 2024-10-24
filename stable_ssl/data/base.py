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

from .sampler import Sampler
from .augmentations import TransformsConfig


@dataclass
class DatasetConfig:
    """
    Configuration for the data used for training the model.

    Parameters:
    -----------
    dir : str
        Path to the directory containing the training data.
        Default is "data".
    name : str
        Name of the dataset to use (e.g., "CIFAR10", "CIFAR100").
        Default is "CIFAR10".
    split : str
        Name of the dataset split to use (e.g., "train", "test").
        Default is "train".
    num_workers : int
        NUmber of process to use
    """

    dir: str = "data"
    name: str = "CIFAR10"
    split: str = "train"
    num_workers: int = 0
    batch_size: int = 32
    transforms: list[TransformsConfig] = None
    drop_last: bool = False
    shuffle: bool = False

    def __post_init__(self):
        if self.transforms is None:
            self.transforms = [TransformsConfig("None")]
        else:
            self.transforms = [
                TransformsConfig(name, t) for name, t in self.transforms.items()
            ]

    @property
    def num_classes(self):
        if self.name == "CIFAR10":
            return 10
        elif self.name == "CIFAR100":
            return 100

    @property
    def resolution(self):
        if self.name in ["CIFAR10", "CIFAR100"]:
            return 32

    @property
    def data_path(self):
        return os.path.join(hydra.utils.get_original_cwd(), self.dir, self.name)

    def get_dataset(self):
        """
        Load a dataset from torchvision.datasets.
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

    def get_dataloader(self):
        dataset = self.get_dataset()

        # FIXME: handle those cases
        # if self.config.hardware.world_size > 1:
        #     sampler = torch.utils.data.distributed.DistributedSampler(
        #         dataset, shuffle=not train, drop_last=train
        #     )
        #     assert self.config.optim.batch_size % self.config.hardware.world_size == 0
        # else:
        #     sampler = None

        # per_device_batch_size = (
        #     self.config.optim.batch_size // self.config.hardware.world_size
        # )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=None,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
        )

        return loader


@dataclass
class DataConfig:
    """
    Configuration for the data used for training the model.

    Parameters:
    -----------
    data_dir : str
        Path to the directory containing the training data.
        Default is "data".
    dataset : str
        Name of the dataset to use (e.g., "CIFAR10", "CIFAR100").
        Default is "CIFAR10".
    resolution : int
        Resolution of the images in the dataset. Default is 32.
    num_classes : int
        Number of classes in the dataset. Default is 10.
    coeff_imbalance : float
        Coefficient for creating an imbalanced version of the dataset.
        Default is None.
    """

    train_on: str
    datasets: dict[str, DatasetConfig]

    def __init__(self, train_on, *args, **datasets):
        assert len(args) == 0
        self.train_on = train_on
        self.datasets = {name: DatasetConfig(**d) for name, d in datasets.items()}

    def get_datasets(self):
        return {name: d.get_dataset() for name, d in self.datasets.items()}

    def get_dataloaders(self):
        return {name: d.get_dataloader() for name, d in self.datasets.items()}


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
    dataset = dataset(
        root=data_path, train=True, download=True, transform=transforms.ToTensor()
    )
    features = torch.stack([dataset[i][0] for i in range(len(dataset))])
    labels = torch.tensor([dataset[i][1] for i in range(len(dataset))])
    return features, labels


def resample_classes(dataset, samples_or_freq, random_seed=None):
    """
    Create an exponential class imbalance.
    Args:
        dataset (torch.utils.data.Dataset): The input data, shape (N, ...).
        samples_or_freq (iterable): Number of samples or frequency
            for each class in the new dataset.
        random_seed (int): The random seed.
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
            "You can't have negative values in `sampels_or_freq`, "
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
            f"samples_or_freq needs to sum to <= 1 or len(datset) ({len(dataset)}), "
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
    def __init__(self, root, transform=None):
        """
        root: Directory where the dataset is stored
        transform: A function/transform to apply to the data
        """
        self.transform = transform

        # Load the dataset from the .pt file
        data = torch.load(root)
        self.features = data["features"]
        self.labels = data["labels"]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        feature = to_pil_image(feature)

        label = self.labels[idx]

        if self.transform:
            feature = self.transform(feature)

        return feature, label


# if __name__ == "__main__":
#     data_path = "../runs/data/CIFAR10/"

#     load_dataset("CIFAR10", data_path, train=True, coeff_imbalance=1.1)
