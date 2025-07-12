import torch
from torch.utils.data import Dataset
from contextlib import contextmanager
from random import getstate, setstate
from random import seed as rseed
import numpy as np
from typing import Iterable


def set_seed(seeds):
    if hasattr(seeds[0], "__len__"):
        version, state, gauss = seeds[0]
        setstate((version, tuple(state), gauss))
    else:
        rseed(seeds[0])
    if hasattr(seeds[1], "__len__"):
        np.random.set_state(seeds[1])
    else:
        np.random.seed(seeds[1])
    if hasattr(seeds[2], "__len__"):
        torch.set_rng_state(seeds[2])
    else:
        torch.manual_seed(seeds[2])
    if len(seeds) == 4:
        if hasattr(seeds[3], "__len__"):
            torch.cuda.set_rng_state_all(seeds[3])
        else:
            torch.cuda.manual_seed(seeds[3])


@contextmanager
def random_seed(seed):
    seeds = [getstate(), np.random.get_state(), torch.get_rng_state()]
    # for now we don't use that functionality since it creates issues
    # with DataLoader and multiple processes...
    # RuntimeError: Cannot re-initialize CUDA in forked subprocess.
    # To use CUDA with multiprocessing, you must use the 'spawn' start method
    if False:  # torch.cuda.is_available():
        seeds.append(torch.cuda.get_rng_state_all())
    new_seeds = [int(seed)] * len(seeds)
    set_seed(new_seeds)
    yield
    set_seed(seeds)


class DictFormat(Dataset):

    def __init__(self, dataset: Iterable):
        """
        Arguments:
            dataset (Iterable): Path to the csv file with annotations.
        """
        self.dataset = dataset
        assert type(dataset) not in [
            DictFormat,
            AddTransform,
        ]
        # self.original = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image, label = self.dataset[idx]
        sample = dict(image=image, label=label, idx=idx)
        return sample


class AddTransform(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dataset: Iterable, transform: callable):
        """
        Arguments:
            dataset (Iterable): Path to the csv file with annotations.
            transform (callable): Transform to be applied on a sample.
        """
        assert type(dataset) in [
            DictFormat,
            AddTransform,
        ]
        assert callable(transform)
        self.dataset = dataset
        self.transform = transform
        # self.original = dataset.original

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.dataset[idx]
        sample = self.transform(sample)
        return sample
