import itertools
import time

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2

import stable_ssl as ossl
import stable_ssl.data.transforms as transforms


def test_collator():
    assert ossl.data.Collator._test()


@pytest.mark.parametrize(
    "our_transform,true_transform",
    [
        (transforms.GaussianBlur(3), v2.GaussianBlur(3)),
        (transforms.RandomChannelPermutation(), v2.RandomChannelPermutation()),
        (transforms.RandomHorizontalFlip(0.5), v2.RandomHorizontalFlip(0.5)),
        (transforms.RandomGrayscale(0.5), v2.RandomGrayscale(0.5)),
        (
            transforms.ColorJitter(0.8, 0.4, 0.4, 0.4),
            v2.ColorJitter(0.8, 0.4, 0.4, 0.4),
        ),
        (transforms.RandomResizedCrop((32, 32)), v2.RandomResizedCrop((32, 32))),
        (transforms.RandomSolarize(0.5, 0.2), v2.RandomSolarize(0.5, 0.2)),
        (transforms.RandomRotation(90), v2.RandomRotation(90)),
    ],
)
def test_controlled_transforms(our_transform, true_transform):
    transform = transforms.Compose(
        transforms.ControlledTransform(transform=our_transform, seed_offset=0),
        transforms.ToImage(),
    )
    our_dataset = ossl.data.dataset.DictFormat(CIFAR10("~/data", download=True))
    our_dataset = ossl.data.dataset.AddTransform(our_dataset, transform)
    t = v2.Compose(
        [true_transform, v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
    )
    true_dataset = CIFAR10("~/data", transform=t)
    for _ in range(3):
        for i in range(10):
            ours = our_dataset[i]
            torch.manual_seed(i)
            truth = true_dataset[i]
            torch.manual_seed(0)
            assert ours["label"] == truth[1]
            assert torch.isclose(ours["image"], truth[0], atol=1e-5).all().item()


@pytest.mark.parametrize(
    "our_transform,true_transform",
    [
        (transforms.GaussianBlur(3), v2.GaussianBlur(3)),
        (transforms.RandomChannelPermutation(), v2.RandomChannelPermutation()),
        (transforms.RandomHorizontalFlip(0.5), v2.RandomHorizontalFlip(0.5)),
        (transforms.RandomGrayscale(0.5), v2.RandomGrayscale(0.5)),
        (transforms.RandomCrop(8), v2.RandomCrop(8)),
        (
            transforms.ColorJitter(0.8, 0.4, 0.4, 0.4),
            v2.ColorJitter(0.8, 0.4, 0.4, 0.4),
        ),
        (transforms.RandomResizedCrop((32, 32)), v2.RandomResizedCrop((32, 32))),
        (transforms.CenterCrop(16), v2.CenterCrop(16)),
        (transforms.Resize(16), v2.Resize(16)),
        (transforms.RandomSolarize(0.5, 0.2), v2.RandomSolarize(0.5, 0.2)),
        (transforms.RGB(), v2.RGB()),
        (transforms.RandomRotation(90), v2.RandomRotation(90)),
    ],
)
def test_transforms_batch(our_transform, true_transform):
    transform = transforms.Compose(our_transform, transforms.ToImage())
    ours = ossl.data.dataset.DictFormat(CIFAR10("~/data", download=True))
    ours = ossl.data.dataset.AddTransform(ours, transform)
    ours = DataLoader(ours, batch_size=64, shuffle=False)
    transform = v2.Compose(
        [true_transform, v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
    )
    truth = CIFAR10("~/data", transform=transform)
    truth = DataLoader(truth, batch_size=64, shuffle=False)

    torch.manual_seed(0)
    for X, y in truth:
        break
    torch.manual_seed(0)
    for batch in ours:
        break
    assert "image" in batch
    # THE RGB transform is the exception of not adding an entry
    if (
        not isinstance(our_transform, transforms.RGB)
        and not isinstance(our_transform, transforms.CenterCrop)
        and not isinstance(our_transform, transforms.Resize)
    ):
        assert our_transform.__class__.__name__ in batch
    assert batch["label"].eq(y).all().item()
    assert torch.isclose(batch["image"], X, atol=1e-5).all().float().mean().item() == 1


@pytest.mark.parametrize(
    "our_transform,proba",
    itertools.product(
        [
            transforms.GaussianBlur(3, sigma=(1, 2)),
            transforms.RandomSolarize(128),
            transforms.RandomGrayscale(),
            transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
        ],
        [0, 0.1, 0.9, 1],
    ),
)
def test_proba(our_transform, proba):
    our_transform.p = proba
    transforms = transforms.Compose(our_transform, transforms.ToImage())
    our_dataset = ossl.data.dataset.DictFormat(
        Subset(CIFAR10("~/data", download=True), range(2000))
    )
    our_dataset = ossl.data.dataset.AddTransform(our_dataset, transform=transforms)
    t = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    true_dataset = CIFAR10("~/data", transform=t)
    identity = []
    for _ in range(200):
        identity.append(
            torch.isclose(our_dataset[0]["image"], true_dataset[0][0])
            .all()
            .float()
            .item()
        )
    assert np.abs(np.mean(identity) - (1 - proba)) < 0.06


@pytest.mark.parametrize(
    "our_transform,true_transform,controlled",
    [
        (transforms.GaussianBlur(3), v2.GaussianBlur(3), False),
        (transforms.GaussianBlur(3), v2.GaussianBlur(3), True),
        (transforms.RandomHorizontalFlip(), v2.RandomHorizontalFlip(), False),
        (transforms.RandomHorizontalFlip(), v2.RandomHorizontalFlip(), True),
        (transforms.RandomGrayscale(), v2.RandomGrayscale(), False),
        (transforms.RandomGrayscale(), v2.RandomGrayscale(), True),
        (transforms.RandomSolarize(128), v2.RandomSolarize(128), False),
        (transforms.RandomSolarize(128), v2.RandomSolarize(128), True),
        (transforms.RandomResizedCrop((32, 32)), v2.RandomResizedCrop((32, 32)), False),
        (transforms.RandomResizedCrop((32, 32)), v2.RandomResizedCrop((32, 32)), True),
    ],
)
def test_timing(our_transform, true_transform, controlled):
    transforms = transforms.Compose(our_transform, transforms.ToImage())
    our_dataset = ossl.data.dataset.DictFormat(
        Subset(CIFAR10("~/data", download=True), range(256))
    )
    if controlled:
        transforms = transforms.ControlledTransform(transform=transforms, seed_offset=0)
        our_dataset = ossl.data.dataset.AddTransform(our_dataset, transform=transforms)
    else:
        our_dataset = ossl.data.dataset.AddTransform(
            our_dataset,
            transform=transforms,
        )
    our_dataset = DataLoader(our_dataset, batch_size=8)
    t = v2.Compose(
        [true_transform, v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
    )
    true_dataset = Subset(CIFAR10("~/data", transform=t), range(256))
    true_dataset = DataLoader(true_dataset, batch_size=8)
    t1 = time.time()
    for X, y in true_dataset:
        continue
    t1 = time.time() - t1
    print("Time for Torchvision is", t1)
    t2 = time.time()
    for batch in our_dataset:
        continue
    t2 = time.time() - t2
    print(f"Time for ours (controlled={controlled}) is", t2)
    if controlled:
        assert t2 < 10 * t1
    else:
        assert t2 < 1.3 * t1
