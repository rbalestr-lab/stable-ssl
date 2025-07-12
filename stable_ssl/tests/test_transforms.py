import itertools
import time

import numpy as np
import stable_ssl as ossl
import stable_ssl.data.transforms as ot
import pytest
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2


def test_collator():
    assert ossl.data.Collator._test()


@pytest.mark.parametrize(
    "our_transform,true_transform",
    [
        (ot.GaussianBlur(3), v2.GaussianBlur(3)),
        (ot.RandomChannelPermutation(), v2.RandomChannelPermutation()),
        (ot.RandomHorizontalFlip(0.5), v2.RandomHorizontalFlip(0.5)),
        (ot.RandomGrayscale(0.5), v2.RandomGrayscale(0.5)),
        (ot.ColorJitter(0.8, 0.4, 0.4, 0.4), v2.ColorJitter(0.8, 0.4, 0.4, 0.4)),
        (ot.RandomResizedCrop((32, 32)), v2.RandomResizedCrop((32, 32))),
        (ot.RandomSolarize(0.5, 0.2), v2.RandomSolarize(0.5, 0.2)),
        (ot.RandomRotation(90), v2.RandomRotation(90)),
    ],
)
def test_controlled_transforms(our_transform, true_transform):
    transform = ot.Compose(
        ot.ControlledTransform(transform=our_transform, seed_offset=0), ot.ToImage()
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
        (ot.GaussianBlur(3), v2.GaussianBlur(3)),
        (ot.RandomChannelPermutation(), v2.RandomChannelPermutation()),
        (ot.RandomHorizontalFlip(0.5), v2.RandomHorizontalFlip(0.5)),
        (ot.RandomGrayscale(0.5), v2.RandomGrayscale(0.5)),
        (ot.RandomCrop(8), v2.RandomCrop(8)),
        (ot.ColorJitter(0.8, 0.4, 0.4, 0.4), v2.ColorJitter(0.8, 0.4, 0.4, 0.4)),
        (ot.RandomResizedCrop((32, 32)), v2.RandomResizedCrop((32, 32))),
        (ot.CenterCrop(16), v2.CenterCrop(16)),
        (ot.Resize(16), v2.Resize(16)),
        (ot.RandomSolarize(0.5, 0.2), v2.RandomSolarize(0.5, 0.2)),
        (ot.RGB(), v2.RGB()),
        (ot.RandomRotation(90), v2.RandomRotation(90)),
    ],
)
def test_transforms_batch(our_transform, true_transform):
    transform = ot.Compose(our_transform, ot.ToImage())
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
        not isinstance(our_transform, ot.RGB)
        and not isinstance(our_transform, ot.CenterCrop)
        and not isinstance(our_transform, ot.Resize)
    ):
        assert our_transform.__class__.__name__ in batch
    assert batch["label"].eq(y).all().item()
    assert torch.isclose(batch["image"], X, atol=1e-5).all().float().mean().item() == 1


@pytest.mark.parametrize(
    "our_transform,proba",
    itertools.product(
        [
            ot.GaussianBlur(3, sigma=(1, 2)),
            ot.RandomSolarize(128),
            ot.RandomGrayscale(),
            ot.ColorJitter(0.5, 0.5, 0.5, 0.5),
        ],
        [0, 0.1, 0.9, 1],
    ),
)
def test_proba(our_transform, proba):
    our_transform.p = proba
    transforms = ot.Compose(our_transform, ot.ToImage())
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
        (ot.GaussianBlur(3), v2.GaussianBlur(3), False),
        (ot.GaussianBlur(3), v2.GaussianBlur(3), True),
        (ot.RandomHorizontalFlip(), v2.RandomHorizontalFlip(), False),
        (ot.RandomHorizontalFlip(), v2.RandomHorizontalFlip(), True),
        (ot.RandomGrayscale(), v2.RandomGrayscale(), False),
        (ot.RandomGrayscale(), v2.RandomGrayscale(), True),
        (ot.RandomSolarize(128), v2.RandomSolarize(128), False),
        (ot.RandomSolarize(128), v2.RandomSolarize(128), True),
        (ot.RandomResizedCrop((32, 32)), v2.RandomResizedCrop((32, 32)), False),
        (ot.RandomResizedCrop((32, 32)), v2.RandomResizedCrop((32, 32)), True),
    ],
)
def test_timing(our_transform, true_transform, controlled):
    transforms = ot.Compose(our_transform, ot.ToImage())
    our_dataset = ossl.data.dataset.DictFormat(
        Subset(CIFAR10("~/data", download=True), range(256))
    )
    if controlled:
        transforms = ot.ControlledTransform(transform=transforms, seed_offset=0)
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
