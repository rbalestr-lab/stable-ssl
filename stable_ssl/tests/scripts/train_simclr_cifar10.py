#!/usr/bin/env python
"""Manual test script for SimCLR training with online probing.

This script is extracted from the integration tests to allow manual testing
with different datasets when the default dataset is not available.
"""

import lightning as pl
import torch
import torchmetrics
import torchvision
from transformers import AutoConfig, AutoModelForImageClassification

import stable_ssl as ssl
from stable_ssl.data import transforms
from stable_ssl.data.utils import Dataset

# without transform
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
train_transform = transforms.Compose(
    transforms.RGB(),
    transforms.RandomResizedCrop((32, 32)),  # CIFAR-10 is 32x32
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
    ),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=(5, 5), p=1.0),
    transforms.ToImage(mean=mean, std=std),
)

# Use torchvision CIFAR-10 wrapped in FromTorchDataset
cifar_train = torchvision.datasets.CIFAR10(
    root="/tmp/cifar10", train=True, download=True
)

# Create a custom wrapper that adds sample_idx


class IndexedDataset(Dataset):
    """Custom dataset wrapper that adds sample_idx to each sample."""

    def __init__(self, dataset, transform=None):
        super().__init__(transform)
        self.dataset = dataset

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        sample = {"image": image, "label": label, "sample_idx": idx}
        return self.process_sample(sample)

    def __len__(self):
        return len(self.dataset)


train_dataset = IndexedDataset(cifar_train, transform=train_transform)
train = torch.utils.data.DataLoader(
    dataset=train_dataset,
    sampler=ssl.data.sampler.RepeatedRandomSampler(train_dataset, n_views=2),
    batch_size=64,
    num_workers=20,
    drop_last=True,
)
val_transform = transforms.Compose(
    transforms.RGB(),
    transforms.Resize((32, 32)),
    transforms.CenterCrop((32, 32)),
    transforms.ToImage(mean=mean, std=std),
)

# Use torchvision CIFAR-10 for validation
cifar_val = torchvision.datasets.CIFAR10(
    root="/tmp/cifar10", train=False, download=True
)
val_dataset = IndexedDataset(cifar_val, transform=val_transform)
val = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=128,
    num_workers=10,
)
data = ssl.data.DataModule(train=train, val=val)


def forward(self, batch, stage):
    batch["embedding"] = self.backbone(batch["image"])["logits"]
    if self.training:
        proj = self.projector(batch["embedding"])
        views = ssl.data.fold_views(proj, batch["sample_idx"])
        batch["loss"] = self.simclr_loss(views[0], views[1])
    return batch


config = AutoConfig.from_pretrained("microsoft/resnet-18")
backbone = AutoModelForImageClassification.from_config(config)
projector = torch.nn.Linear(512, 128)
backbone.classifier[1] = torch.nn.Identity()
module = ssl.Module(
    backbone=backbone,
    projector=projector,
    forward=forward,
    simclr_loss=ssl.losses.NTXEntLoss(temperature=0.1),
)
linear_probe = ssl.callbacks.OnlineProbe(
    "linear_probe",
    module,
    "embedding",
    "label",
    probe=torch.nn.Linear(512, 10),
    loss_fn=torch.nn.CrossEntropyLoss(),
    metrics={
        "top1": torchmetrics.classification.MulticlassAccuracy(10),
        "top5": torchmetrics.classification.MulticlassAccuracy(10, top_k=5),
    },
)
knn_probe = ssl.callbacks.OnlineKNN(
    name="knn_probe",
    input="embedding",
    target="label",
    queue_length=20000,
    metrics={"accuracy": torchmetrics.classification.MulticlassAccuracy(10)},
    input_dim=512,
    k=10,
)

trainer = pl.Trainer(
    max_epochs=6,
    num_sanity_val_steps=0,  # Skip sanity check as queues need to be filled first
    callbacks=[linear_probe, knn_probe],
    precision="16-mixed",
    logger=False,
    enable_checkpointing=False,
)
manager = ssl.Manager(trainer=trainer, module=module, data=data)
manager()
