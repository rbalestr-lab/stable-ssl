#!/usr/bin/env python
"""Manual test script for SimCLR training with multi-optimizer configuration.

This script mirrors the single-optimizer example but assigns different optimizers
and schedulers to `backbone` and `projector` via regex-based grouping to exercise
and validate the logic implemented in `stable_ssl.module.Module`.
"""

import lightning as pl
import torch
import torchvision

import stable_ssl as ssl
from stable_ssl.data import transforms
from stable_ssl.data.utils import Dataset

# ------------------------------
# Data
# ------------------------------
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

val_transform = transforms.Compose(
    transforms.RGB(),
    transforms.Resize((32, 32)),
    transforms.CenterCrop((32, 32)),
    transforms.ToImage(mean=mean, std=std),
)


# Use torchvision CIFAR-10 wrapped in an Indexed dataset that adds sample_idx
class IndexedDataset(Dataset):
    """Wrap a dataset to add `sample_idx` and apply optional transforms.

    This adapter returns dict samples with keys: `image`, `label`, and
    `sample_idx`, and delegates transform handling to
    `stable_ssl.data.utils.Dataset`.
    """

    def __init__(self, dataset, transform=None):
        super().__init__(transform)
        self.dataset = dataset

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        sample = {"image": image, "label": label, "sample_idx": idx}
        return self.process_sample(sample)

    def __len__(self):
        return len(self.dataset)


cifar_train = torchvision.datasets.CIFAR10(
    root="/tmp/cifar10", train=True, download=True
)
train_dataset = IndexedDataset(cifar_train, transform=train_transform)
train = torch.utils.data.DataLoader(
    dataset=train_dataset,
    sampler=ssl.data.sampler.RepeatedRandomSampler(train_dataset, n_views=2),
    batch_size=64,
    num_workers=8,
    drop_last=True,
)

cifar_val = torchvision.datasets.CIFAR10(
    root="/tmp/cifar10", train=False, download=True
)
val_dataset = IndexedDataset(cifar_val, transform=val_transform)
val = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=128,
    num_workers=4,
)

data = ssl.data.DataModule(train=train, val=val)


# ------------------------------
# Model
# ------------------------------
# A very small backbone to keep the demo quick; 512-dim penultimate features
backbone = torchvision.models.resnet18(weights=None, num_classes=10)
# Remove classifier, expose penultimate features as logits for simplicity
backbone.fc = torch.nn.Identity()
projector = torch.nn.Linear(512, 128)


def forward(self, batch, stage):
    state = {}
    feats = self.backbone(batch["image"])  # shape [B, 512]
    state["embedding"] = feats
    if self.training:
        proj = self.projector(feats)
        views = ssl.data.fold_views(proj, batch["sample_idx"])  # two views
        state["loss"] = self.simclr_loss(views[0], views[1])
    return state


module = ssl.Module(
    backbone=backbone,
    projector=projector,
    forward=forward,
    simclr_loss=ssl.losses.NTXEntLoss(temperature=0.2),
)

# ------------------------------
# Multi-optimizer configuration using regex patterns
# ------------------------------
# - Assign "backbone" params to AdamW with a cosine schedule, step every step
# - Assign "projector" params to SGD with StepLR, step every 2 steps (frequency=2)
module.optim = {
    "encoder_opt": {
        "modules": r"^backbone(\.|$)",
        "optimizer": {"type": "AdamW", "lr": 3e-4, "weight_decay": 1e-4},
        "scheduler": "CosineAnnealingLR",  # uses smart defaults (T_max from trainer)
        "interval": "step",
        "frequency": 1,
    },
    "head_opt": {
        "modules": r"^projector(\.|$)",
        "optimizer": {"type": "SGD", "lr": 1e-2, "momentum": 0.9},
        "scheduler": {"type": "StepLR", "step_size": 50, "gamma": 0.5},
        "interval": "step",
        "frequency": 2,
    },
}

# Optional: demonstrate probing helper listing modules by regex
matched = module.get_modules_by_regex(r"^(backbone|projector)(\.|$)")
print("Matched modules:")
for name, _ in matched[:20]:
    print(" -", name)


# ------------------------------
# Training
# ------------------------------
pl.seed_everything(42)
trainer = pl.Trainer(
    max_epochs=1,
    num_sanity_val_steps=0,
    precision="16-mixed",
    enable_checkpointing=False,
    log_every_n_steps=10,
)

manager = ssl.Manager(trainer=trainer, module=module, data=data)
if __name__ == "__main__":
    manager()
