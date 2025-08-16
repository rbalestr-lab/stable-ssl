"""Test unified queue with SimCLR on CIFAR10."""

import lightning as pl
import torch
import torchmetrics
import torchvision
from lightning.pytorch.loggers import WandbLogger
from torch import nn

import stable_ssl as ssl
from stable_ssl.data import transforms
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_data_dir

simclr_transform = transforms.MultiViewTransform(
    [
        transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((32, 32), scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomSolarize(threshold=0.5, p=0.0),
            transforms.ToImage(**ssl.data.static.CIFAR10),
        ),
        transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((32, 32), scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomSolarize(threshold=0.5, p=0.2),
            transforms.ToImage(**ssl.data.static.CIFAR10),
        ),
    ]
)

val_transform = transforms.Compose(
    transforms.RGB(),
    transforms.Resize((32, 32)),
    transforms.ToImage(**ssl.data.static.CIFAR10),
)

data_dir = get_data_dir("cifar10")
cifar_train = torchvision.datasets.CIFAR10(
    root=str(data_dir), train=True, download=True
)
cifar_val = torchvision.datasets.CIFAR10(root=str(data_dir), train=False, download=True)

train_dataset = ssl.data.FromTorchDataset(
    cifar_train,
    names=["image", "label"],
    transform=simclr_transform,
)
val_dataset = ssl.data.FromTorchDataset(
    cifar_val,
    names=["image", "label"],
    transform=val_transform,
)

# Smaller batch size for testing
train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    sampler=ssl.data.sampler.RepeatedRandomSampler(train_dataset, n_views=2),
    batch_size=128,  # Reduced for testing
    num_workers=4,
    drop_last=True,
)
val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=128,
    num_workers=4,
)

data = ssl.data.DataModule(train=train_dataloader, val=val_dataloader)


def forward(self, batch, stage):
    out = {}
    out["embedding"] = self.backbone(batch["image"])
    if self.training:
        proj = self.projector(out["embedding"])
        views = ssl.data.fold_views(proj, batch["sample_idx"])
        out["loss"] = self.simclr_loss(views[0], views[1])
    return out


backbone = ssl.backbone.from_torchvision(
    "resnet18",
    low_resolution=True,
)
backbone.fc = torch.nn.Identity()

projector = nn.Sequential(
    nn.Linear(512, 2048),
    nn.BatchNorm1d(2048),
    nn.ReLU(inplace=True),
    nn.Linear(2048, 2048),
    nn.BatchNorm1d(2048),
    nn.ReLU(inplace=True),
    nn.Linear(2048, 256),
)

module = ssl.Module(
    backbone=backbone,
    projector=projector,
    forward=forward,
    simclr_loss=ssl.losses.NTXEntLoss(temperature=0.5),
    optim={
        "optimizer": {
            "type": "LARS",
            "lr": 2.5,  # Reduced for smaller batch size
            "weight_decay": 1e-6,
        },
        "scheduler": {
            "type": "LinearWarmupCosineAnnealing",
        },
        "interval": "epoch",
    },
)

# Create multiple KNN probes with different queue sizes to test unified queue
knn_probe_small = ssl.callbacks.OnlineKNN(
    name="knn_probe_small",
    input="embedding",
    target="label",
    queue_length=5000,  # Small queue
    metrics={"accuracy": torchmetrics.classification.MulticlassAccuracy(10)},
    input_dim=512,
    k=5,
)

knn_probe_medium = ssl.callbacks.OnlineKNN(
    name="knn_probe_medium",
    input="embedding",
    target="label",
    queue_length=10000,  # Medium queue
    metrics={"accuracy": torchmetrics.classification.MulticlassAccuracy(10)},
    input_dim=512,
    k=10,
)

knn_probe_large = ssl.callbacks.OnlineKNN(
    name="knn_probe_large",
    input="embedding",
    target="label",
    queue_length=20000,  # Large queue
    metrics={"accuracy": torchmetrics.classification.MulticlassAccuracy(10)},
    input_dim=512,
    k=20,
)

# Add RankMe callback
rankme_callback = ssl.callbacks.RankMe(
    name="rankme",
    target="embedding",
    queue_length=10000,  # Should share queue with knn_probe_medium
    target_shape=512,
)

# Linear probe for comparison
linear_probe = ssl.callbacks.OnlineProbe(
    name="linear_probe",
    input="embedding",
    target="label",
    probe=torch.nn.Linear(512, 10),
    loss_fn=torch.nn.CrossEntropyLoss(),
    metrics={
        "top1": torchmetrics.classification.MulticlassAccuracy(10),
        "top5": torchmetrics.classification.MulticlassAccuracy(10, top_k=5),
    },
)

wandb_logger = WandbLogger(
    entity="stable-ssl",
    project="test-unified-queue",
    log_model=False,
    mode="offline",  # Offline mode for testing
)

trainer = pl.Trainer(
    max_epochs=5,  # Just a few epochs for testing
    num_sanity_val_steps=0,
    callbacks=[
        knn_probe_small,
        knn_probe_medium,
        knn_probe_large,
        rankme_callback,
        linear_probe,
    ],
    precision="16-mixed",
    logger=wandb_logger,
    enable_checkpointing=False,
    val_check_interval=0.5,  # Validate twice per epoch for testing
)

# Print queue information before training
print("\n" + "=" * 60)
print("Testing Unified Queue Management")
print("=" * 60)
print("\nExpected behavior:")
print("- All KNN probes and RankMe should share underlying queues")
print("- Queue for 'embedding' should be size 20000 (max requested)")
print("- Queue for 'label' should also be size 20000")
print("- Each callback gets its requested amount of data")
print("=" * 60 + "\n")

manager = ssl.Manager(trainer=trainer, module=module, data=data)
manager()

print("\n" + "=" * 60)
print("Test completed successfully!")
print("=" * 60)
