#!/usr/bin/env python
"""BYOL training on CIFAR-10."""

import lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import torchvision
from lightning.pytorch.loggers import WandbLogger

import stable_ssl as ssl
from stable_ssl.data import transforms


# BYOL augmentations - stronger than SimCLR as BYOL benefits from aggressive augmentation
byol_transform = transforms.MultiViewTransform(
    [
        transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((32, 32), scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            # View 1: no solarization (default BYOL v1 behavior)
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
            # View 2: enable solarization; no gaussian blur for CIFAR-10
            transforms.RandomSolarize(threshold=0.5, p=0.2),
            transforms.ToImage(**ssl.data.static.CIFAR10),
        ),
    ]
)

# Dataset setup
cifar_train = torchvision.datasets.CIFAR10(
    root="/tmp/cifar10", train=True, download=True
)
train_dataset = ssl.data.FromTorchDataset(
    cifar_train, names=["image", "label"], transform=byol_transform, add_sample_idx=True
)
train = torch.utils.data.DataLoader(
    dataset=train_dataset,
    sampler=ssl.data.sampler.RepeatedRandomSampler(train_dataset, n_views=2),
    batch_size=512,  # BYOL typically uses larger batches
    num_workers=20,
    drop_last=True,
)

# Validation dataset
val_transform = transforms.Compose(
    transforms.RGB(),
    transforms.Resize((32, 32)),
    transforms.CenterCrop((32, 32)),
    transforms.ToImage(**ssl.data.static.CIFAR10),
)
cifar_val = torchvision.datasets.CIFAR10(
    root="/tmp/cifar10", train=False, download=True
)
val_dataset = ssl.data.FromTorchDataset(
    cifar_val, names=["image", "label"], transform=val_transform, add_sample_idx=True
)
val = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=256,
    num_workers=10,
)
data = ssl.data.DataModule(train=train, val=val)


def forward(self, batch, stage):
    """BYOL forward pass using TeacherStudentWrapper.

    The backbone is wrapped with TeacherStudentWrapper, giving us:
    - forward_student(): online network (with gradients)
    - forward_teacher(): target network (EMA, no gradients)
    """
    if self.training:
        # Get the two augmented views
        images = batch["image"]
        sample_idx = batch["sample_idx"]

        # Process through online (student) network
        online_features = self.backbone.forward_student(images)
        online_proj = self.projector(online_features)
        online_pred = self.predictor(online_proj)

        # Process through target (teacher) network - no gradients
        with torch.no_grad():
            target_features = self.backbone.forward_teacher(images)
            target_proj = self.projector_target(target_features)

        # Fold views for BYOL loss computation
        online_pred_views = ssl.data.fold_views(online_pred, sample_idx)
        target_proj_views = ssl.data.fold_views(target_proj, sample_idx)

        # BYOL loss: predict view 2 from view 1 and vice versa
        loss_1 = self.byol_loss(online_pred_views[0], target_proj_views[1])
        loss_2 = self.byol_loss(online_pred_views[1], target_proj_views[0])
        batch["loss"] = (loss_1 + loss_2) / 2

        # For evaluation callbacks
        batch["embedding"] = online_features.detach()
    else:
        # Use online network for evaluation
        batch["embedding"] = self.backbone.forward_student(batch["image"])

    return batch


# Model setup
backbone = ssl.backbone.from_torchvision("resnet18", low_resolution=True, weights=None)
backbone.fc = nn.Identity()

# Wrap backbone with TeacherStudentWrapper for EMA
wrapped_backbone = ssl.TeacherStudentWrapper(
    backbone,
    warm_init=True,
    base_ema_coefficient=0.996,  # BYOL typically starts with 0.996
    final_ema_coefficient=1.0,  # and increases to 1.0
)

# Projector (MLP) - used by both online and target networks
projector = nn.Sequential(
    nn.Linear(512, 1024),
    nn.BatchNorm1d(1024),
    nn.ReLU(inplace=True),
    nn.Linear(1024, 256),
)

# Target projector (separate copy for target network)
projector_target = nn.Sequential(
    nn.Linear(512, 1024),
    nn.BatchNorm1d(1024),
    nn.ReLU(inplace=True),
    nn.Linear(1024, 256),
)
# Initialize target projector to match online projector
projector_target.load_state_dict(projector.state_dict())
projector_target.requires_grad_(False)

# Predictor (MLP) - only for online network, key to BYOL
predictor = nn.Sequential(
    nn.Linear(256, 1024),
    nn.BatchNorm1d(1024),
    nn.ReLU(inplace=True),
    nn.Linear(1024, 256),
)

# Create module
module = ssl.Module(
    backbone=wrapped_backbone,
    projector=projector,
    projector_target=projector_target,
    predictor=predictor,
    forward=forward,
    byol_loss=ssl.losses.BYOLLoss(),
    optim={
        "optimizer": {
            "type": "AdamW",
            "lr": 1e-3,
            "weight_decay": 1e-5,
        },
        "scheduler": {
            "type": "CosineAnnealingLR",
            "T_max": 200,
        },
        "interval": "epoch",
    },
)

# Evaluation callbacks
linear_probe = ssl.callbacks.OnlineProbe(
    name="linear_probe",
    input="embedding",
    target="label",
    probe=nn.Linear(512, 10),
    loss_fn=nn.CrossEntropyLoss(),
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
    k=20,
)

# Initialize W&B logger
wandb_logger = WandbLogger(
    entity="stable-ssl",
    project="cifar10-byol",
    name="byol-resnet18",
    log_model=False,
)

# Trainer with TeacherStudentCallback (will be auto-added by Manager)
trainer = pl.Trainer(
    max_epochs=500,  # BYOL typically needs more epochs
    num_sanity_val_steps=0,
    callbacks=[
        linear_probe,
        knn_probe,
        # TeacherStudentCallback will be auto-added by Manager
    ],
    precision="16-mixed",
    logger=wandb_logger,
    enable_checkpointing=True,
    # gradient_clip_val=1.0,  # BYOL benefits from gradient clipping
)

# Manager will auto-detect TeacherStudentWrapper and add the callback
manager = ssl.Manager(trainer=trainer, module=module, data=data)
manager()

print("BYOL training completed!")
