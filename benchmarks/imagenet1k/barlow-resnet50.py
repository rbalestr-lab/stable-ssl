import lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from datasets import load_dataset
from lightning.pytorch.loggers import WandbLogger

import stable_ssl as ssl
from stable_ssl.data import transforms
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_data_dir

barlow_transform = transforms.MultiViewTransform(
    [
        transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((224, 224), scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0), p=1.0),
            transforms.ToImage(**ssl.data.static.ImageNet),
        ),
        transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((224, 224), scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0), p=0.1),
            transforms.RandomSolarize(threshold=0.5, p=0.2),
            transforms.ToImage(**ssl.data.static.ImageNet),
        ),
    ]
)

val_transform = transforms.Compose(
    transforms.RGB(),
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToImage(**ssl.data.static.ImageNet),
)

data_dir = get_data_dir("imagenet1k")
dataset = load_dataset("randall-lab/face-obfuscated-imagenet", cache_dir=str(data_dir))

train_dataset = ssl.data.FromHuggingFace(
    dataset["train"],
    names=["image", "label"],
    transform=barlow_transform,
)
val_dataset = ssl.data.FromHuggingFace(
    dataset["validation"],
    names=["image", "label"],
    transform=val_transform,
)

batch_size = 256
train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    sampler=ssl.data.sampler.RepeatedRandomSampler(train_dataset, n_views=2),
    batch_size=batch_size,
    num_workers=8,
    drop_last=True,
    persistent_workers=True,
)
val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=8,
    persistent_workers=True,
)

data = ssl.data.DataModule(train=train_dataloader, val=val_dataloader)


def forward(self, batch, stage):
    out = {}
    out["embedding"] = self.backbone(batch["image"])
    if self.training:
        proj = self.projector(out["embedding"])
        views = ssl.data.fold_views(proj, batch["sample_idx"])
        out["loss"] = self.barlow_loss(views[0], views[1])
    return out


backbone = ssl.backbone.from_torchvision(
    "resnet50",
    low_resolution=False,
)
backbone.fc = torch.nn.Identity()

projector = nn.Sequential(
    nn.Linear(2048, 8192),
    nn.BatchNorm1d(8192),
    nn.ReLU(inplace=True),
    nn.Linear(8192, 8192),
    nn.BatchNorm1d(8192),
    nn.ReLU(inplace=True),
    nn.Linear(8192, 8192),
)

module = ssl.Module(
    backbone=backbone,
    projector=projector,
    forward=forward,
    barlow_loss=ssl.losses.BarlowTwinsLoss(lambd=0.005),
    optim={
        "optimizer": {
            "type": "LARS",
            "lr": 0.2 * batch_size / 256,
            "weight_decay": 1e-6,
        },
        "scheduler": {
            "type": "LinearWarmupCosineAnnealing",
        },
        "interval": "epoch",
    },
)

linear_probe = ssl.callbacks.OnlineProbe(
    name="linear_probe",
    input="embedding",
    target="label",
    probe=torch.nn.Linear(2048, 1000),
    loss_fn=torch.nn.CrossEntropyLoss(),
    metrics={
        "top1": torchmetrics.classification.MulticlassAccuracy(1000),
        "top5": torchmetrics.classification.MulticlassAccuracy(1000, top_k=5),
    },
)

knn_probe = ssl.callbacks.OnlineKNN(
    name="knn_probe",
    input="embedding",
    target="label",
    queue_length=20000,
    metrics={"accuracy": torchmetrics.classification.MulticlassAccuracy(1000)},
    input_dim=2048,
    k=20,
)

wandb_logger = WandbLogger(
    entity="stable-ssl",
    project="imagenet1k-barlow",
    name="barlow-resnet50",
    log_model=False,
)

trainer = pl.Trainer(
    max_epochs=200,
    num_sanity_val_steps=0,
    callbacks=[knn_probe, linear_probe],
    precision="16-mixed",
    logger=wandb_logger,
    enable_checkpointing=False,
)

manager = ssl.Manager(trainer=trainer, module=module, data=data)
manager()
