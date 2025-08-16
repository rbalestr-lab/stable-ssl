import lightning as pl
import torch
import torchmetrics
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
            transforms.RandomResizedCrop((224, 224), scale=(0.08, 1.0)),
            transforms.ColorJitter(
                brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2, p=0.8
            ),
            transforms.PILGaussianBlur(p=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToImage(**ssl.data.static.ImageNet),
        ),
        transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((224, 224), scale=(0.08, 1.0)),
            transforms.ColorJitter(
                brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2, p=0.8
            ),
            transforms.PILGaussianBlur(p=0.5),
            transforms.RandomGrayscale(p=0.2),
            # transforms.RandomSolarize(threshold=0.5, p=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
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

data_dir = get_data_dir("imagenet100")

train_dataset = ssl.data.HFDataset(
    "clane9/imagenet-100",
    split="train",
    cache_dir=str(data_dir),
    transform=simclr_transform,
)
val_dataset = ssl.data.HFDataset(
    "clane9/imagenet-100",
    split="validation",
    cache_dir=str(data_dir),
    transform=val_transform,
)

batch_size = 1024  # per-GPU batch size
world_size = 4  # number of GPUs
total_batch_size = batch_size * world_size  # total batch size = 4096
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
        out["loss"] = self.simclr_loss(views[0], views[1])
    return out


backbone = ssl.backbone.from_torchvision(
    "resnet50",
    low_resolution=False,
)
backbone.fc = torch.nn.Identity()

projector = nn.Sequential(
    nn.Linear(2048, 2048),
    nn.ReLU(inplace=True),
    nn.Linear(2048, 128),
)

module = ssl.Module(
    backbone=backbone,
    projector=projector,
    forward=forward,
    simclr_loss=ssl.losses.NTXEntLoss(temperature=0.1),
    optim={
        "optimizer": {
            "type": "LARS",
            "lr": 0.3 * total_batch_size / 256,  # Using total batch size = 4096
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
    probe=torch.nn.Linear(2048, 100),
    loss_fn=torch.nn.CrossEntropyLoss(),
    metrics={
        "top1": torchmetrics.classification.MulticlassAccuracy(100),
        "top5": torchmetrics.classification.MulticlassAccuracy(100, top_k=5),
    },
)

# KNN probe removed to save memory for larger batch sizes
# knn_probe = ssl.callbacks.OnlineKNN(
#     name="knn_probe",
#     input="embedding",
#     target="label",
#     queue_length=20000,
#     metrics={"accuracy": torchmetrics.classification.MulticlassAccuracy(100)},
#     input_dim=2048,
#     k=20,
# )

wandb_logger = WandbLogger(
    entity="stable-ssl",
    project="imagenet100-simclr",
    name="simclr-resnet50-4gpu",
    log_model=False,
)

trainer = pl.Trainer(
    max_epochs=100,
    num_sanity_val_steps=0,
    callbacks=[linear_probe],  # Removed knn_probe to save memory
    precision="16-mixed",
    logger=wandb_logger,
    enable_checkpointing=False,
    strategy="ddp",
    devices=4,
    accelerator="gpu",
)

manager = ssl.Manager(trainer=trainer, module=module, data=data)
manager()
