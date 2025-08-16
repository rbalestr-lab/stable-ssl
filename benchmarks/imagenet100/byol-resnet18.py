import lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from lightning.pytorch.loggers import WandbLogger

import stable_ssl as ssl
from stable_ssl.data import transforms
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_data_dir

byol_transform = transforms.MultiViewTransform(
    [
        transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((224, 224), scale=(0.08, 1.0)),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.PILGaussianBlur(p=1.0),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToImage(**ssl.data.static.ImageNet),
        ),
        transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((224, 224), scale=(0.08, 1.0)),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.PILGaussianBlur(p=0.1),
            transforms.RandomSolarize(threshold=0.5, p=0.2),
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
    transform=byol_transform,
)
val_dataset = ssl.data.HFDataset(
    "clane9/imagenet-100",
    split="validation",
    cache_dir=str(data_dir),
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
    if self.training:
        images = batch["image"]
        sample_idx = batch["sample_idx"]

        online_features = self.backbone.forward_student(images)
        online_proj = self.projector(online_features)
        online_pred = self.predictor(online_proj)

        with torch.no_grad():
            target_features = self.backbone.forward_teacher(images)
            target_proj = self.projector.forward_teacher(target_features)

        online_pred_views = ssl.data.fold_views(online_pred, sample_idx)
        target_proj_views = ssl.data.fold_views(target_proj, sample_idx)

        loss_1 = self.byol_loss(online_pred_views[0], target_proj_views[1])
        loss_2 = self.byol_loss(online_pred_views[1], target_proj_views[0])
        batch["loss"] = (loss_1 + loss_2) / 2

        batch["embedding"] = online_features.detach()
    else:
        batch["embedding"] = self.backbone.forward_student(batch["image"])

    return batch


backbone = ssl.backbone.from_torchvision("resnet18", low_resolution=False, weights=None)
backbone.fc = nn.Identity()

wrapped_backbone = ssl.TeacherStudentWrapper(
    backbone,
    warm_init=True,
    base_ema_coefficient=0.99,
    final_ema_coefficient=1.0,
)

projector = nn.Sequential(
    nn.Linear(512, 4096),
    nn.BatchNorm1d(4096),
    nn.ReLU(inplace=True),
    nn.Linear(4096, 256),
)
wrapped_projector = ssl.TeacherStudentWrapper(
    projector,
    warm_init=True,
    base_ema_coefficient=0.99,
    final_ema_coefficient=1.0,
)

predictor = nn.Sequential(
    nn.Linear(256, 8192),
    nn.BatchNorm1d(8192),
    nn.ReLU(inplace=True),
    nn.Linear(8192, 256),
)

module = ssl.Module(
    backbone=wrapped_backbone,
    projector=wrapped_projector,
    predictor=predictor,
    forward=forward,
    byol_loss=ssl.losses.BYOLLoss(),
    optim={
        "optimizer": {
            "type": "LARS",
            "lr": 0.5 * batch_size / 256,
            "weight_decay": 1e-6,
            "clip_lr": True,
            "eta": 0.02,
            "exclude_bias_n_norm": True,
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
    probe=nn.Linear(512, 100),
    loss_fn=nn.CrossEntropyLoss(),
    metrics={
        "top1": torchmetrics.classification.MulticlassAccuracy(100),
        "top5": torchmetrics.classification.MulticlassAccuracy(100, top_k=5),
    },
)

knn_probe = ssl.callbacks.OnlineKNN(
    name="knn_probe",
    input="embedding",
    target="label",
    queue_length=20000,
    metrics={"accuracy": torchmetrics.classification.MulticlassAccuracy(100)},
    input_dim=512,
    k=20,
)

wandb_logger = WandbLogger(
    entity="stable-ssl",
    project="imagenet100-byol",
    name="byol-resnet18",
    log_model=False,
)

trainer = pl.Trainer(
    max_epochs=200,
    num_sanity_val_steps=0,
    callbacks=[linear_probe, knn_probe],
    precision="16-mixed",
    logger=wandb_logger,
)

manager = ssl.Manager(trainer=trainer, module=module, data=data)
manager()
