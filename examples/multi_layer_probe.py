import lightning as pl
import torch
import torchmetrics
import torchvision
from lightning.pytorch.loggers import WandbLogger
from transformers import AutoModel

import stable_pretraining as spt
from stable_pretraining.data import transforms

# Load CIFAR-10 and wrap in dictionary format
cifar_train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
cifar_val = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)

train_dataset = spt.data.FromTorchDataset(
    cifar_train,
    names=["image", "label"],  # Convert tuple to dictionary
    transform=transforms.Compose(
        transforms.RGB(),
        transforms.Resize((224, 224)),
        transforms.ToImage(**spt.data.static.CIFAR10),
    ),
)

val_dataset = spt.data.FromTorchDataset(
    cifar_val,
    names=["image", "label"],
    transform=transforms.Compose(
        transforms.RGB(),
        transforms.Resize((224, 224)),
        transforms.ToImage(**spt.data.static.CIFAR10),
    ),
)

# Create dataloaders with view sampling for contrastive learning
train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    sampler=spt.data.sampler.RepeatedRandomSampler(train_dataset),
    batch_size=256,
    num_workers=8,
    drop_last=True,
)

val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=256,
    num_workers=10,
)

data = spt.data.DataModule(train=train_dataloader, val=val_dataloader)

# Transformer block number to probe
transformer_block_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# Define the forward function (replaces training_step in PyTorch Lightning)
def forward(self, batch, stage):
    out = {}
    embeddings = self.backbone(batch["image"])["hidden_states"]
    for i in transformer_block_indices:
        embedding = embeddings[1+i] # +1 as 0 is the embedding layer
        out[f"embedding_layer_{i}"] = embedding.mean(dim=1) # average pooling
    return out

# Init Hugging Face model
backbone = AutoModel.from_pretrained(
    "nateraw/vit-base-patch16-224-cifar10",
    output_hidden_states=True, # Enable output of hidden states
)

# Load torch checkpoint if needed
# ...

# Create the module with all components
module = spt.Module(
    backbone=spt.backbone.EvalOnly(backbone), # Freeze backbone
    forward=forward,
    optim=None,
)

probes = []
for i in transformer_block_indices:
    probes.append(
        spt.callbacks.OnlineProbe(
            target="label",
            name=f"linear_probe_block_{i}",
            input=f"embedding_layer_{i}",
            probe=torch.nn.Linear(768, 10),
            loss_fn=torch.nn.CrossEntropyLoss(),
            metrics={
                "top1": torchmetrics.classification.MulticlassAccuracy(10),
                "top5": torchmetrics.classification.MulticlassAccuracy(10, top_k=5),
            },
            optimizer={
                "type": "SGD", "lr": 1e-3,
            },
            scheduler={"type": "CosineAnnealingLR", "T_max": 100},
        )
    )

# Configure training
trainer = pl.Trainer(
    max_epochs=100,
    callbacks=probes,
    precision="16-mixed",
    logger=WandbLogger(project="cifar10-multi-probe"),
)

# Launch training
manager = spt.Manager(trainer=trainer, module=module, data=data)
manager()
