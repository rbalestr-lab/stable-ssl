"""Multi-layer probe for vision models."""

import argparse
from typing import Dict, List, Tuple

import lightning as pl
import torch
import torchmetrics
import torchvision
from lightning.pytorch.loggers import WandbLogger  # type: ignore
from torch import nn
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoModelForZeroShotImageClassification,
    AutoProcessor,
)

import stable_pretraining as spt
from stable_pretraining.data import transforms

# -----------------------------
# Model registry
# -----------------------------
MODEL_ZOO = {
    "DINOv2": {
        "processor_cls": AutoImageProcessor,
        "processor_name": "facebook/dinov2-base",
        "model_cls": AutoModel,
        "model_name": "facebook/dinov2-base",
    },
    "MetaCLIP": {
        "processor_cls": AutoProcessor,
        "processor_name": "facebook/metaclip-l14-400m",
        "model_cls": AutoModelForZeroShotImageClassification,
        "model_name": "facebook/metaclip-l14-400m",
    },
    "IJEPA-1k": {
        "processor_cls": AutoImageProcessor,
        "processor_name": "facebook/ijepa_vith14_1k",
        "model_cls": AutoModel,
        "model_name": "facebook/ijepa_vith14_1k",
    },
    "IJEPA-22k": {
        "processor_cls": AutoImageProcessor,
        "processor_name": "facebook/ijepa_vith14_22k",
        "model_cls": AutoModel,
        "model_name": "facebook/ijepa_vith14_22k",
    },
}


# -----------------------------
# Utilities
# -----------------------------


def build_datasets(
    data_root: str = "./data",
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    cifar_train = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True
    )
    cifar_val = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True)

    train_dataset = spt.data.FromTorchDataset(
        cifar_train,
        names=["image", "label"],
        transform=transforms.ToImage(scale=False),
    )

    val_dataset = spt.data.FromTorchDataset(
        cifar_val,
        names=["image", "label"],
        transform=transforms.ToImage(scale=False),
    )

    return train_dataset, val_dataset


def build_dataloaders(
    train_dataset,
    val_dataset,
    batch_size: int = 256,
    num_workers: int = 6,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        sampler=spt.data.sampler.RepeatedRandomSampler(train_dataset),
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def load_backbone(model_name: str):
    spec = MODEL_ZOO[model_name]
    processor = spec["processor_cls"].from_pretrained(spec["processor_name"])  # type: ignore
    model = spec["model_cls"].from_pretrained(
        spec["model_name"], output_hidden_states=True
    )  # type: ignore
    emb_dim = model.config.hidden_size
    return model, processor, emb_dim


# -----------------------------
# Lightning-compatible `spt.Module`
# -----------------------------


def build_module(model, processor, transformer_block_indices: List[int]) -> spt.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the forward used by `spt.Module`
    def forward(self, batch: Dict, stage: str):  # noqa: ARG001 (stage provided by spt)
        out: Dict[str, torch.Tensor] = {}

        # Preprocess & move to device
        images = processor(batch["image"], return_tensors="pt")
        images = {k: v.to(device=device, non_blocking=True) for k, v in images.items()}

        with torch.inference_mode():
            outputs = self.model(**images)
            hiddens = outputs[
                "hidden_states"
            ]  # tuple: [embeddings, block1, block2, ...]

        # Mean-pool tokens per layer -> (B, D)
        for i in transformer_block_indices:
            x = hiddens[1 + i].mean(dim=1)
            out[f"embedding_layer_{i}"] = x.detach()
        return out

    module = spt.Module(
        model=spt.backbone.EvalOnly(model),  # freeze eval-only backbone
        forward=forward,
        processor=processor,
        optim=None,  # probes have their own optimizers
    )
    return module


# -----------------------------
# Probes
# -----------------------------


def build_probes(emb_dim: int, num_classes: int, transformer_block_indices: List[int]):
    probes = []
    for i in transformer_block_indices:
        probes.append(
            spt.callbacks.OnlineProbe(
                target="label",
                name=f"linear_probe_block_{i}",
                input=f"embedding_layer_{i}",
                probe=nn.Linear(emb_dim, num_classes),
                loss_fn=nn.CrossEntropyLoss(),
                metrics={
                    "top1": torchmetrics.classification.MulticlassAccuracy(num_classes),
                    "top5": torchmetrics.classification.MulticlassAccuracy(
                        num_classes, top_k=5
                    ),
                },
                optimizer={"type": "SGD", "lr": 1e-3},
                scheduler={"type": "CosineAnnealingLR", "T_max": 100},
            )
        )
    return probes


# -----------------------------
# Main
# -----------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CIFAR‑10 transformer block probing with stable_pretraining"
    )
    p.add_argument("--model", choices=list(MODEL_ZOO.keys()), default="DINOv2")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument(
        "--use-wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    p.add_argument("--project", type=str, default="cifar10-multi-probe")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    # Data
    train_ds, val_ds = build_datasets()
    train_loader, val_loader = build_dataloaders(
        train_ds, val_ds, batch_size=args.batch_size, num_workers=args.num_workers
    )
    data = spt.data.DataModule(train=train_loader, val=val_loader)

    # Backbone & module
    model, processor, emb_dim = load_backbone(args.model)
    # Most ViT-like models have 12 blocks; adapt as needed
    transformer_block_indices = list(range(12))
    module = build_module(model, processor, transformer_block_indices)

    # Probes
    probes = build_probes(
        emb_dim=emb_dim,
        num_classes=10,
        transformer_block_indices=transformer_block_indices,
    )

    # Trainer
    precision = "16-mixed" if torch.cuda.is_available() else 32
    logger = None
    if args.use_wandb and WandbLogger is not None:
        logger = WandbLogger(project=args.project)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=probes,
        precision=precision,
        logger=logger,
    )

    # Run
    manager = spt.Manager(trainer=trainer, module=module, data=data)
    manager()


if __name__ == "__main__":
    main()
