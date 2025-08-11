import lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
import torchvision
from lightning.pytorch.loggers import WandbLogger
from timm.models.vision_transformer import VisionTransformer
from torch import nn

import stable_ssl as ssl
from stable_ssl.backbone.utils import TeacherStudentModule
from stable_ssl.data import transforms
from stable_ssl.data.utils import Dataset

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
height, width, patch_size = 32, 32, 4
crop_height, crop_width = 32, 32  #   # CIFAR-10 is 32x32, but on INET, IJEPA uses 224
# We precompute these so the predictor can make sinusoidal posembeds
num_patches = (height // patch_size) * (width // patch_size)
patch_channel_dim = 3 * patch_size * patch_size

# Based on the in1k_vith14_ep300.yaml config in the ijepa repository
mask_transform_kwargs = dict(
    patch_size=patch_size,
    context_scale=(0.85, 1.0),
    context_aspect_ratio=(1.0, 1.0),
    target_scales=((0.15, 0.2),) * 4,
    target_aspect_ratios=((0.75, 1.5),) * 4,
    min_keep=20,
)


train_transform = transforms.Compose(
    transforms.RGB(),
    transforms.RandomResizedCrop((crop_height, crop_width), scale=(0.3, 1.0)),
    transforms.ContextTargetsMultiBlockMask(**mask_transform_kwargs),
    transforms.ToImage(mean=mean, std=std),
)
# Don't mask during validation
val_transform = transforms.Compose(
    transforms.RGB(),
    transforms.Resize((height, width)),
    transforms.CenterCrop((height, width)),
    transforms.ToImage(mean=mean, std=std),
)

# Use torchvision CIFAR-10 wrapped in FromTorchDataset
cifar_train = torchvision.datasets.CIFAR10(
    root="/tmp/cifar10", train=True, download=True
)
cifar_val = torchvision.datasets.CIFAR10(
    root="/tmp/cifar10", train=False, download=True
)


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


def standardize_masks(batch: list[dict]):
    context_indices = [item.pop("mask_context") for item in batch]
    target_indices = [item.pop("masks_target") for item in batch]
    batch = torch.utils.data.default_collate(batch)
    """
    [c.unique().shape for c in collated_masks_pred]
    [torch.Size([194]), torch.Size([168]), torch.Size([190]), torch.Size([158])]
    """
    min_keep_enc = min(len(ctx) for ctx in context_indices)
    min_keep_pred = min(
        len(block) for multiblock in target_indices for block in multiblock
    )

    context_batch = [ctx[:min_keep_enc] for ctx in context_indices]
    target_batch = [
        [tgt[:min_keep_pred] for tgt in multiblock] for multiblock in target_indices
    ]

    collated_masks_context = torch.utils.data.default_collate(context_batch)
    collated_masks_target = torch.utils.data.default_collate(target_batch)

    batch["mask_context"] = collated_masks_context
    batch["masks_target"] = collated_masks_target
    return batch


train_dataset = IndexedDataset(cifar_train, transform=train_transform)
# IJEPA does not use multi-view sampling like SimCLR etc. because it processes
# single views and handles masking at the model level
train = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True,  # Regular shuffling, no RepeatedRandomSampler
    num_workers=0,
    drop_last=True,
    collate_fn=standardize_masks,
)

val_dataset = IndexedDataset(cifar_val, transform=val_transform)
val = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=128,
    num_workers=0,
    shuffle=True,
)


data = ssl.data.DataModule(train=train, val=val)


# TODO sinusoidal posembed. for now this is just a dummy fn
def pos_embed(patches: torch.Tensor, device: torch.device, TMP_DIM=768) -> torch.Tensor:
    return torch.zeros(patches.shape[0], patches.shape[1], TMP_DIM, device=device)


def apply_masks(x: torch.Tensor, *masks: torch.Tensor) -> torch.Tensor:
    B, N, D = x.shape
    M = len(masks)
    idx = torch.stack(
        [m.to(x.device, dtype=torch.long) for m in masks], dim=1
    )  # [B, M, K]
    x_exp = x.unsqueeze(1).expand(-1, M, -1, -1)  # [B, M, N, D]
    out = x_exp.gather(2, idx.unsqueeze(-1).expand(-1, -1, -1, D))  # [B, M, K, D]
    return out.reshape(B * M, idx.size(-1), D)  # [B*M, K, D]


class IJEPA_Encoder(VisionTransformer):
    # TODO
    def __init__(self, *args, **kwargs):
        self.weight_init = kwargs.get("weight_init", "")
        self.fix_init = kwargs.get("fix_init", False)
        ijepa_in_dim = kwargs.pop("ijepa_in_dim")
        super().__init__(*args, **kwargs)

        self.ijepa_patch_project = nn.Linear(ijepa_in_dim, self.embed_dim)

        if self.weight_init != "skip":
            self.init_weights(self.weight_init)
        if self.fix_init:
            self.fix_init_weight()

    def patchify(self, image: torch.Tensor) -> torch.Tensor:
        """Convert image tensor into patches.

        Args:
            image: Tensor of shape [B, C, H, W]

        Returns:
            patches: Tensor of shape [B, N, P*P*C] where:
                N = number of patches (H/P * W/P)
                P = patch size
        """
        B, C, H, W = image.shape
        P = patch_size

        # Unfold into patches
        patches = image.unfold(2, P, P).unfold(3, P, P)

        # Reshape to [B, num_patches, patch_dim]
        patches = patches.permute(0, 2, 3, 1, 4, 5)
        num_patch_h, num_patch_w = patches.shape[1], patches.shape[2]
        patches = patches.reshape(B, num_patch_h * num_patch_w, P * P * C)

        return patches

    def project_patches(self, patches: torch.Tensor) -> torch.Tensor:
        # assume they are already reshaped to patches
        return self.ijepa_patch_project(patches)

    def encode_patches(
        self, patches: torch.Tensor, with_layernorm: bool = True
    ) -> torch.Tensor:
        x = self.blocks(patches)
        if with_layernorm:
            x = F.layer_norm(x, (x.size(-1),))
        return x

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        patches = self.patchify(image)
        patches = self.project_patches(patches)
        patches = patches + pos_embed(patches)
        return self.encode_patches(patches)


class IJEPA_Predictor(VisionTransformer):
    def __init__(self, *args, **kwargs):
        self.weight_init = kwargs.get("weight_init", "")
        self.fix_init = kwargs.get("fix_init", False)
        self.predictor_num_patches = kwargs.pop("predictor_num_patches")
        self.ijepa_encoder_dim = kwargs.pop("ijepa_encoder_dim")
        # TODO Fix device somehow and embeddim
        self.predictor_pos_embed = pos_embed(
            torch.zeros(1, self.predictor_num_patches, kwargs["embed_dim"]),
            device=torch.device("cuda"),
            TMP_DIM=kwargs["embed_dim"],
        )
        super().__init__(*args, **kwargs)
        self.predictor_pos_embed = nn.Parameter(
            self.predictor_pos_embed, requires_grad=False
        )
        self.predictor_inproj = nn.Linear(self.ijepa_encoder_dim, self.embed_dim)
        self.predictor_outproj = nn.Linear(self.embed_dim, self.ijepa_encoder_dim)
        self.predictor_norm = nn.LayerNorm(self.embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        if self.weight_init != "skip":
            self.init_weights(self.weight_init)
        if self.fix_init:
            self.fix_init_weight()

    def predict_targets(
        self, context_patches: torch.Tensor, masks_target: list[torch.Tensor]
    ) -> torch.Tensor:
        # NOTE context_patches already positionally embedded
        B, *_ = context_patches.shape
        M = len(masks_target)

        ctx: torch.Tensor = self.predictor_inproj(context_patches)  # [B, N_ctx, Dp]

        # target position embeddings (stacked per mask): [B*M, K_tgt, D]
        pos_all = self.predictor_pos_embed.expand(B, -1, -1)
        tgt_pos = apply_masks(pos_all, *masks_target)

        # repeat context across M target blocks: [B*M, N_ctx, D]. this means that
        # the predictor predicts each target block independently, and not their union,
        # as the ijepa repo does
        ctx = ctx.repeat_interleave(M, dim=0)

        # mask tokens placed at target positions
        N_tgt = tgt_pos.size(1)
        pred_tokens = self.mask_token.expand(B * M, N_tgt, -1) + tgt_pos

        # each target block now gets predicted: [B*M, N_ctx+N_tgt, D]
        x = torch.cat([ctx, pred_tokens], dim=1)
        x = self.blocks(x)
        x = self.predictor_norm(x)

        pred = x[:, -N_tgt:]
        pred = self.predictor_outproj(pred)
        return pred


encoder_kwargs = dict(
    patch_size=patch_size,
    embed_dim=768,
    depth=12,
    num_heads=12,
    qkv_bias=False,
    ijepa_in_dim=patch_channel_dim,
)
predictor_kwargs = dict(
    patch_size=patch_size,
    embed_dim=384,
    depth=6,
    num_heads=6,
    qkv_bias=False,
    ijepa_encoder_dim=768,
    predictor_num_patches=num_patches,
)


def forward(self: ssl.Module, batch, stage):
    out = {}
    target_encoder: IJEPA_Encoder = (
        self.target_encoder.teacher
    )  # NOTE Would this break anything?
    context_encoder: IJEPA_Encoder = self.context_encoder
    predictor: IJEPA_Predictor = self.predictor
    ijepa_loss: nn.Module = self.ijepa_loss

    image_patches = target_encoder.patchify(batch["image"])
    pos_embedding = pos_embed(image_patches, device=batch["image"].device)
    target_patches = target_encoder.project_patches(image_patches) + pos_embedding
    out["embedding"] = target_encoder.encode_patches(
        target_patches, with_layernorm=True
    )

    if not self.training:
        return out

    mask_context, masks_target = batch["mask_context"], batch["masks_target"]
    # target encoder is applied on full patches, then masked
    out["target_patches"] = apply_masks(out["embedding"], *masks_target)
    # context encoder is applied on masked patches
    context_patches = apply_masks(image_patches, mask_context)
    context_patches = context_encoder.project_patches(context_patches)
    context_patches = context_patches + apply_masks(pos_embedding, mask_context)
    out["context_patches"] = context_encoder.encode_patches(
        context_patches, with_layernorm=False
    )
    out["predicted_patches"] = predictor.predict_targets(context_patches, masks_target)

    out["loss"] = ijepa_loss(out["predicted_patches"], out["target_patches"])
    return out


module = ssl.Module(
    context_encoder=(ctx := IJEPA_Encoder(**encoder_kwargs)),
    target_encoder=TeacherStudentModule(ctx),
    predictor=IJEPA_Predictor(**predictor_kwargs),
    forward=forward,
    ijepa_loss=F.smooth_l1_loss,
)

trainer = pl.Trainer(
    max_epochs=6,
    num_sanity_val_steps=0,
    precision="16-mixed",
    enable_checkpointing=False,
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

# Initialize W&B logger with explicit settings
wandb_logger = WandbLogger(
    project="ijepa-cifar10",
    entity="slightly-more-badass",  # Your W&B entity
    name="ijepa-cifar10-run",
    log_model=False,  # Set to True if you want to save model artifacts
    offline=False,  # Ensure online mode
)

trainer = pl.Trainer(
    max_epochs=6,
    num_sanity_val_steps=0,  # Skip sanity check as queues need to be filled first
    callbacks=[knn_probe],
    precision="16-mixed",
    logger=wandb_logger,
    enable_checkpointing=False,
)
manager = ssl.Manager(trainer=trainer, module=module, data=data)
manager()
