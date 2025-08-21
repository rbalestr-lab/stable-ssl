import  torch
import  math
import  lightning as pl
import  torch.nn.functional as F
import  torchmetrics
from    lightning.pytorch.loggers import WandbLogger
from    timm.models.vision_transformer import VisionTransformer
from    torch import nn
from    functools import partial

import  stable_ssl as ssl
from    stable_ssl.backbone.utils import TeacherStudentWrapper
from    stable_ssl.data import transforms
from    stable_ssl.utils.pos_embed import get_2d_sincos_pos_embed
from    lightning.pytorch.strategies import DDPStrategy
from    stable_ssl.callbacks.teacher_student import TeacherStudentCallback

train_batch_size = 128
val_batch_size   = 128
num_workers      = 32
num_classes      = 1000
num_epochs       = 300
lr_warmup_epochs = 60
start_lr         = 0.0002
lr               = 0.001
final_lr         = 1.0e-6
# max_grad_norm    = 5.0
max_grad_norm    = None
ema              = (0.996, 1.0)


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
height, width, patch_size = 256, 256, 14
crop_height, crop_width = 224, 224  # CIFAR-10 is 32x32, but on INET, IJEPA uses 224
# We precompute these so the predictor can make sinusoidal posembeds
num_patches = (crop_height // patch_size) * (crop_width // patch_size)
patch_channel_dim = 3 * patch_size * patch_size
encoder_embed_dim = 768
predictor_embed_dim = 384

# Based on the in1k_vith14_ep300.yaml config in the ijepa repository
mask_transform_kwargs = dict(
    patch_size=patch_size,
    context_scale=(0.85, 1.0),
    context_aspect_ratio=(1.0, 1.0),
    target_scales=((0.15, 0.2),) * 4,
    target_aspect_ratios=((0.75, 1.5),) * 4,
    min_keep=10,
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
    transforms.CenterCrop((crop_height, crop_width)),
    transforms.ToImage(mean=mean, std=std),
)


inet1k_train = ssl.data.HFDataset(
    path="ILSVRC/imagenet-1k",
    split="train",
    transform=train_transform,
)

inet1k_val = ssl.data.HFDataset(
    path="ILSVRC/imagenet-1k",
    split="validation",
    transform=val_transform,
)


def standardize_masks(batch: list[dict]):
    context_indices = [item.pop("mask_context") for item in batch]
    target_indices = [item.pop("masks_target") for item in batch]
    batch = torch.utils.data.default_collate(batch)

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


# IJEPA does not use multi-view sampling like SimCLR etc. because it processes
# single views and handles masking at the model level
train = torch.utils.data.DataLoader(
    dataset=inet1k_train,
    batch_size=train_batch_size,
    shuffle=True,  # Regular shuffling, no RepeatedRandomSampler
    num_workers=num_workers,
    drop_last=True,
    collate_fn=standardize_masks,
    pin_memory=True,
    persistent_workers=True,
)

val = torch.utils.data.DataLoader(
    dataset=inet1k_val,
    batch_size=val_batch_size,
    num_workers=num_workers,
    shuffle=False,
)


data = ssl.data.DataModule(train=train, val=val)


def pos_embed(patches: torch.Tensor) -> torch.Tensor:
    return (
        torch.from_numpy(
            get_2d_sincos_pos_embed(patches.shape[-1], int(math.sqrt(patches.shape[1])))
        )
        .to(patches.device)
        .float()
        .repeat(patches.shape[0], 1, 1)
    )


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
    """IJEPA encoder.

    Args:
        ijepa_in_dim: Input dimension of the encoder, which is the patch dimension after re-arranging the image.
    """

    def __init__(self, *args, **kwargs):
        self.weight_init = kwargs.get("weight_init", "")
        self.fix_init = kwargs.get("fix_init", True)
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
    """IJEPA predictor, handles the logic of conditioning the predictor based on the context and target masks.

    Args:
        predictor_num_patches: Number of patches in the predictor. This is typically equal to the number of patches in the context/target encoder.
        ijepa_encoder_dim: Dimension of the IJEPA context/target encoder. This is used to up/down project the latents.
    """

    def __init__(self, *args, **kwargs):
        self.weight_init = kwargs.get("weight_init", "")
        self.fix_init = kwargs.get("fix_init", True)
        self.predictor_num_patches = kwargs.pop("predictor_num_patches")
        self.ijepa_encoder_dim = kwargs.pop("ijepa_encoder_dim")
        self.predictor_pos_embed = pos_embed(
            torch.zeros(1, self.predictor_num_patches, kwargs["embed_dim"])
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

    def project_context(self, context_patches: torch.Tensor) -> torch.Tensor:
        return self.predictor_inproj(context_patches)

    def predict_targets(
        self, context_patches: torch.Tensor, masks_target: list[torch.Tensor]
    ) -> torch.Tensor:
        # NOTE context_patches already positionally embedded
        B, *_ = context_patches.shape
        M = len(masks_target)

        # NOTE: These are already projected -> posembedded 
        ctx: torch.Tensor = context_patches

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
    embed_dim=encoder_embed_dim,
    depth=12,
    num_heads=12,
    qkv_bias=True,
    ijepa_in_dim=patch_channel_dim,
)
predictor_kwargs = dict(
    patch_size=patch_size,
    embed_dim=predictor_embed_dim,
    depth=12,
    num_heads=12,
    qkv_bias=True,
    ijepa_encoder_dim=encoder_embed_dim,
    predictor_num_patches=num_patches,
)


def forward(self: ssl.Module, batch, stage):
    out = {}
    target_encoder: IJEPA_Encoder = self.target_encoder.teacher
    context_encoder: IJEPA_Encoder = self.context_encoder
    predictor: IJEPA_Predictor = self.predictor
    ijepa_loss: nn.Module = self.ijepa_loss
    with torch.no_grad():
        image_patches = target_encoder.patchify(batch["image"])
        target_patches = target_encoder.project_patches(image_patches)
        pos_embedding = pos_embed(target_patches)
        target_patches = target_patches + pos_embedding
        out["target_embedding"] = target_encoder.encode_patches(
            target_patches, with_layernorm=True
        )
        unmasked_context_patches = context_encoder.project_patches(image_patches)
        unmasked_pos_embedding = pos_embed(unmasked_context_patches)
        out["context_embedding"] = context_encoder.encode_patches(
            unmasked_context_patches + unmasked_pos_embedding,
            with_layernorm = False
        )
        out["meanpool_context_embedding"] = out["context_embedding"].mean(dim=1)
        out["sum_context_embedding"] = out["context_embedding"].sum(dim=1)
        out["flat_context_embedding"] = out["context_embedding"].reshape(out["context_embedding"].shape[0], -1)
    
    if not self.training:
        return out

    mask_context, masks_target = batch["mask_context"], batch["masks_target"]
    # target encoder is applied on full patches, then masked
    out["target_patches"] = apply_masks(out["target_embedding"], *masks_target)
    # context encoder is applied on masked patches
    context_patches = apply_masks(image_patches, mask_context)
    context_patches = context_encoder.project_patches(context_patches)
    context_patches = context_patches + apply_masks(pos_embedding, mask_context)
    context_patches = context_encoder.encode_patches(
        context_patches, with_layernorm=False
    )
    out["context_patches"] = context_patches
    # TODO Re-write this whole thing with the patchembeds inside the models themselves and using timms patchembed/block
    # but making ur own vits.
    # use context_patches.shape[0] because applying 4 masks quadruples the batch dimension for the predictor output
    predictor_pos_embed = apply_masks(predictor.predictor_pos_embed.repeat(context_patches.shape[0],1,1), mask_context)
    out["predicted_patches"] = predictor.predict_targets(
        predictor.project_context(context_patches) + predictor_pos_embed,
        masks_target
    )
    out["loss"] = ijepa_loss(out["predicted_patches"], out["target_patches"])
    # context embedding
    return out



module = ssl.Module(
    context_encoder=(ctx := IJEPA_Encoder(**encoder_kwargs)),
    target_encoder=TeacherStudentWrapper(ctx),
    predictor=IJEPA_Predictor(**predictor_kwargs),
    forward=forward,
    ijepa_loss=F.smooth_l1_loss,
    optim={
        "optimizer": {
            "type": "AdamW",
            "lr": lr,
            "weight_decay": 0.04,
        },
        "scheduler": {
            "type": "LinearWarmupCosineAnnealing",
            "peak_step": lr_warmup_epochs * len(train),
            "total_steps": num_epochs * len(train),
            "end_lr": final_lr,
        },
        "interval": "step",
    }
)


linear_probe = ssl.callbacks.OnlineProbe(
    "linear_probe",
    "meanpool_context_embedding",
    "label",
    probe=torch.nn.Sequential(
        torch.nn.BatchNorm1d(encoder_embed_dim, affine=False),
        torch.nn.Linear(encoder_embed_dim, num_classes)
    ),
    loss_fn=torch.nn.CrossEntropyLoss(),
    metrics={
        "top1": torchmetrics.classification.MulticlassAccuracy(num_classes),
        "top5": torchmetrics.classification.MulticlassAccuracy(num_classes, top_k=5),
    },
)

rankme = ssl.callbacks.RankMe(
    name="rankme",
    target="flat_context_embedding",
    queue_length=min(512, train_batch_size),  # NOTE must be >= batch_size
    target_shape=(encoder_embed_dim * num_patches),
)

class PerModuleGradLogger(pl.Callback):
    def __init__(self, modules=("predictor", "context_encoder", "target_encoder"), norm_type=2):
        self.modules = modules
        self.norm_type = norm_type

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        device = pl_module.device
        if trainer.global_step % 100 == 0:
            for mod in self.modules:
                group = [(n, p) for n, p in pl_module.named_parameters() if n.startswith(f"{mod}.")]
                grads = [p.grad for _, p in group if p.grad is not None]
                if not grads:
                    pl_module.log(f"grad/{mod}_norm", torch.tensor(0.0, device=device), on_step=True, logger=True, sync_dist=True)
                    pl_module.log(f"grad/{mod}_nz_params", torch.tensor(0, device=device), on_step=True, logger=True, sync_dist=True)
                    continue
                norms = torch.stack([g.detach().data.float().norm(self.norm_type).to(device) for g in grads])
                total_norm = torch.norm(norms, self.norm_type)
                nonzero = (norms > 0).sum()
                pl_module.log(f"grad/{mod}_norm", total_norm, on_step=True, logger=True, sync_dist=True)
                pl_module.log(f"grad/{mod}_nz_params", nonzero, on_step=True, logger=True, sync_dist=True)


# Initialize W&B logger with explicit settings
wandb_logger = WandbLogger(
    project="ijepa-cifar10",
    entity="samibg",  # Your W&B entity
    name="ijepa-inet1k-run",
    log_model=False,  # Set to True if you want to save model artifacts
    offline=False,  # Ensure offline mode
)

trainer = pl.Trainer(
    max_epochs=num_epochs,
    num_sanity_val_steps=0,  # Skip sanity check as queues need to be filled first
    callbacks=[linear_probe, rankme, PerModuleGradLogger(modules=("predictor","context_encoder","target_encoder")),
    TeacherStudentCallback(update_frequency=1, update_after_backward=True),
    ],
    precision="16-mixed",
    logger=wandb_logger,
    enable_checkpointing=False,
    accelerator="gpu",
    devices=8,
    gradient_clip_val=max_grad_norm,
    strategy=DDPStrategy(
        find_unused_parameters=True, # this is because only teacher's params are used in the teacher-student module
        static_graph=True,
        gradient_as_bucket_view=True,
    )
)

manager = ssl.Manager(trainer=trainer, module=module, data=data)
manager()
