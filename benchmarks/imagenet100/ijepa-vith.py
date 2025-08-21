import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from timm.models.vision_transformer import PatchEmbed, Block
import lightning.pytorch.loggers as pl_loggers
import torchmetrics
from stable_ssl.backbone.utils import TeacherStudentWrapper
from stable_ssl.callbacks.teacher_student import TeacherStudentCallback
from lightning.pytorch.strategies import DDPStrategy
from stable_ssl.data import transforms


import stable_ssl as ssl
import lightning as pl
from stable_ssl.utils.pos_embed import get_2d_sincos_pos_embed

NUM_DEVICES = 8
BATCH_SIZE = 256
VAL_BATCH_SIZE = (16384 // NUM_DEVICES)
# VAL_LR_SWEEPS = [0.01, 0.05, 0.001]
LR_MULTIPLIER = (BATCH_SIZE * NUM_DEVICES) / (128 * 16)


def apply_masks(x: torch.Tensor, masks: list[torch.Tensor]) -> torch.Tensor:
    B, N, D = x.shape
    M = len(masks)
    idx = torch.stack(
        [m.to(x.device, dtype=torch.long) for m in masks], dim=1
    )  # [B, M, K]
    x_exp = x.unsqueeze(1).expand(-1, M, -1, -1)  # [B, M, N, D]
    out = x_exp.gather(2, idx.unsqueeze(-1).expand(-1, -1, -1, D))  # [B, M, K, D]
    return out.reshape(B * M, idx.size(-1), D)  # [B*M, K, D]

def repeat_interleave_batch(x: torch.Tensor, B, repeat):
    N = x.shape[0] // B
    x = x[:N*B]
    return x.reshape(N, B, *x.shape[1:]) \
            .repeat_interleave(repeat, dim=1) \
            .reshape(N * B * repeat, *x.shape[1:])

def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def fix_init_weight(blocks):
    def rescale(param, layer_id):
        param.div_(math.sqrt(2.0 * layer_id))

    for layer_id, layer in enumerate(blocks):
        rescale(layer.attn.proj.weight.data, layer_id + 1)
        rescale(layer.mlp.fc2.weight.data, layer_id + 1)

class IJEPA_ViT_Encoder(nn.Module):
    def __init__(
        self,
        img_size=224, patch_size=14, embed_dim=768,
        depth=12, num_heads=12,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.norm_layer = nn.LayerNorm
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=3, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        # --
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1],
                                            int(self.patch_embed.num_patches**.5),
                                            cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # --
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads,
                mlp_ratio=4.0, qkv_bias=True, attn_drop=0.0, drop_path=0.0,
                norm_layer=self.norm_layer
            )
            for _ in range(depth)
        ])
        self.norm = self.norm_layer(embed_dim)
        self.apply(init_weights)
        fix_init_weight(self.blocks)

    def forward(self, x, masks: None | list = None):
        x = self.patch_embed(x)
        x = x + self.pos_embed

        if masks is not None:
            x = apply_masks(x, masks)

        for blk in self.blocks:
            x = blk(x)

        return self.norm(x)

class IJEPA_ViT_Predictor(nn.Module):
    def __init__(
        self,
        num_patches, embed_dim=768, predictor_embed_dim=384,
        depth=6, num_heads=12,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        # --
        self.predictor_pos_embed = nn.Parameter(torch.zeros(1, num_patches, predictor_embed_dim),
                                                requires_grad=False)
        predictor_pos_embed = get_2d_sincos_pos_embed(self.predictor_pos_embed.shape[-1],
                                                      int(num_patches**.5),
                                                      cls_token=False)
        self.predictor_pos_embed.data.copy_(torch.from_numpy(predictor_pos_embed).float().unsqueeze(0))
        # --
        self.predictor_blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim, num_heads=num_heads,
                mlp_ratio=4, qkv_bias=True,
                attn_drop=0.0, drop_path=0.0,
                norm_layer=nn.LayerNorm
            )
            for _ in range(depth)
        ])
        self.predictor_norm = nn.LayerNorm(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        # ------
        trunc_normal_(self.mask_token, std=0.02)
        self.apply(init_weights)
        fix_init_weight(self.predictor_blocks)

    def forward(self, context_patches, masks_context: list, masks_target: list):
        assert len(masks_context) == 1
        B = len(context_patches) // len(masks_context)
        x = self.predictor_embed(context_patches)
        x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
        x += apply_masks(x_pos_embed, masks_context)

        _, N_ctxt, D = x.shape

        # -- concat mask tokens to x
        pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
        pos_embs = apply_masks(pos_embs, masks_target)
        pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_context))
        # --
        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
        # --
        pred_tokens += pos_embs
        x = x.repeat(len(masks_target), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        # -- fwd prop
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        # -- return preds for mask tokens
        x = x[:, N_ctxt:]
        x = self.predictor_proj(x)

        return x

def forward_ijepa(self: ssl.Module, batch: dict, stage: str) -> dict:
    out = {}

    with torch.no_grad():
        # -- keys for rankme and linear probe
        out['context_embeddings'] = self.context_encoder(batch['image'], masks = None)
        out['meanpool'] = out['context_embeddings'].mean(dim=1)
        out['flat'] = out['context_embeddings'].reshape(out['context_embeddings'].shape[0], -1)
        
        if not self.training:
            return out

        target_patches = self.target_encoder(batch['image'])
        target_patches = F.layer_norm(target_patches, (target_patches.size(-1),))
        target_patches = apply_masks(target_patches, batch['masks_target'])
        out['target_embeddings'] = target_patches

    out['context_embeddings'] = self.context_encoder(batch['image'], [batch['mask_context']])
    out['predictions'] = self.predictor(
        out['context_embeddings'],
        [batch['mask_context']], 
        batch['masks_target']
    )
    out['loss'] = self.ijepa_loss(out['predictions'], out['target_embeddings'])
    return out


inet1k_train = ssl.data.HFDataset(
    path="clane9/imagenet-100",
    split="train",
    transform=transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop((224, 224), scale=(0.3, 1.0)),
        transforms.ContextTargetsMultiBlockMask(
            patch_size=14,
            context_scale=(0.85, 1.0),
            context_aspect_ratio=(1.0, 1.0),
            target_scales=((0.15, 0.2),) * 4,
            target_aspect_ratios=((0.75, 1.5),) * 4,
            min_keep=10,
        ),
        transforms.ToImage(
            mean=(0.485, 0.456, 0.406),
            std= (0.229, 0.224, 0.225),
        ),
    ),
)

inet1k_val = ssl.data.HFDataset(
    path="clane9/imagenet-100",
    split="validation",
    transform=transforms.Compose(
        transforms.RGB(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToImage(
            mean=(0.485, 0.456, 0.406),
            std= (0.229, 0.224, 0.225),
        ),
    ),
)

# collate function that makes each mask have the same number of indices, so they can be batched
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

train = torch.utils.data.DataLoader(
    dataset=inet1k_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=16,
    drop_last=True,
    collate_fn=standardize_masks,
    pin_memory=True,
    persistent_workers=True,
)

val = torch.utils.data.DataLoader(
    dataset=inet1k_val,
    batch_size=VAL_BATCH_SIZE,
    num_workers=16,
    shuffle=False,
)

encoder = IJEPA_ViT_Encoder()
predictor = IJEPA_ViT_Predictor(num_patches=encoder.patch_embed.num_patches)
target_encoder = TeacherStudentWrapper(encoder, base_ema_coefficient=0.996, final_ema_coefficient=1.0)

ema_callback = TeacherStudentCallback(update_frequency=1, update_after_backward=True)
rankme = ssl.callbacks.RankMe(
    name="rankme", target="flat",
    queue_length=min(512, BATCH_SIZE),
    target_shape=(encoder.embed_dim * encoder.patch_embed.num_patches)
)
sweep_lr = 0.005
linear_probe = ssl.callbacks.OnlineProbe(
        name=f'linear_probe_lr{sweep_lr:.0e}', input='meanpool', target='label',
        probe=torch.nn.Sequential(
            torch.nn.BatchNorm1d(encoder.embed_dim, affine=False),
            torch.nn.Linear(encoder.embed_dim, 100)
        ),
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer={
            "type": "LARS",
            "lr": sweep_lr,
            "weight_decay": 1e-6,
        },
        scheduler={
            "type": "StepLR",
            "step_size": 15,
            "gamma": 0.1,
        },
        metrics={
            "top1": torchmetrics.classification.MulticlassAccuracy(100),
            "top5": torchmetrics.classification.MulticlassAccuracy(100, top_k=5),
        }
    )

module = ssl.Module(
    context_encoder=encoder,
    target_encoder=target_encoder,
    predictor=predictor,
    forward=forward_ijepa,
    ijepa_loss=F.smooth_l1_loss,
    optim={
        "optimizer": {
            "type": "AdamW",
            "lr": 0.001 * LR_MULTIPLIER,
            "weight_decay": 0.04,  # TODO Scheduler to 0.4
        },
        "scheduler": {
            "type": "LinearWarmupCosineAnnealing",
            "peak_step": 40 * len(train),
            "start_factor": 1 / 5,
            "total_steps": 300 * len(train),
            "end_lr": 1.0e-6 * LR_MULTIPLIER,
        },
        "interval": "step",
    }
)

trainer = pl.Trainer(
    max_epochs=300, num_sanity_val_steps=0,
    callbacks=[
        linear_probe, rankme, ema_callback
    ],
    precision='16-mixed',
    logger=pl_loggers.WandbLogger(
        project="ijepa-cifar10", entity="samibg", name="new-ijepa-inet100",
        log_model=False, offline=False,
    ),
    enable_checkpointing=False,
    accelerator="gpu", devices=NUM_DEVICES, gradient_clip_val=None,
    strategy=DDPStrategy(
        find_unused_parameters=True, # this is because only teacher's params are used in the teacher-student module
        static_graph=True,
        gradient_as_bucket_view=True,
    )
)

data = ssl.data.DataModule(train=train, val=val)

manager = ssl.Manager(trainer=trainer, module=module, data=data)

if __name__ == "__main__":
    manager()