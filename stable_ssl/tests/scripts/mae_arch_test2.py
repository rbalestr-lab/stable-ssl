import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer

# -------------------- utils: positional embeddings --------------------

def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = True):
    """Returns [1, grid_size*grid_size + cls, embed_dim] fixed sin-cos embeddings."""
    def get_1d_sin_cos(d, n):
        omega = torch.arange(d // 2, dtype=torch.float32) / (d // 2)
        omega = 1.0 / (10000 ** omega)  # [d/2]
        pos = torch.arange(n, dtype=torch.float32)  # [n]
        out = torch.einsum('n,d->nd', pos, omega)  # [n, d/2]
        return torch.cat([out.sin(), out.cos()], dim=1)  # [n, d]

    assert embed_dim % 2 == 0
    # 2D grid
    pe_h = get_1d_sin_cos(embed_dim // 2, grid_size)  # [G, D/2]
    pe_w = get_1d_sin_cos(embed_dim // 2, grid_size)  # [G, D/2]
    pe = (
        torch.stack([pe_h.unsqueeze(1).expand(-1, grid_size, -1),
                     pe_w.unsqueeze(0).expand(grid_size, -1, -1)], dim=2)
        .reshape(grid_size * grid_size, embed_dim)
    )  # [G*G, D]
    if cls_token:
        pe = torch.cat([torch.zeros(1, embed_dim), pe], dim=0)  # [1+G*G, D]
    return pe.unsqueeze(0)  # [1, 1+G*G, D]

def pos_embed(x: torch.Tensor, with_cls: bool = True) -> torch.Tensor:
    """
    x: [B, N(=H*W or tokens), D]
    Returns fixed sin-cos PE of shape [B, (1+N), D] if with_cls else [B, N, D].
    """
    B, N, D = x.shape
    G = int(math.sqrt(N))  # assumes square grid of patches
    assert G * G == N, f"pos_embed expects square grid, got N={N}"
    pe = get_2d_sincos_pos_embed(D, G, cls_token=with_cls).to(x.device, x.dtype)
    return pe.expand(B, -1, -1)  # [B, 1+N, D] or [B, N, D] depending on with_cls

# -------------------- utils: masking and patchify --------------------

def apply_mask(x: torch.Tensor, ids_keep: torch.Tensor) -> torch.Tensor:
    """
    Keep selection by indices.
    x: [B, N, D], ids_keep: [B, K]
    returns: [B, K, D]
    """
    B, N, D = x.shape
    idx = ids_keep.unsqueeze(-1).expand(-1, -1, D)  # [B, K, D]
    return x.gather(dim=1, index=idx)


# -------------------- encoder --------------------

class MAE_Encoder(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # number of patch tokens (no cls)
        self.patch_size = kwargs.get('patch_size', 16)
        self.num_patches = self.patch_embed.num_patches  # do NOT add prefix here

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [B, C, H, W] -> [B, N, P*P*C]
        """
        B, C, H, W = images.shape
        P = self.patch_size
        assert H % P == 0 and W % P == 0
        h = H // P
        w = W // P
        x = images.reshape(B, C, h, P, w, P)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, h * w, P * P * C)
        return x


def forward_encoder(encoder: MAE_Encoder, images: torch.Tensor, ids_keep: torch.Tensor) -> torch.Tensor:
    """
    Returns visible token latents (with cls prepended), projected by encoder blocks.
    """
    # Patch embed: [B, N, D]
    x = encoder.patch_embed(images)
    # Fixed PE
    pe = pos_embed(x, with_cls=True)            # [B, 1+N, D]
    x = x + pe[:, 1:, :]                        # add PE to patch tokens
    # cls token with PE
    cls_tok = encoder.cls_token + pe[:, :1, :]  # [B, 1, D]
    cls_tok = cls_tok.expand(x.shape[0], -1, -1)

    # Keep only visible tokens
    x_vis = apply_mask(x, ids_keep)             # [B, K, D]
    # prepend cls
    x = torch.cat([cls_tok, x_vis], dim=1)      # [B, 1+K, D]

    # Blocks
    for blk in encoder.blocks:
        x = blk(x)
    x = encoder.norm(x)
    return x  # [B, 1+K, D]

# -------------------- decoder --------------------

class MAE_Decoder(VisionTransformer):
    def __init__(self, *args, **kwargs):
        in_chans = kwargs.get('in_chans', 3)
        patch_size = kwargs.get('patch_size', 16)
        self.enc_dim = kwargs.get('enc_dim', 768)
        super().__init__(*args, **kwargs)

        # tokens in the decoded grid (no cls)
        self.num_patches = self.patch_embed.num_patches

        # Map encoder dim -> decoder dim (use embed_dim of this decoder)
        # You must set this externally once you know the encoder dim:
        self.decoder_embed = nn.Linear(self.enc_dim, self.embed_dim)  # set to Linear(enc_dim, self.embed_dim) after init

        # mask token for missing positions
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        # predict pixel values per token (P^2 * C)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.decoder_pred = nn.Linear(self.embed_dim, (patch_size ** 2) * in_chans)

def forward_decoder(
    decoder: MAE_Decoder,
    enc_tokens_vis: torch.Tensor,  # [B, 1+K, D_enc] (cls + visible)
    ids_restore: torch.Tensor      # [B, N]
) -> torch.Tensor:
    """
    Returns pixel predictions for all tokens (no cls): [B, N, P^2*C]
    """
    B, _, D_enc = enc_tokens_vis.shape

    # project to decoder dim
    x = decoder.decoder_embed(enc_tokens_vis)   # [B, 1+K, D_dec]

    # prepare full grid by inserting mask tokens at missing positions
    # 1) split cls vs visible
    x_cls, x_vis = x[:, :1, :], x[:, 1:, :]     # [B,1,D], [B,K,D]
    # 2) number of tokens in grid
    N = decoder.num_patches
    K = x_vis.shape[1]
    # 3) tokens to fill
    mask_tokens = decoder.mask_token.expand(B, N - K, -1)  # [B, N-K, D]
    # 4) combine vis + mask, then unshuffle
    x_ = torch.cat([x_vis, mask_tokens], dim=1)  # [B, N, D]
    x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x_.shape[-1]))  # [B, N, D]
    # 5) prepend cls
    x = torch.cat([x_cls, x_], dim=1)  # [B, 1+N, D]

    # add fixed PE for decoder
    pe = pos_embed(x[:, 1:, :], with_cls=True)  # reuse shape logic
    x = x + pe                                  # [B, 1+N, D]

    # transformer blocks
    for blk in decoder.blocks:
        x = blk(x)
    x = decoder.norm(x)

    # predict pixels per token, drop cls
    x = decoder.decoder_pred(x[:, 1:, :])  # [B, N, P^2*C]
    return x

# -------------------- MAE training forward --------------------
mask_ratio = 0.75

def forward_mae(self, batch: dict, stage):
    """
    Expects batch["image"]: [B,3,H,W]
    Produces MAE reconstruction loss on masked patches.
    """
    out = {}
    encoder: MAE_Encoder = self.encoder
    decoder: MAE_Decoder = self.decoder
    
    images      = batch["image"]  # [B,3,H,W]
    ids_keep    = batch["mask_visible"]
    ids_restore = batch["ids_restore"]
    mask_masked = batch["mask_masked"]
    
    B = images.shape[0]
    N = encoder.num_patches

    # 2) encode visible tokens (cls+visible)
    enc_tokens_vis = forward_encoder(encoder, images, ids_keep)  # [B,1+K,D_enc]

    # 4) decode to pixel predictions for ALL tokens (no cls)
    pred_pix = forward_decoder(decoder, enc_tokens_vis, ids_restore)  # [B,N,P^2*C]

    # 5) ground-truth patches
    target_pix = encoder.patchify(images)  # [B,N,P^2*C]

    # 6) compute loss ONLY on masked tokens
    mask_exp = mask_masked.unsqueeze(-1).type_as(pred_pix)  # [B,N,1]
    loss = self.loss_fn(pred_pix, target_pix, mask_exp)

    # 7) populate outputs for logging / probes
    if stage != "train":
        return out
    
    out["loss"] = loss
    out["mask_ratio"] = torch.tensor(mask_ratio, device=images.device)
    out["ids_restore"] = ids_restore
    out["mask"] = mask_masked
    return out