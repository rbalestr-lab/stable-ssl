import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer

encoder_kwargs = dict(
    img_size=32,
    patch_size=16,
    in_chans=3,
    embed_dim=768,
    depth=12,
    # include cls token for positional embedding:
    # https://github.com/facebookresearch/mae/blob/main/models_mae.py#L68-L69
    no_embed_class=False, 
    norm_layer=nn.LayerNorm,
)

decoder_kwargs = dict(
    img_size=32,
    patch_size=16,
    in_chans=3,
    embed_dim=768,
    depth=12,
    class_token=True,
    no_embed_class=False,
    norm_layer=nn.LayerNorm,
)

class MAE_Encoder(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
                # --------------------------------------------------------------------------
        # MAE encoder specifics
        # replicated from timm's 
        self.num_patches = self.patch_embed.num_patches + self.num_prefix_tokens
        # TODO Exclude this and add posembeds from outside ? Can we do that ? I dont think so since we work on raw pixels
        # self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.embed_dim), requires_grad=False)  # fixed sin-cos embedding


class MAE_Decoder(VisionTransformer):
    def __init__(self, *args, **kwargs):
        patch_size = kwargs.get('patch_size', 16)
        super().__init__(*args, **kwargs)
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        # replicated from timm's 
        self.num_patches = self.patch_embed.num_patches + self.num_prefix_tokens
        self.out_proj = nn.Linear(self.embed_dim, self.num_patches * patch_size**2)

def pos_embed(patches: torch.Tensor, with_cls: bool = True) -> torch.Tensor:
    pass


def forward_encoder(encoder: MAE_Encoder, images: torch.Tensor, mask:torch.Tensor) -> torch.Tensor:
    patches = encoder.patch_embed(images)
    posemb = pos_embed(patches, with_cls = True)
    posemb_patches = patches + posemb[:, 1:, :]
    cls_token = encoder.cls_token + posemb[:, :1, :]
    cls_tokens = cls_token.expand(patches.shape[0], -1, -1)

    masked_patches = apply_mask(posemb_patches, mask)
    
    patches = torch.cat([cls_tokens, masked_patches], dim=1)
    for blk in encoder.blocks:
        patches = blk(patches)
    
    patches = encoder.norm(patches)
    return patches

def apply_mask(patches: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # TODO
    pass

def forward_decoder(decoder: MAE_Decoder, batch: dict) -> torch.Tensor:
    patches = batch["patches"]
    ids_restore = batch["ids_restore"]

    patches = decoder.patch_embed(patches)
    
    # NOTE from mae repo directly
    # ids_restore im assuming can be the total set of indices
    # dim1 is the number of tokens to predict. +1 for cls token?
    # so we dont predict cls token?
    mask_tokens = decoder.mask_token.repeat(
        patches.shape[0],
        ids_restore.shape[1] + 1 - patches.shape[1],
        1
    )
    # we want to pos-embed according to their original positions
    # so we unshuffle first using ids_restore which is the inverse permutation
    patches_ = torch.cat([patches[:, 1:, :], mask_tokens], dim=1)
    patches_ = torch.gather(patches_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, patches.shape[2]))  # unshuffle
    # append cls token
    patches  = torch.cat([patches[:, :1, :], patches_], dim=1)
    patches  = patches + pos_embed(patches, with_cls = True)
    # apply transformer blocks
    patches = decoder.blocks(patches)
    patches = decoder.norm(patches)
    # predictor projection
    patches = decoder.pred(patches)
    # remove cls token
    return patches[:, 1:, :]

def forward_mae(self, batch: dict, stage):
    out = {}
    encoder = self.encoder
    decoder = self.decoder
    
    if stage == "train":
        pass