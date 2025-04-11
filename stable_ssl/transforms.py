import math
from functools import cache
from typing import Union

import torch
from einops import rearrange
from torch import nn

from stable_ssl.utils import warn_once


#### NOTE All of these are slightly modified versions of what's in the JEPA repo.
def get_3d_sincos_pos_embed(
    embed_dim,
    grid_height,
    grid_width,
    grid_depth,
    cls_token=False,
    uniform_power=False,  # weights pos embeddings differently for temporal dimension vs spatial dimensions
):
    """
    Generate 3D sinusoidal positional embeddings for a 3D grid (depth, height, width).

    This function creates positional encodings for 3D data by generating sine and cosine
    embeddings for each dimension (depth, height, width) and concatenating them.

    Parameters
    ----------
        embed_dim (int): Dimension of the output embeddings.
        grid_height (int): Height of the 3D grid.
        grid_width (int): Width of the 3D grid.
        grid_depth (int): Depth (temporal dimension) of the 3D grid.
        cls_token (bool, optional): Whether to prepend a zero embedding for a classification token.
                                   Defaults to False.
        uniform_power (bool, optional): Distribution of embedding dimensions across spatial/temporal dimensions.
                                      If False, temporal dimension gets half the embedding dimensions.
                                      If True, dimensions are distributed equally. Defaults to False.

    Returns
    -------
        torch.Tensor: Positional embeddings of shape [grid_depth*grid_height*grid_width, embed_dim] without cls_token
                     or [1+grid_depth*grid_height*grid_width, embed_dim] with cls_token.
    """
    grid_d = torch.arange(grid_depth, dtype=torch.float)
    grid_h = torch.arange(grid_height, dtype=torch.float)
    grid_w = torch.arange(grid_width, dtype=torch.float)
    grid_h, grid_d, grid_w = torch.meshgrid(
        grid_h, grid_d, grid_w, indexing="ij"
    )  # order of meshgrid is very important for indexing as [d,h,w]

    if not uniform_power:
        h_embed_dim = embed_dim // 4
        w_embed_dim = embed_dim // 4
        d_embed_dim = embed_dim // 2
    else:
        h_embed_dim = w_embed_dim = d_embed_dim = int(
            torch.ceil(torch.tensor(embed_dim / 6)) * 2
        )

    emb_h = get_1d_sincos_pos_embed_from_grid(h_embed_dim, grid_h)  # (T*H*W, D1)
    emb_w = get_1d_sincos_pos_embed_from_grid(w_embed_dim, grid_w)  # (T*H*W, D2)
    emb_d = get_1d_sincos_pos_embed_from_grid(d_embed_dim, grid_d)  # (T*H*W, D3)
    pos_embed = torch.cat([emb_d, emb_h, emb_w], dim=1)
    pos_embed = pos_embed[:, :embed_dim]
    if cls_token:
        pos_embed = torch.cat([torch.zeros(1, embed_dim), pos_embed], dim=0)
    return pos_embed


def get_2d_sincos_pos_embed(embed_dim: int, grid_h: int, grid_w: int, cls_token=False):
    """
    Generate 2D sinusoidal positional embeddings for a 2D grid (height, width).

    This function creates positional encodings for 2D data by generating sine and cosine
    embeddings for each dimension (height, width) and concatenating them.

    Parameters
    ----------
        embed_dim (int): Dimension of the output embeddings.
        grid_h (int): Height of the 2D grid.
        grid_w (int): Width of the 2D grid.
        cls_token (bool, optional): Whether to prepend a zero embedding for a classification token.
                                   Defaults to False.

    Returns
    -------
        torch.Tensor: Positional embeddings of shape [grid_h*grid_w, embed_dim] without cls_token
                     or [1+grid_h*grid_w, embed_dim] with cls_token.
    """
    grid_h = torch.arange(grid_h, dtype=torch.float)
    grid_w = torch.arange(grid_w, dtype=torch.float)
    grid_w, grid_h = torch.meshgrid(
        grid_w, grid_h, indexing="ij"
    )  # order of meshgrid is very important for indexing as [h, w]

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_h)  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_w)  # (H*W, D/2)
    pos_embed = torch.cat([emb_h, emb_w], dim=1)  # (H*W, D)
    if cls_token:
        pos_embed = torch.cat([torch.zeros(1, embed_dim), pos_embed], dim=0)
    return pos_embed


def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    Generate 1D sinusoidal positional embeddings for a sequence of positions.

    This function creates positional encodings for a 1D sequence by generating
    sine and cosine embeddings for each position.

    Parameters
    ----------
        embed_dim (int): Dimension of the output embeddings.
        grid_size (int): Length of the sequence.
        cls_token (bool, optional): Whether to prepend a zero embedding for a classification token.
                                   Defaults to False.

    Returns
    -------
        torch.Tensor: Positional embeddings of shape [grid_size, embed_dim] without cls_token
                     or [1+grid_size, embed_dim] with cls_token.
    """
    grid = torch.arange(grid_size, dtype=torch.float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = torch.cat([torch.zeros(1, embed_dim), pos_embed], dim=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    Generate 1D sinusoidal positional embeddings from a grid of positions.

    This is the core function that implements sinusoidal position encoding according to the formula:
    PE(pos, 2i) = sin(pos / 10000^(2i/embed_dim))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/embed_dim))

    Parameters
    ----------
        embed_dim (int): Dimension of the output embeddings. Must be divisible by 2.
        pos (torch.Tensor): A tensor of positions to be encoded of shape (M,).

    Returns
    -------
        torch.Tensor: Sinusoidal embeddings of shape (M, embed_dim).

    Raises
    ------
        AssertionError: If embed_dim is not divisible by 2.
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


class Patchify2D(nn.Module):
    """Turn an image into patches of patch_size x patch_size.

    Parameters
    ----------
    patch_size : Union[int, tuple[int, int]], default=16
        Size of patches to extract. Can be either:
        - An integer for square patches (patch_size x patch_size)
        - A tuple of (height, width) for rectangular patches
    positional_encoding : nn.Module, default=PositionalEncoding()
        Positional encoding to apply to the patches

    Returns
    -------
    torch.Tensor
        Tensor of patches with shape [N, P, D] where:
        - N is number of patches (H/patch_size_height * W/patch_size_width)
        - P is number of pixels per patch (patch_size_height * patch_size_width)
        - D is the input channel dimension: 3x16x16 for RGB images
    """

    def __init__(
        self,
        patch_size: Union[int, tuple[int, int]] = 16,
        use_pos_embed: bool = True,
    ):
        super().__init__()
        if isinstance(patch_size, tuple):
            assert len(patch_size) == 2
            self.patch_size = patch_size
        if isinstance(patch_size, int):
            self.patch_size = (patch_size,) * 2

        self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
        self.use_pos_embed = use_pos_embed

    @cache
    def get_pos_embed(self, embed_dim: int, grid_h: int, grid_w: int) -> torch.Tensor:
        # NOTE Converting this to and from numpy could be slow but this is just a PoC for now
        sincos = get_2d_sincos_pos_embed(embed_dim, grid_h, grid_w, cls_token=False)
        return sincos

    # TODO Change this to hwc if unfold allows for it to make it similar to 3d patchify
    def __call__(self, image_hwc: torch.Tensor) -> torch.Tensor:
        H, W, C = image_hwc.shape

        grid_height = H // self.patch_size[0]
        grid_width = W // self.patch_size[1]
        assert (grid_height * self.patch_size[0]) == H and (
            grid_width * self.patch_size[1]
        ) == W

        # NOTE unfold needs channel dim to come first, but images are loaded as HWC
        # so we permute in this method
        image_patched_flat: torch.Tensor = self.unfold(image_hwc.permute(2, 0, 1)).T

        if self.use_pos_embed:
            # NOTE Since positional embeddings are lazily created based on the patched shape
            # and not the input shape, we don't ever need to interpolate them
            D = image_patched_flat.shape[-1]
            pos_embed = self.get_pos_embed(
                embed_dim=D, grid_h=grid_height, grid_w=grid_width
            )
            image_patched_flat += pos_embed

        # NOTE Unflatten patches to recover original shape
        image_patched_hwd: torch.Tensor = rearrange(
            image_patched_flat, "(gh gw) d -> gh gw d", gh=grid_height, gw=grid_width
        )
        # 224,224,3 16,16
        # 14, 14, (16*16*3)
        return image_patched_hwd


class Patchify3D(nn.Module):
    """Patchify a video tensor into tubelets with a certain patch size, similar to 3D convolutions.

    This module converts a video tensor into spatiotemporal patches (tubelets) by:
    1. Grouping frames into temporal chunks of size tubelet_size
    2. Within each chunk, extracting spatial patches of size patch_size x patch_size

    Parameters
    ----------
    patch_size : Union[int, tuple[int, int]], default=16
        Size of spatial patches to extract. Can be either:
        - An integer for square patches (patch_size x patch_size)
        - A tuple of (height, width) for rectangular patches
    tubelet_size : int, default=2
        Number of consecutive frames to group into each tubelet
    positional_encoding : nn.Module, default=PositionalEncoding()
        Positional encoding to apply to the patches

    Returns
    -------
    torch.Tensor
        Tensor of tubelets with shape [T, H, W, C] where:
        - T is number of frames (original T/tubelet_size)
        - H,W are spatial grid dimensions (original H,W/patch_size)
        - C is channel dimension (original C * patch_size^2 * tubelet_size)
    """

    def __init__(
        self,
        patch_size: Union[int, tuple[int, int]] = 16,
        tubelet_size: int = 2,
        use_pos_embed: bool = True,
    ):
        super().__init__()
        if isinstance(patch_size, tuple):
            assert len(patch_size) == 2
            self.patch_size = patch_size
        if isinstance(patch_size, int):
            self.patch_size = (patch_size,) * 2

        self.tubelet_size = tubelet_size
        self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
        self.use_pos_embed = use_pos_embed

    @cache
    def get_pos_embed(
        self, embed_dim: int, grid_h: int, grid_w: int, grid_t: int
    ) -> torch.Tensor:
        # NOTE Converting this to and from numpy could be slow but this is just a PoC for now
        sincos = get_3d_sincos_pos_embed(
            embed_dim, grid_h, grid_w, grid_t, cls_token=False
        )
        # sincos = torch.from_numpy(sincos).float()
        # NOTE Will this preserve the values of the positional embeddings
        # NOTE How can I check this? Means over th
        sincos = rearrange(sincos, "(gh gw t) d -> t d (gh gw)", gh=grid_h, gw=grid_w)
        return sincos

    def __call__(self, video_thwc: torch.Tensor) -> torch.Tensor:
        T, H, W, C = video_thwc.shape

        timesteps: int = T // self.tubelet_size
        assert (timesteps * self.tubelet_size) == T

        grid_h, grid_w = H // self.patch_size[0], W // self.patch_size[1]
        assert (grid_h * self.patch_size[0]) == H and (grid_w * self.patch_size[1]) == W

        video_tubed = rearrange(
            video_thwc, "(n t) h w c -> n (t c) h w", n=timesteps, t=self.tubelet_size
        )

        video_patched_flattened: torch.Tensor = self.unfold(video_tubed)
        if self.use_pos_embed:
            D = video_patched_flattened.shape[1]
            pos_embed = self.get_pos_embed(
                embed_dim=D, grid_h=grid_h, grid_w=grid_w, grid_t=timesteps
            )
            video_patched_flattened += pos_embed

        video_patched_thwc: torch.Tensor = rearrange(
            video_patched_flattened,
            "n (t c ph pw) (gh gw) -> (n t) gh gw (c ph pw)",
            t=self.tubelet_size,
            c=C,
            gh=grid_h,
            gw=grid_w,
            ph=self.patch_size[0],
            pw=self.patch_size[1],
        )
        return video_patched_thwc


class TubeMask:
    """Apply tube masking to spatiotemporal video data by masking aligned spatial patches across time.

    This class implements tube masking as used in V-JEPA and similar architectures. It can handle:
    1. Raw video tensors [T, H, W, C]
    2. Pre-patchified tensors where H,W represent a grid of patches

    For example, given:
    - Raw video: [16, 224, 224, 3]
    - Patchified video: [16, 14, 14, 768] (using 16x16 patches)
    The masking pattern is consistent across the temporal dimension, creating "tubes".

    Parameters
    ----------
    ratio : float
        Ratio of patches to mask out (between 0 and 1)
    patch_size : Union[tuple[int, int], int]
        Size of patches for masking. For pre-patchified input, use (1,1)

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Two tensors containing:
        1. Kept patches with shape [T, N_kept, C]
        2. Masked patches with shape [T, N_masked, C]
        where N_kept + N_masked = H*W/patch_size^2
    """

    def __init__(
        self,
        ratio: float,
        patch_size: Union[tuple[int, int], int],
    ):
        super(TubeMask, self).__init__()
        self.ratio = ratio
        if isinstance(patch_size, int):
            self.patch_size = (patch_size,) * 2
        else:
            self.patch_size = patch_size

    def sample_spatial_mask(
        self, ratio: float, num_spatial_patches: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate spatial masking pattern to be applied across temporal dimension.

        Parameters
        ----------
        ratio : float
            Ratio of patches to mask (between 0 and 1)
        num_spatial_patches : int
            Total number of spatial patches (H*W/patch_size^2)

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Indices of patches to keep and discard
        """
        num_keep_spatial = int(num_spatial_patches * (1.0 - ratio))
        mask = torch.cat(
            [
                torch.zeros(num_spatial_patches - num_keep_spatial),
                torch.ones(num_keep_spatial),
            ]
        )
        # NOTE Equivalent to np.random.shuffle(mask)
        mask = mask.view(-1)[torch.randperm(mask.nelement())].view(mask.size())
        mask_discard = torch.argwhere(mask == 0).squeeze()
        mask_keep = torch.nonzero(mask).squeeze()
        return mask_keep, mask_discard

    def __call__(self, video_thwc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply tube masking to input video.

        Parameters
        ----------
        video_thwc : torch.Tensor
            Input video tensor in [T, H, W, C] format
            Can be either raw video or pre-patchified
            If input tensor is an image-like [H, W, C],
            it will be casted to a video-like by the addition
            of an extra temporal dimension into [1, H, W, C]

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Kept patches and masked patches
            Both have shape [T, N, C] where N varies based on ratio
        """
        if is_image := bool(video_thwc.dim() == 3):
            warn_once(
                f"Received image-like input with shape {video_thwc.shape}, ",
                "adding temporal dimension of 1!",
            )
            video_thwc = video_thwc.unsqueeze(0)

        assert video_thwc.dim() == 4, (
            f"TubeMask only accepts videos with dimensions \
                                [T, H, W, C], received {video_thwc.shape} instead! "
        )

        T, H, W, C = video_thwc.shape
        num_patches_spatial: int = (H // self.patch_size[0]) * (W // self.patch_size[1])
        mask_keep, mask_discard = self.sample_spatial_mask(
            self.ratio, num_patches_spatial
        )
        mask_keep, mask_discard = (
            mask_keep.unsqueeze(-1).expand(T, -1, C),
            mask_discard.unsqueeze(-1).expand(T, -1, C),
        )
        # Flatten video across the H,W dimensions
        video_flattened_grid = rearrange(video_thwc, "t h w c -> t (h w) c")
        # since the masks contain indices to keep, we can use gather to apply the masking:
        masked_video_keep = torch.gather(video_flattened_grid, dim=1, index=mask_keep)
        masked_video_discard = torch.gather(
            video_flattened_grid, dim=1, index=mask_discard
        )

        if is_image:
            masked_video_keep = masked_video_keep.squeeze(0)
            masked_video_discard = masked_video_discard.squeeze(0)

        return (
            masked_video_keep.clone(),
            mask_keep.clone(),
            masked_video_discard.clone(),
            mask_discard.clone(),
        )


class MultiBlock3DMask:
    """Apply multi-block 3D masking to spatiotemporal video data.

    This implements the masking strategy from JEPA, which generates multiple block masks
    with configurable spatial and temporal scales, aspect ratios, and number of blocks.

    Parameters
    ----------
    spatial_scale : tuple[float, float]
        Min and max scale for spatial masking (e.g. (0.2, 0.8))
    temporal_scale : tuple[float, float]
        Min and max scale for temporal masking (e.g. (1.0, 1.0))
    aspect_ratio : tuple[float, float]
        Min and max aspect ratios for blocks (e.g. (0.3, 3.0))
    num_blocks : int
        Number of mask blocks to generate
    max_temporal_keep : float
        Maximum ratio of temporal frames to keep (1.0 = all frames)
    patch_size : Union[tuple[int, int], int]
        Size of patches to mask
    """

    def __init__(
        self,
        spatial_scale: tuple[float, float] = (0.2, 0.8),
        temporal_scale: tuple[float, float] = (1.0, 1.0),
        aspect_ratio: tuple[float, float] = (0.3, 3.0),
        num_blocks: int = 1,
        max_temporal_keep: float = 1.0,
        patch_size: Union[tuple[int, int], int] = (16, 16),
    ):
        if isinstance(patch_size, int):
            self.patch_size = (patch_size,) * 2
        else:
            self.patch_size = patch_size

        self.spatial_scale = spatial_scale
        self.temporal_scale = temporal_scale
        self.aspect_ratio = aspect_ratio
        self.num_blocks = num_blocks
        self.max_temporal_keep = max_temporal_keep

    def _sample_block_size(
        self,
        height: int,
        width: int,
        duration: int,
        generator: torch.Generator,
    ) -> tuple[int, int, int]:
        """Sample a random block size given constraints."""
        min_t, max_t = self.temporal_scale
        t_scale = min_t + torch.rand(1, generator=generator).item() * (max_t - min_t)
        t = max(1, int(duration * t_scale))

        # Sample spatial block size
        min_s, max_s = self.spatial_scale
        s_scale = min_s + torch.rand(1, generator=generator).item() * (max_s - min_s)
        spatial_num_keep = int(height * width * s_scale)

        # Sample aspect ratio
        min_ar, max_ar = self.aspect_ratio
        ar = min_ar + torch.rand(1, generator=generator).item() * (max_ar - min_ar)

        # Calculate block height/width
        h = int(round(math.sqrt(spatial_num_keep * ar)))
        w = int(round(math.sqrt(spatial_num_keep / ar)))
        h = min(h, height)
        w = min(w, width)

        return (t, h, w)

    def _sample_block_mask(
        self,
        block_size: tuple[int, int, int],
        height: int,
        width: int,
        duration: int,
    ) -> torch.Tensor:
        """Generate a single block mask."""
        t, h, w = block_size
        top = torch.randint(0, height - h + 1, (1,))
        left = torch.randint(0, width - w + 1, (1,))
        start = torch.randint(0, duration - t + 1, (1,))

        mask = torch.ones((duration, height, width), dtype=torch.int32)
        mask[start : start + t, top : top + h, left : left + w] = 0

        max_context_duration = max(1, int(duration * self.max_temporal_keep))
        if max_context_duration < duration:
            mask[max_context_duration:, :, :] = 0

        return mask

    def __call__(self, video_thwc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply masking to input video. An image tensor can be passed as well, in which case the mask will interpret it as a single frame.

        Parameters
        ----------
        video_thwc : torch.Tensor
            Input video tensor [T, H, W, C]

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Kept patches [T, N_kept, C] and masked patches [T, N_masked, C]
        """
        # NOTE Add temporal dimension for image
        if is_image := bool(video_thwc.ndim == 3):
            video_thwc = video_thwc.unsqueeze(0)

        assert video_thwc.dim() == 4, (
            f"MultiBlock3D only accepts videos with dimensions \
                                [T, H, W, C] or [H, W, C] received {video_thwc.shape} instead! "
        )

        T, H, W, C = video_thwc.shape
        grid_h, grid_w = H // self.patch_size[0], W // self.patch_size[1]

        block_size = self._sample_block_size(grid_h, grid_w, T, torch.Generator())

        mask = torch.ones((T, grid_h, grid_w), dtype=torch.int32)
        for _ in range(self.num_blocks):
            mask *= self._sample_block_mask(block_size, grid_h, grid_w, T)

        mask = mask.flatten()
        mask_discard = torch.argwhere(mask == 0).squeeze()
        mask_keep = torch.nonzero(mask).squeeze()

        video_flat = rearrange(video_thwc, "t h w c -> (t h w) c")

        masked_video_keep = video_flat[mask_keep].reshape(T, -1, C)
        masked_video_discard = video_flat[mask_discard].reshape(T, -1, C)

        # remove temporal dimension for image
        if is_image:
            masked_video_keep = masked_video_keep.squeeze(0)
            masked_video_discard = masked_video_discard.squeeze(0)

        return (
            masked_video_keep.clone(),
            mask_keep.clone(),
            masked_video_discard.clone(),
            mask_discard.clone(),
        )
