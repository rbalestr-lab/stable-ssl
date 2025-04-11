import pytest
import torch

from stable_ssl.transforms import MultiBlock3DMask, Patchify2D, Patchify3D, TubeMask


# -----------------------------------------------------------------------------
# FIXTURES
# -----------------------------------------------------------------------------
@pytest.fixture
def image() -> torch.Tensor:
    """Return a random image tensor of shape (224, 224, 3)."""
    return torch.randn(224, 224, 3)


@pytest.fixture
def video() -> torch.Tensor:
    """Return a random video tensor of shape (16, 224, 224, 3)."""
    return torch.randn(16, 224, 224, 3)


@pytest.fixture(scope="module")
def patchifier_2d() -> Patchify2D:
    """Return a default 2D patchifier with positional embedding enabled."""
    return Patchify2D(patch_size=16, use_pos_embed=True)


@pytest.fixture(scope="module")
def patchifier_3d() -> Patchify3D:
    """Return a default 3D patchifier with tubelet_size=2 and positional embedding enabled."""
    return Patchify3D(patch_size=16, tubelet_size=2, use_pos_embed=True)


@pytest.fixture(scope="module")
def tube_mask_half_1x1() -> TubeMask:
    """Return a default TubeMask with a ratio=0.5 and a patch_size of 1x1."""
    return TubeMask(ratio=0.5, patch_size=(1, 1))


@pytest.fixture(scope="module")
def multi_block_3d_mask() -> MultiBlock3DMask:
    return MultiBlock3DMask(
        spatial_scale=(0.2, 0.8),
        temporal_scale=(1.0, 1.0),
        aspect_ratio=(0.3, 3.0),
        num_blocks=1,
        max_temporal_keep=1.0,
        patch_size=(1, 1),
    )


# -----------------------------------------------------------------------------
# PATCHIFY2D TESTS
# -----------------------------------------------------------------------------
def test_patchify2d_smoke(image: torch.Tensor):
    """
    Smoke test for Patchify2D without positional embedding.

    This test instantiates a local patchifier with use_pos_embed=False.
    """
    patchifier_no_posembed = Patchify2D(patch_size=16, use_pos_embed=False)
    patches = patchifier_no_posembed(image)
    grid_h, grid_w, expected_channels = (
        image.shape[0] // 16,  # 14
        image.shape[1] // 16,  # 14
        16 * 16 * image.shape[2],  # 768
    )
    assert patches.shape == (grid_h, grid_w, expected_channels)


def test_patchify2d_with_pos_embed(image: torch.Tensor, patchifier_2d: Patchify2D):
    """
    Test Patchify2D with positional embedding enabled (default fixture).

    Verifies that the output shape matches the expected grid and flattened patch size.
    """
    patches = patchifier_2d(image)
    grid_h, grid_w, expected_channels = (
        image.shape[0] // 16,
        image.shape[1] // 16,
        16 * 16 * image.shape[2],
    )
    assert patches.shape == (grid_h, grid_w, expected_channels)


def test_patchify2d_invalid_shape(patchifier_2d: Patchify2D):
    """Verify that Patchify2D raises an assertion when image dimensions are not divisible by the patch size."""
    # height is not divisible by num_patches (225 % 16 != 0)
    bad_image = torch.randn(225, 224, 3)
    with pytest.raises(AssertionError):
        patchifier_2d(bad_image)


# -----------------------------------------------------------------------------
# PATCHIFY3D TESTS
# -----------------------------------------------------------------------------
def test_patchify3d_smoke(video: torch.Tensor):
    """
    Smoke test for Patchify3D without positional embedding.

    Instantiates a local patchifier with use_pos_embed=False.
    """
    patchifier = Patchify3D(patch_size=16, tubelet_size=2, use_pos_embed=False)
    patches = patchifier(video)
    grid_h, grid_w, expected_channels = (
        video.shape[1] // 16,
        video.shape[2] // 16,
        16 * 16 * video.shape[3],
    )
    # expected output shape: (T, grid_h, grid_w, patch_pixels)
    assert patches.shape == (video.shape[0], grid_h, grid_w, expected_channels)


def test_patchify3d_with_pos_embed(video: torch.Tensor, patchifier_3d: Patchify3D):
    """
    Test Patchify3D with positional embedding enabled (default fixture).

    Checks that the patchified video has the expected grid and patch dimensions.
    """
    patches = patchifier_3d(video)
    grid_h = video.shape[1] // 16
    grid_w = video.shape[2] // 16
    expected_channels = 16 * 16 * video.shape[3]
    assert patches.shape == (video.shape[0], grid_h, grid_w, expected_channels)


def test_patchify3d_invalid_temporal(patchifier_3d: Patchify3D):
    """Verify that Patchify3D raises an assertion when the temporal dimension is not divisible by the tubelet size."""
    # time dimension is not divislbe by tubelet size (15 % 2 != 0)
    bad_video = torch.randn(15, 224, 224, 3)
    with pytest.raises(AssertionError):
        patchifier_3d(bad_video)


# -----------------------------------------------------------------------------
# TUBEMASK TESTS
# -----------------------------------------------------------------------------


def test_tube_mask_on_patchified_video(tube_mask_half_1x1: TubeMask):
    """
    Test TubeMask on a patchified video (grid format).

    Checks that for each frame in a video (T, 14, 14, 768),
    the kept and masked patches sum to the total number of grid patches.
    """
    T, grid_h, grid_w, channels = 16, 14, 14, 768
    patchified_video = torch.randn(T, grid_h, grid_w, channels)
    kept, mask_keep, masked, mask_discard = tube_mask_half_1x1(patchified_video)
    total_patches = grid_h * grid_w
    assert kept.shape[0] == T
    assert masked.shape[0] == T
    for t in range(T):
        assert kept[t].shape[0] + masked[t].shape[0] == total_patches
        assert kept[t].shape[1] == channels
        assert masked[t].shape[1] == channels


def test_tube_mask_on_patchified_image(tube_mask_half_1x1: TubeMask):
    """
    Test TubeMask on a patchified image (grid format).

    Simulates a patchified image of shape (14, 14, 768) and verifies that
    the number of kept and masked patches sums to the total grid size.
    Note that TubeMasking assumes inputs are videos, but will cast an
    image-like into a video with temporal dimension of 1.
    """
    grid_h, grid_w, channels = 14, 14, 768
    patchified_image = torch.randn(grid_h, grid_w, channels)
    kept, mask_keep, masked, mask_discard = tube_mask_half_1x1(patchified_image)
    total_patches = grid_h * grid_w
    # for an image input, outputs are squeezed to 2D tensors (N, C)
    assert kept.ndim == 2
    assert masked.ndim == 2
    assert kept.shape[0] + masked.shape[0] == total_patches
    assert kept.shape[1] == channels
    assert masked.shape[1] == channels


def test_tube_mask_invalid_dimensions(tube_mask_half_1x1: TubeMask):
    """
    Test TubeMask on invalid input (not enough dims, too many dims).

    Checks that TubeMask raises an AssertionError.
    """
    bad_input = torch.randn(224, 224)
    with pytest.raises(AssertionError):
        tube_mask_half_1x1(bad_input)
    bad_input = torch.randn(16, 224, 224, 3, 1)
    with pytest.raises(AssertionError):
        tube_mask_half_1x1(bad_input)


# -----------------------------------------------------------------------------
# MULTIBLOCK3DMASK TESTS
# -----------------------------------------------------------------------------
def test_multiblock3d_mask_on_patchified_video(multi_block_3d_mask: MultiBlock3DMask):
    """
    Test MultiBlock3DMask on a patchified video (grid format).

    Verifies that for each frame, the sum of kept and masked patches equals
    the total number of grid patches.
    """
    T, grid_h, grid_w, channels = 16, 14, 14, 768
    patchified_video = torch.randn(T, grid_h, grid_w, channels)
    kept, mask_keep, masked, mask_discard = multi_block_3d_mask(patchified_video)
    total_patches = grid_h * grid_w
    assert kept.shape[0] == T
    assert masked.shape[0] == T
    for t in range(T):
        assert kept[t].shape[0] + masked[t].shape[0] == total_patches
        assert kept[t].shape[1] == channels
        assert masked[t].shape[1] == channels


def test_multiblock3d_mask_on_patchified_image(multi_block_3d_mask: MultiBlock3DMask):
    """
    Test MultiBlock3DMask on a patchified image (grid format).

    Ensures that for a single image, the outputs (kept and masked) are 2D and
    their combined patches match the grid size.
    """
    grid_h, grid_w, channels = 14, 14, 768
    patchified_image = torch.randn(grid_h, grid_w, channels)
    kept, mask_keep, masked, mask_discard = multi_block_3d_mask(patchified_image)
    total_patches = grid_h * grid_w
    assert kept.ndim == 2
    assert masked.ndim == 2
    assert kept.shape[0] + masked.shape[0] == total_patches
    assert kept.shape[1] == channels
    assert masked.shape[1] == channels


def test_multiblock3d_mask_invalid_dimensions(multi_block_3d_mask: MultiBlock3DMask):
    """
    Test MultiBlock3DMask on invalid input (not enough dims, too many dims).

    Checks that MultiBlock3DMask raises an AssertionError.
    """
    bad_input = torch.randn(224, 224)
    with pytest.raises(AssertionError):
        multi_block_3d_mask(bad_input)
    bad_input = torch.randn(16, 224, 224, 3, 1)
    with pytest.raises(AssertionError):
        multi_block_3d_mask(bad_input)
