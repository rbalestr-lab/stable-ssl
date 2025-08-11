import math

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap


def _sample_block_size(
    height: int,
    width: int,
    min_scale: float,
    max_scale: float,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
):
    """Sample a single block mask for an image.

    Args:
        height (int): Height of the image in patches.
        width (int): Width of the image in patches.
        min_scale (float): Minimum scale factor for block area relative to total image area.
        max_scale (float): Maximum scale factor for block area relative to total image area.
        min_aspect_ratio (float): Minimum aspect ratio (height/width) for the block.
        max_aspect_ratio (float): Maximum aspect ratio (height/width) for the block.

    Returns:
        tuple[int, int]: A tuple (h, w) containing the sampled block height and width.
    """
    _rand = torch.rand(1).item()
    mask_scale = min_scale + _rand * (max_scale - min_scale)
    max_keep = int(height * width * mask_scale)
    aspect_ratio = min_aspect_ratio + _rand * (max_aspect_ratio - min_aspect_ratio)

    # Compute block height and width (given scale and aspect-ratio)
    h = int(round(math.sqrt(max_keep * aspect_ratio)))
    h = min(h, height - 1)

    w = int(round(math.sqrt(max_keep / aspect_ratio)))
    w = min(w, width - 1)

    return (h, w)


def _sample_block_mask(
    image_size: tuple[int, int],
    block_size: tuple[int, int],
    min_keep: int = 1,
):
    """Sample a single block mask for an image.
    Because mask positions are chosen randomly, we can occasionally end up with a mask that is too small.
    This function will retry until a valid mask is found.

    Args:
        image_size: Tuple[int, int] - Size of the image in patches
        block_size: Tuple[int, int] - Size of the block in patches
        min_keep (int): Minimum number of patches that must be in the block.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - mask: Binary tensor indices of patches exposed to encoder (1 = visible, 0 = masked)
            - pred_mask: Binary tensor where of combined target block masks to be predicted (1 = visible, 0 = masked)
    """
    h, w = block_size
    height, width = image_size
    max_attempts = 20

    for _ in range(max_attempts):
        top = torch.randint(0, height - h + 1, (1,)).item()
        left = torch.randint(0, width - w + 1, (1,)).item()

        mask = torch.zeros((height, width), dtype=torch.int32)
        mask[top : top + h, left : left + w] = 1

        # Return the first mask that satisfies min_keep.
        if torch.sum(mask) >= min_keep:
            return mask

    # If we run out of attempts, return whatever we had last.
    else:
        return mask


def multi_block_mask(
    height: int,
    width: int,
    block_scales: list[tuple[float, float]] = [(0.85, 1.0), *((0.15, 0.2),) * 4],
    aspect_ratios: list[tuple[float, float]] = [(1.0, 1.0), *((0.75, 1.5),) * 4],
    min_keep: int = 1,
    seed: int = 0,
) -> list[torch.Tensor, ...]:
    g = torch.Generator()
    g.manual_seed(seed)

    # mapping from unique combinations of size x aspect ratio to the block size.
    block_scale_to_size = {
        (scale, ratio): _sample_block_size(
            height, width, scale[0], scale[1], ratio[0], ratio[1]
        )
        for scale, ratio in set(zip(block_scales, aspect_ratios))
    }

    masks: list[torch.Tensor] = [
        _sample_block_mask(
            (height, width), block_scale_to_size[((sh, sw), (ah, aw))], min_keep
        )
        for (sh, sw), (ah, aw) in zip(block_scales, aspect_ratios)
    ]
    # -- Return binary masks
    return masks


def visualize_masking_strategy(
    height=14, width=14, num_examples=6, save_path="ijepa_masking_visualization.png"
):
    """Visualize the I-JEPA masking strategy with multiple examples.

    Args:
        height: Image height in patches
        width: Image width in patches
        num_examples: Number of masking examples to show
        save_path: Path to save the visualization
    """
    # Each example shows: context + 4 target blocks = 5 columns total
    fig, axes = plt.subplots(num_examples, 5, figsize=(15, 3 * num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, 5)

    # Set random seed for reproducible examples
    torch.manual_seed(42)

    for i in range(num_examples):
        # Generate masks - returns (context, combined_targets, individual_targets)
        cleaned_context_mask, individual_target_masks = multi_block_mask(
            height,
            width,
            num_blocks=4,
            context_scale=(0.85, 1.0),
            target_scale=(0.15, 0.2),
            aspect_ratio=(0.75, 1.5),
        )

        # Convert to numpy for visualization
        context_np = cleaned_context_mask.numpy()

        # Column 0: Context mask only
        ax = axes[i, 0]
        cmap_context = ListedColormap(["white", "lightblue"])
        ax.imshow(context_np, cmap=cmap_context, vmin=0, vmax=1)
        ax.set_title("Context" if i == 0 else "", fontsize=10)
        ax.set_xticks(range(0, width, 4))
        ax.set_yticks(range(0, height, 4))
        ax.grid(True, alpha=0.3)

        # Add grid lines for patches
        for x in range(width + 1):
            ax.axvline(x - 0.5, color="gray", linewidth=0.5, alpha=0.5)
        for y in range(height + 1):
            ax.axhline(y - 0.5, color="gray", linewidth=0.5, alpha=0.5)

        # Add row label
        if i == 0:
            ax.set_ylabel("Example 1", fontsize=12, rotation=0, ha="right", va="center")
        else:
            ax.set_ylabel(
                f"Example {i + 1}", fontsize=12, rotation=0, ha="right", va="center"
            )

        # Columns 1-4: Individual target masks
        target_colors = ["red", "orange", "green", "purple"]
        for j, target_mask in enumerate(individual_target_masks):
            ax = axes[i, j + 1]
            target_np = target_mask.numpy()

            # Create colormap for this target
            cmap_target = ListedColormap(["white", target_colors[j]])
            ax.imshow(target_np, cmap=cmap_target, vmin=0, vmax=1)
            ax.set_title(f"Target {j + 1}" if i == 0 else "", fontsize=10)
            ax.set_xticks(range(0, width, 4))
            ax.set_yticks(range(0, height, 4))
            ax.grid(True, alpha=0.3)

            # Add grid lines for patches
            for x in range(width + 1):
                ax.axvline(x - 0.5, color="gray", linewidth=0.5, alpha=0.5)
            for y in range(height + 1):
                ax.axhline(y - 0.5, color="gray", linewidth=0.5, alpha=0.5)

    # Add legend
    legend_elements = [
        patches.Patch(color="white", label="Visible patches"),
        patches.Patch(color="lightblue", label="Context block"),
        patches.Patch(color="red", label="Target block 1"),
        patches.Patch(color="orange", label="Target block 2"),
        patches.Patch(color="green", label="Target block 3"),
        patches.Patch(color="purple", label="Target block 4"),
    ]

    fig.legend(
        handles=legend_elements,
        loc="center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=6,
        fontsize=10,
    )

    plt.suptitle(
        "I-JEPA Masking Strategy: Context and Individual Target Blocks",
        fontsize=14,
        y=0.95,
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, top=0.88)

    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Visualization saved to: {save_path}")
    plt.close()


def analyze_masking_statistics(height=14, width=14, num_samples=1000):
    """Analyze statistics of the masking strategy."""
    context_scales = []
    target_scales = []
    aspect_ratios = []

    torch.manual_seed(42)

    for _ in range(num_samples):
        cleaned_context_mask, individual_target_masks = multi_block_mask(
            height,
            width,
            num_blocks=4,
            context_scale=(0.85, 1.0),
            target_scale=(0.15, 0.2),
            aspect_ratio=(0.75, 1.5),
        )

        total_patches = height * width
        context_scale = torch.sum(cleaned_context_mask).item() / total_patches
        context_scales.append(context_scale)

        for target_mask in individual_target_masks:
            target_scale = torch.sum(target_mask).item() / total_patches
            target_scales.append(target_scale)

            # Calculate aspect ratio of target
            coords = torch.where(target_mask == 1)
            if len(coords[0]) > 0:
                h_extent = coords[0].max() - coords[0].min() + 1
                w_extent = coords[1].max() - coords[1].min() + 1
                aspect_ratios.append(h_extent.item() / w_extent.item())

    # Create statistics plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(
        context_scales, bins=30, alpha=0.7, color="lightblue", edgecolor="black"
    )
    axes[0].set_title("Context Block Scales")
    axes[0].set_xlabel("Scale (fraction of image)")
    axes[0].set_ylabel("Frequency")
    axes[0].axvline(
        np.mean(context_scales),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(context_scales):.3f}",
    )
    axes[0].legend()

    axes[1].hist(target_scales, bins=30, alpha=0.7, color="orange", edgecolor="black")
    axes[1].set_title("Target Block Scales")
    axes[1].set_xlabel("Scale (fraction of image)")
    axes[1].set_ylabel("Frequency")
    axes[1].axvline(
        np.mean(target_scales),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(target_scales):.3f}",
    )
    axes[1].legend()

    axes[2].hist(aspect_ratios, bins=30, alpha=0.7, color="green", edgecolor="black")
    axes[2].set_title("Target Block Aspect Ratios")
    axes[2].set_xlabel("Aspect Ratio (height/width)")
    axes[2].set_ylabel("Frequency")
    axes[2].axvline(
        np.mean(aspect_ratios),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(aspect_ratios):.3f}",
    )
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("ijepa_masking_statistics.png", dpi=300, bbox_inches="tight")
    print("Statistics saved to: ijepa_masking_statistics.png")
    plt.close()

    return {
        "context_scales": context_scales,
        "target_scales": target_scales,
        "aspect_ratios": aspect_ratios,
    }


if __name__ == "__main__":
    print("Generating I-JEPA masking visualizations...")

    # Create main visualization
    visualize_masking_strategy(
        height=14, width=14, num_examples=8, save_path="ijepa_masking_examples.png"
    )

    # Generate statistics
    stats = analyze_masking_statistics()

    print("\nMasking Statistics:")
    print(
        f"Context scale - Mean: {np.mean(stats['context_scales']):.3f}, Std: {np.std(stats['context_scales']):.3f}"
    )
    print(
        f"Target scale - Mean: {np.mean(stats['target_scales']):.3f}, Std: {np.std(stats['target_scales']):.3f}"
    )
    print(
        f"Aspect ratio - Mean: {np.mean(stats['aspect_ratios']):.3f}, Std: {np.std(stats['aspect_ratios']):.3f}"
    )

    print("\nVisualization complete! Check the generated PNG files.")
