import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import IntSlider, interact

# flake8: noqa: E501


def display_image_cursor(img: np.ndarray, title_str: str):
    """
    Interactive slider with black background and fixed colorbar
    """
    # Calculate global min/max for consistent colorbar
    vmin, vmax = img.min(), img.max()

    @interact(
        slice_idx=IntSlider(
            min=0,
            max=img.shape[0] - 1,
            step=1,
            value=img.shape[0] // 2,
            description="Slice:",
            continuous_update=True,
        )
    )
    def show_slice(slice_idx):
        fig, ax = plt.subplots(figsize=(10, 8))

        # Display image with fixed colorbar limits
        im = ax.imshow(img[slice_idx], cmap="gray_r", vmin=vmin, vmax=vmax)

        # Create colorbar
        cbar = plt.colorbar(im, ax=ax)

        # Style the plot
        ax.set_title(f"Slice {slice_idx}/{img.shape[0]-1}", color="white", fontsize=14)

        plt.suptitle(title_str, fontsize=16, weight="bold")
        plt.tight_layout()
        plt.show()


def visualize_otsu_hist(img_np, seg_otsu, threshold):

    slice_idx = img_np.shape[0] // 2
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    img_slice = img_np[slice_idx]
    binary_slice = seg_otsu[slice_idx]

    # Original image
    axes[0].imshow(img_slice, cmap="gray")
    axes[0].set_title("Original Image", fontsize=14)

    # Histogram with threshold line
    axes[1].hist(
        img_np.ravel(), bins=256, color="tab:blue", alpha=0.7, edgecolor="none"
    )
    axes[1].axvline(
        threshold,
        color="red",
        linestyle="--",
        linewidth=3,
        label=f"Otsu Threshold = {threshold:.2f}",
    )
    axes[1].set_title("Intensity Histogram", fontsize=14)
    axes[1].set_xlabel("Intensity", color="black")
    axes[1].set_ylabel("Frequency", color="black")
    axes[1].legend(fontsize=12)
    axes[1].tick_params(colors="black")
    axes[1].spines["bottom"].set_color("black")
    axes[1].spines["left"].set_color("black")
    axes[1].spines["top"].set_visible(True)
    axes[1].spines["right"].set_visible(True)

    # Binary result - FIX THIS LINE
    axes[2].imshow(binary_slice, cmap="gray")  # Changed from img_slice to binary_slice
    axes[2].set_title("Otsu Thresholded", fontsize=14)

    plt.tight_layout()
    plt.show()

    print(f"Otsu's threshold value: {threshold:.4f}")
    print(
        f"Pixels above threshold: {np.sum(img_np > threshold):,} ({100*np.mean(img_np > threshold):.2f}%)"
    )
    print(
        f"Pixels below threshold: {np.sum(img_np <= threshold):,} ({100*np.mean(img_np <= threshold):.2f}%)"
    )
    return None
