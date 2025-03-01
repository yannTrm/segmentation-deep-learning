import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
import numpy as np

from matplotlib.colors import ListedColormap


def visualize_binaire(image: np.ndarray, mask: np.ndarray) -> None:
    """
    Plot the original image, the ground truth mask, and an overlay of the mask on the image.

    Args:
        image (np.ndarray): The original image in CHW format (C, H, W).
        mask (np.ndarray): The ground truth mask in HW format (H, W).

    Example:
        >>> visualize_binaire(image, mask)
    """
    image_rgb = image.transpose(1, 2, 0)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Ground Truth Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(image_rgb)
    plt.imshow(mask, cmap="gray", alpha=0.5)  
    plt.title("Overlay (Image + Mask)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()



def visualize_gray(image: np.ndarray, mask: np.ndarray, class_mapping: Dict[int, str], cmap_name: str = "tab20") -> None:
    """
    Plot the original image, the ground truth mask, and an overlay of the mask on the image for multi-class segmentation.
    The legend includes both the class name and its associated number, including the background (0).

    Args:
        image (np.ndarray): The original image in CHW format (C, H, W).
        mask (np.ndarray): The ground truth mask in HW format (H, W).
        class_mapping (Dict[int, str]): Mapping from class indices to class names.
        cmap_name (str): Name of the colormap to use for displaying the mask.

    Example:
        >>> visualize_gray(image, mask, class_mapping)
    """
    image_rgb = image.transpose(1, 2, 0)

    num_classes = len(class_mapping) +1  # +1 pour le background
    cmap = plt.get_cmap(cmap_name, num_classes)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    mask_display = plt.imshow(mask, cmap=cmap, vmin=0, vmax=num_classes - 1)
    plt.title("Ground Truth Mask")
    plt.axis("off")

    handles = []
    labels = []

    handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(0), markersize=10))
    labels.append("0: Background")

    for i in range(1, num_classes):
        handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i), markersize=10))
        labels.append(f"{i}: {class_mapping[i]}")

    plt.legend(handles, labels, title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.subplot(1, 3, 3)
    plt.imshow(image_rgb)
    plt.imshow(mask, cmap=cmap, alpha=0.5, vmin=0, vmax=num_classes - 1)  
    plt.title("Overlay (Image + Mask)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()