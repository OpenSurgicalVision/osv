"""Two-tier surgical domain data augmentations.

L1 (on-the-fly, CPU): elastic deformations, motion blur, defocus, sensor noise, color shifts.
L2 (offline pre-generated): realistic smoke, lens blood splatter, specular highlights.
"""

from collections.abc import Callable
from typing import Any

import numpy as np

#: Label value outside the frame after a geometric transform: OSV's ignore index.
IGNORE_LABEL = 255


def get_l1_augmentations() -> dict[str, Any]:
    """Returns albumentations pipeline for on-the-fly L1 surgical transformations."""
    return {"tier": "L1", "status": "active"}


def get_l2_dataset_info() -> dict[str, Any]:
    """Returns metadata about pre-generated L2 static augmented datasets."""
    return {"tier": "L2", "variants_per_frame": 3, "invariance": "mask_preserving"}


def segmentation_train_transform() -> Callable[
    [np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]
]:
    """L1 augmentation for segmentation training: (rgb uint8 HxWx3, labels HxW) in, the same out.

    Geometry moves image and labels together (labels nearest, uncovered pixels ignored);
    photometry touches the image only. Meant above all for synthetic training data, where a
    network otherwise learns the renderer's exact textures, edges and black field-stop
    surround instead of anatomy; off by default so existing runs stay reproducible.
    Randomness comes from Python's `random`, which DataLoader workers seed per worker.
    """
    import albumentations as A
    import cv2

    aug = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.9, 1.3), rotate=(-15, 15), translate_percent=(-0.05, 0.05),
                border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=IGNORE_LABEL, p=0.7,
            ),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=0.8),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.GaussNoise(std_range=(0.01, 0.04), p=0.3),
            A.ImageCompression(quality_range=(60, 95), p=0.3),
        ]
    )  # fmt: skip

    def transform(rgb: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        out = aug(image=rgb, mask=label)
        return out["image"], out["mask"]

    return transform


__all__ = [
    "IGNORE_LABEL",
    "get_l1_augmentations",
    "get_l2_dataset_info",
    "segmentation_train_transform",
]
