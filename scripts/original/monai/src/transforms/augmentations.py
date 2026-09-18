"""Data augmentation transforms using MONAI."""
from monai.transforms import (
    Compose,
    RandFlipd,
    RandRotate90d,
    RandAffined,
    RandGaussianNoised,
    RandAdjustContrastd,
    EnsureTyped,
)


def get_train_transforms() -> Compose:
    """Get training data augmentation transforms.

    Returns:
        MONAI Compose transform for training
    """
    return Compose([
        # Random horizontal flip
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=1,
            prob=0.5
        ),
        # Random vertical flip
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=0,
            prob=0.5
        ),
        # Random 90-degree rotation
        RandRotate90d(
            keys=["image", "label"],
            prob=0.5,
            max_k=3
        ),
        # Random affine (rotation, scale, shear)
        RandAffined(
            keys=["image", "label"],
            prob=0.5,
            rotate_range=[0.26],  # ~15 degrees
            scale_range=[0.1, 0.1],
            shear_range=[0.1, 0.1],
            mode=["bilinear", "nearest"],
            padding_mode="zeros"
        ),
        # Gaussian noise (image only)
        RandGaussianNoised(
            keys=["image"],
            prob=0.3,
            mean=0.0,
            std=0.05
        ),
        # Contrast adjustment (image only)
        RandAdjustContrastd(
            keys=["image"],
            prob=0.3,
            gamma=(0.8, 1.2)
        ),
        # Ensure PyTorch tensor type
        EnsureTyped(keys=["image", "label"]),
    ])


def get_val_transforms() -> Compose:
    """Get validation transforms (no augmentation).

    Returns:
        MONAI Compose transform for validation
    """
    return Compose([
        # Only ensure tensor type, no augmentation
        EnsureTyped(keys=["image", "label"]),
    ])
