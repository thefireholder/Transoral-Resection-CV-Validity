"""MONAI-compatible dataset for surgical instrument segmentation."""
import json
import numpy as np
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import DataConfig


class InstrumentDataset(Dataset):
    """Dataset for surgical instrument segmentation."""

    def __init__(
        self,
        image_paths: List[Path],
        mask_paths: List[Path],
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (512, 512)
    ):
        """
        Args:
            image_paths: Paths to frame images
            mask_paths: Paths to mask images (must match image_paths order)
            transform: MONAI transforms to apply
            target_size: (height, width) to resize images to
        """
        assert len(image_paths) == len(mask_paths), \
            f"Mismatch: {len(image_paths)} images, {len(mask_paths)} masks"

        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        # Load image (BGR -> RGB)
        image = cv2.imread(str(self.image_paths[index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask
        mask = cv2.imread(str(self.mask_paths[index]), cv2.IMREAD_GRAYSCALE)

        # Resize
        image = cv2.resize(image, (self.target_size[1], self.target_size[0]))
        mask = cv2.resize(mask, (self.target_size[1], self.target_size[0]),
                          interpolation=cv2.INTER_NEAREST)

        # Convert to tensor format
        # Image: (H, W, C) -> (C, H, W), normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        # Mask: (H, W) -> (1, H, W)
        mask = mask.astype(np.int64)
        mask = np.expand_dims(mask, axis=0)

        data = {
            "image": torch.from_numpy(image),
            "label": torch.from_numpy(mask),
        }

        if self.transform is not None:
            data = self.transform(data)

        return data


def get_image_mask_pairs(
    frames_dir: Path,
    masks_dir: Path,
    image_glob: str = "frame_*.png",
) -> Tuple[List[Path], List[Path]]:
    """Get matched image and mask file paths.

    Mask files are always .png (single-channel class IDs). Frame extension is
    configurable via image_glob — the instrument pipeline uses .png frames, the
    anatomy pipeline uses .jpg frames (because annotations reference .jpg files).
    """
    image_paths = sorted(frames_dir.glob(image_glob))
    pairs: List[Tuple[Path, Path]] = []
    for image_path in image_paths:
        mask_path = masks_dir / f"{image_path.stem}.png"
        if mask_path.exists():
            pairs.append((image_path, mask_path))

    pairs.sort(key=lambda x: x[0].name)
    return [p[0] for p in pairs], [p[1] for p in pairs]


def create_data_loaders_from_split(
    frames_dir: Path,
    masks_dir: Path,
    split_file: Path,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    data_config: Optional[DataConfig] = None,
    batch_size: int = 4,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test loaders from an explicit split file.

    The split file maps split name -> list of frame paths; only the basename
    is used, resolved against frames_dir (the team's split files use their own
    directory prefixes like "JML619-images/"). Frames whose mask is missing
    are dropped with a warning — unlike the random-split path, an explicit
    split should not silently shrink.

    Returns (train_loader, val_loader, test_loader).
    """
    config = data_config or DataConfig()
    with open(split_file) as f:
        split = json.load(f)

    def build(name: str, transform: Optional[Callable], shuffle: bool,
              drop_last: bool) -> DataLoader:
        images, masks, missing = [], [], []
        for entry in split[name]:
            image_path = frames_dir / Path(entry).name
            mask_path = masks_dir / f"{image_path.stem}.png"
            if image_path.exists() and mask_path.exists():
                images.append(image_path)
                masks.append(mask_path)
            else:
                missing.append(entry)
        if missing:
            print(f"Warning: {name} split lists {len(missing)} frames with "
                  f"no image/mask on disk (e.g. {missing[0]})")
        print(f"{name}: {len(images)} frames")
        dataset = InstrumentDataset(
            images, masks, transform=transform, target_size=config.image_size
        )
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, drop_last=drop_last,
        )

    return (
        build("train", train_transform, shuffle=True, drop_last=True),
        build("val", val_transform, shuffle=False, drop_last=False),
        build("test", val_transform, shuffle=False, drop_last=False),
    )


def create_data_loaders(
    frames_dir: Path,
    masks_dir: Path,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    data_config: Optional[DataConfig] = None,
    batch_size: int = 4,
    num_workers: int = 0,
    image_glob: str = "frame_*.png",
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders.

    Args:
        frames_dir: Directory containing frame images
        masks_dir: Directory containing mask images
        train_transform: Transforms for training data
        val_transform: Transforms for validation data
        data_config: Data configuration
        batch_size: Batch size
        num_workers: Number of data loading workers

    Returns:
        (train_loader, val_loader)
    """
    config = data_config or DataConfig()

    # Get matched pairs
    image_paths, mask_paths = get_image_mask_pairs(frames_dir, masks_dir, image_glob=image_glob)
    print(f"Found {len(image_paths)} matched image-mask pairs")

    # Train/val split
    train_images, val_images, train_masks, val_masks = train_test_split(
        image_paths,
        mask_paths,
        train_size=config.train_split,
        random_state=config.random_seed
    )

    print(f"Train: {len(train_images)}, Val: {len(val_images)}")

    # Create datasets
    train_dataset = InstrumentDataset(
        train_images,
        train_masks,
        transform=train_transform,
        target_size=config.image_size
    )

    val_dataset = InstrumentDataset(
        val_images,
        val_masks,
        transform=val_transform,
        target_size=config.image_size
    )

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader
