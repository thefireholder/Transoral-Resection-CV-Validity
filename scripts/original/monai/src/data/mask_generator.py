"""Generate segmentation masks from polygon annotations."""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from tqdm import tqdm

from .via_parser import FrameAnnotations


class MaskGenerator:
    """Generate segmentation masks from polygon annotations."""

    def __init__(self, frame_size: Tuple[int, int], num_classes: int = 7):
        """
        Args:
            frame_size: (width, height) of the frames
            num_classes: Number of classes including background
        """
        self.frame_size = frame_size  # (width, height)
        self.num_classes = num_classes

    def generate_mask(self, frame_annotations: FrameAnnotations) -> np.ndarray:
        """Generate a single segmentation mask from frame annotations.

        Args:
            frame_annotations: Annotations for a single frame

        Returns:
            Mask array of shape (height, width) with class IDs as values
        """
        # Create empty mask (background = 0)
        height, width = self.frame_size[1], self.frame_size[0]
        mask = np.zeros((height, width), dtype=np.uint8)

        # Draw polygons (later polygons overwrite earlier ones)
        for polygon in frame_annotations.polygons:
            # Convert vertices to integer points for cv2.fillPoly
            points = polygon.vertices.astype(np.int32)

            # Clip points to image bounds
            points[:, 0] = np.clip(points[:, 0], 0, width - 1)
            points[:, 1] = np.clip(points[:, 1], 0, height - 1)

            # Fill polygon with class ID
            cv2.fillPoly(mask, [points], polygon.class_id)

        return mask

    def generate_masks(
        self,
        frames: List[FrameAnnotations],
        output_dir: Path,
        show_progress: bool = True
    ) -> List[Path]:
        """Generate masks for all frames and save to output directory.

        Args:
            frames: List of frame annotations
            output_dir: Directory to save masks
            show_progress: Show progress bar

        Returns:
            List of paths to saved masks
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = []

        iterator = tqdm(frames, desc="Generating masks") if show_progress else frames

        for frame in iterator:
            mask = self.generate_mask(frame)

            # Save mask as PNG (values 0-6, so 8-bit is fine)
            output_path = output_dir / f"{frame.frame_id}.png"
            cv2.imwrite(str(output_path), mask)
            saved_paths.append(output_path)

        return saved_paths

    def create_visualization(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """Create a visualization overlay of mask on image.

        Args:
            image: Original image (H, W, 3) in RGB
            mask: Segmentation mask (H, W) with class IDs
            alpha: Transparency for overlay

        Returns:
            Visualization image (H, W, 3) in RGB
        """
        # Color palette for classes
        colors = [
            (0, 0, 0),        # 0: Background - black
            (255, 0, 0),      # 1: Maryland - red
            (0, 255, 0),      # 2: Cauterizer - green
            (0, 0, 255),      # 3: Tube - blue
            (255, 255, 0),    # 4: Suction - yellow
            (255, 0, 255),    # 5: Retractor - magenta
            (0, 255, 255),    # 6: Needle Driver - cyan
        ]

        # Create colored mask
        colored_mask = np.zeros_like(image)
        for class_id, color in enumerate(colors):
            colored_mask[mask == class_id] = color

        # Blend with original image
        result = image.copy()
        foreground = mask > 0  # Only blend non-background
        result[foreground] = (
            (1 - alpha) * image[foreground] + alpha * colored_mask[foreground]
        ).astype(np.uint8)

        return result


def get_mask_statistics(mask: np.ndarray, num_classes: int = 7) -> dict:
    """Get pixel count statistics for a mask."""
    stats = {}
    total = mask.size
    for class_id in range(num_classes):
        count = np.sum(mask == class_id)
        stats[class_id] = {
            'count': int(count),
            'percentage': float(count / total * 100)
        }
    return stats
