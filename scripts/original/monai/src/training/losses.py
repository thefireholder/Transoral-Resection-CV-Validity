"""Loss functions for segmentation training."""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from monai.losses import DiceLoss, FocalLoss

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import ClassConfig, AnatomyClassConfig, JML619ClassConfig

# The instrument and anatomy class configs share the structural interface used
# by the loss/weight helpers below (num_classes, class_to_id, id_to_class,
# weight_multipliers, background_weight). Aliasing them here keeps call sites
# clear without forcing a Protocol abstraction.
AnyClassConfig = Union[ClassConfig, AnatomyClassConfig, JML619ClassConfig]


def get_class_weights(config: Optional[ClassConfig] = None) -> torch.Tensor:
    """Get hardcoded class weights tensor (fallback when no masks available)."""
    cfg = config or ClassConfig()
    return torch.tensor(cfg.class_weights, dtype=torch.float32)


def compute_class_weights_from_masks(
    masks_dir: Path,
    config: Optional[AnyClassConfig] = None,
    max_weight: float = 15.0,
) -> Tuple[torch.Tensor, dict]:
    """Compute inverse-frequency class weights from the actual mask distribution.

    The result is normalized so the median non-background weight is 1.0, then
    the per-class multipliers from `config.weight_multipliers` are applied,
    background is clamped to `config.background_weight`, and finally any class
    weight is clipped to `max_weight` to keep training stable. Pathologically
    rare classes (a handful of polygons) get astronomical inverse-frequency
    weights that destabilize gradients — clipping is the standard guard.

    Returns:
        (weights tensor, stats dict with 'pixels', 'percent', 'raw_weights' arrays)
    """
    cfg = config or ClassConfig()
    n = cfg.num_classes
    mask_files = sorted(Path(masks_dir).glob("*.png"))
    if not mask_files:
        raise FileNotFoundError(f"No masks found in {masks_dir}")

    counts = np.zeros(n, dtype=np.int64)
    total = 0
    for mp in mask_files:
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise RuntimeError(f"Failed to read mask: {mp}")
        for c in range(n):
            counts[c] += int(np.sum(m == c))
        total += m.size

    freq = counts / total
    safe_freq = np.where(freq == 0, 1e-12, freq)
    inv = 1.0 / safe_freq

    # Normalize so the median non-background weight is 1.0
    median_nonbg = np.median(inv[1:]) if n > 1 else 1.0
    weights = inv / max(median_nonbg, 1e-12)

    # Apply per-class boost multipliers from config
    for class_name, mult in cfg.weight_multipliers.items():
        cid = cfg.class_to_id.get(class_name)
        if cid is None:
            print(f"Warning: weight_multipliers references unknown class '{class_name}'")
            continue
        weights[cid] *= mult

    # Clamp background
    weights[0] = cfg.background_weight

    # Clip any non-background weight to a sane maximum (extremely rare classes
    # otherwise produce 40x+ weights that blow up training).
    if max_weight > 0:
        weights[1:] = np.clip(weights[1:], a_min=None, a_max=max_weight)

    return torch.tensor(weights, dtype=torch.float32), {
        "pixels": counts,
        "percent": freq * 100,
        "raw_weights": inv / max(median_nonbg, 1e-12),
        "num_masks": len(mask_files),
    }


def print_class_weight_report(
    weights: torch.Tensor,
    stats: dict,
    config: Optional[AnyClassConfig] = None,
) -> None:
    """Pretty-print the class distribution and final weights training will use."""
    cfg = config or ClassConfig()
    print(f"\nClass distribution over {stats['num_masks']} masks:")
    print(f"  {'Class':<15} {'Pixels':>14} {'Percent':>9} {'Raw wt':>8} {'Final wt':>10}")
    print("  " + "-" * 60)
    for cid in range(cfg.num_classes):
        name = cfg.id_to_class[cid]
        marker = ""
        raw = stats["raw_weights"][cid]
        final = float(weights[cid])
        if name in cfg.weight_multipliers:
            marker = f" (x{cfg.weight_multipliers[name]} boost)"
        elif cid == 0:
            marker = " (clamped to background_weight)"
        elif raw > final + 1e-6:
            marker = f" (clipped from {raw:.2f})"
        print(
            f"  {name:<15} {stats['pixels'][cid]:>14,} "
            f"{stats['percent'][cid]:>8.4f}% "
            f"{raw:>8.3f} "
            f"{final:>10.3f}{marker}"
        )


class DiceFocalLoss(nn.Module):
    """Combined Dice and Focal loss for handling class imbalance."""

    def __init__(
        self,
        num_classes: int = 7,
        class_weights: Optional[torch.Tensor] = None,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        focal_gamma: float = 2.0,
        smooth: float = 1e-5
    ):
        """
        Args:
            num_classes: Number of classes
            class_weights: Tensor of class weights for focal loss
            dice_weight: Weight for Dice loss component
            focal_weight: Weight for Focal loss component
            focal_gamma: Gamma parameter for focal loss
            smooth: Smoothing factor for Dice loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

        # MONAI Dice loss
        self.dice_loss = DiceLoss(
            include_background=True,
            to_onehot_y=True,
            softmax=True,
            reduction="mean",
            smooth_nr=smooth,
            smooth_dr=smooth
        )

        # MONAI Focal loss
        self.focal_loss = FocalLoss(
            include_background=True,
            to_onehot_y=True,
            gamma=focal_gamma,
            weight=class_weights,
            reduction="mean"
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute combined loss.

        Args:
            logits: Model output of shape (B, C, H, W)
            targets: Ground truth of shape (B, 1, H, W) with class indices

        Returns:
            Combined loss value
        """
        dice = self.dice_loss(logits, targets)
        focal = self.focal_loss(logits, targets)

        return self.dice_weight * dice + self.focal_weight * focal


class WeightedCrossEntropyLoss(nn.Module):
    """Weighted cross-entropy loss for class imbalance."""

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = -100
    ):
        super().__init__()
        self.class_weights = class_weights
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss.

        Args:
            logits: Model output of shape (B, C, H, W)
            targets: Ground truth of shape (B, 1, H, W) with class indices

        Returns:
            Loss value
        """
        # Remove channel dimension from targets
        targets = targets.squeeze(1).long()

        # Move weights to same device as logits
        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)

        return F.cross_entropy(
            logits,
            targets,
            weight=weight,
            ignore_index=self.ignore_index
        )
