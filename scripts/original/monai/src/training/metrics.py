"""Evaluation metrics for segmentation."""
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from monai.metrics import DiceMetric

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import ClassConfig


class SegmentationMetrics:
    """Compute segmentation metrics (Dice, IoU) per class."""

    def __init__(
        self,
        num_classes: int = 7,
        class_names: Optional[List[str]] = None,
        include_background: bool = True
    ):
        """
        Args:
            num_classes: Number of classes
            class_names: Optional list of class names
            include_background: Whether to include background in metrics
        """
        self.num_classes = num_classes
        self.include_background = include_background

        if class_names is None:
            config = ClassConfig()
            self.class_names = [config.id_to_class[i] for i in range(num_classes)]
        else:
            self.class_names = class_names

        # MONAI Dice metric
        self.dice_metric = DiceMetric(
            include_background=include_background,
            reduction="mean_batch"
        )

        self.reset()

    def reset(self):
        """Reset accumulated metrics."""
        self.dice_metric.reset()
        self._intersection = torch.zeros(self.num_classes)
        self._union = torch.zeros(self.num_classes)
        self._total_pixels = torch.zeros(self.num_classes)

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ):
        """Update metrics with a batch of predictions.

        Args:
            predictions: Model output logits (B, C, H, W) or labels (B, H, W)
            targets: Ground truth (B, 1, H, W) with class indices
        """
        # Get predicted labels if logits provided
        if predictions.dim() == 4 and predictions.shape[1] > 1:
            pred_labels = torch.argmax(predictions, dim=1)
        else:
            pred_labels = predictions.squeeze(1) if predictions.dim() == 4 else predictions

        target_labels = targets.squeeze(1)

        # Convert to one-hot for MONAI Dice metric
        pred_onehot = self._to_onehot(pred_labels)
        target_onehot = self._to_onehot(target_labels)

        # Update MONAI metric
        self.dice_metric(pred_onehot, target_onehot)

        # Update IoU counters
        for c in range(self.num_classes):
            pred_c = pred_labels == c
            target_c = target_labels == c

            intersection = (pred_c & target_c).sum().float()
            union = (pred_c | target_c).sum().float()

            self._intersection[c] += intersection.cpu()
            self._union[c] += union.cpu()
            self._total_pixels[c] += target_c.sum().float().cpu()

    def _to_onehot(self, labels: torch.Tensor) -> torch.Tensor:
        """Convert label tensor to one-hot encoding."""
        # labels: (B, H, W) -> (B, C, H, W)
        B, H, W = labels.shape
        onehot = torch.zeros(B, self.num_classes, H, W, device=labels.device)
        labels_long = labels.long().unsqueeze(1)
        onehot.scatter_(1, labels_long, 1)
        return onehot

    def compute(self) -> Dict[str, float]:
        """Compute final metrics.

        Returns:
            Dictionary with per-class and mean metrics
        """
        # Get Dice scores
        dice_scores = self.dice_metric.aggregate().cpu().numpy()
        if dice_scores.ndim > 1:
            dice_scores = dice_scores.mean(axis=0)

        # Compute IoU per class
        iou_scores = (self._intersection / (self._union + 1e-6)).numpy()

        results = {}

        # Per-class metrics
        start_idx = 0 if self.include_background else 1
        for i in range(start_idx, self.num_classes):
            class_name = self.class_names[i]
            dice_idx = i if self.include_background else i - 1

            if dice_idx < len(dice_scores):
                results[f"dice_{class_name}"] = float(dice_scores[dice_idx])
            results[f"iou_{class_name}"] = float(iou_scores[i])

        # Mean metrics (excluding background if not included)
        valid_dice = dice_scores[start_idx:] if self.include_background else dice_scores
        valid_iou = iou_scores[start_idx:]

        results["mean_dice"] = float(np.nanmean(valid_dice))
        results["mean_iou"] = float(np.nanmean(valid_iou))

        return results

    def compute_per_class_report(self) -> str:
        """Generate a formatted report of per-class metrics."""
        metrics = self.compute()

        lines = ["=" * 50]
        lines.append("Per-Class Segmentation Metrics")
        lines.append("=" * 50)
        lines.append(f"{'Class':<15} {'Dice':>10} {'IoU':>10}")
        lines.append("-" * 50)

        start_idx = 0 if self.include_background else 1
        for i in range(start_idx, self.num_classes):
            class_name = self.class_names[i]
            dice = metrics.get(f"dice_{class_name}", float('nan'))
            iou = metrics.get(f"iou_{class_name}", float('nan'))
            lines.append(f"{class_name:<15} {dice:>10.4f} {iou:>10.4f}")

        lines.append("-" * 50)
        lines.append(f"{'Mean':<15} {metrics['mean_dice']:>10.4f} {metrics['mean_iou']:>10.4f}")
        lines.append("=" * 50)

        return "\n".join(lines)


def compute_confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int
) -> np.ndarray:
    """Compute confusion matrix.

    Args:
        predictions: Predicted labels (B, H, W)
        targets: Ground truth labels (B, H, W)
        num_classes: Number of classes

    Returns:
        Confusion matrix of shape (num_classes, num_classes)
    """
    pred_flat = predictions.view(-1).cpu().numpy()
    target_flat = targets.view(-1).cpu().numpy()

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(target_flat, pred_flat):
        confusion[int(t), int(p)] += 1

    return confusion
