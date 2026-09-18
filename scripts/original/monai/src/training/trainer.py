"""Training loop for segmentation model."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Dict, Optional, Tuple
from tqdm import tqdm
import time

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import TrainingConfig
from src.training.metrics import SegmentationMetrics
from src.utils.device import to_device


def _extract_class_weights(criterion):
    """Best-effort extraction of class weights from a DiceFocalLoss-like object."""
    focal = getattr(criterion, "focal_loss", None)
    if focal is None:
        return None
    weight = getattr(focal, "weight", None)
    if weight is None or not isinstance(weight, torch.Tensor):
        return None
    return weight.detach().cpu().tolist()


class EarlyStopping:
    """Early stopping handler."""

    def __init__(self, patience: int = 15, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        """Check if training should stop.

        Args:
            score: Current validation score (higher is better)

        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.should_stop


class Trainer:
    """Training handler for segmentation model."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: torch.device = torch.device("cpu"),
        config: Optional[TrainingConfig] = None,
        checkpoint_dir: Optional[Path] = None,
        log_dir: Optional[Path] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config or TrainingConfig()

        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir

        # Initialize
        self.model.to(device)
        self.best_dice = 0.0
        self.current_epoch = 0

        # TensorBoard writer
        self.writer = None
        if log_dir is not None:
            self.writer = SummaryWriter(log_dir=str(log_dir))

        # Metrics
        self.metrics = SegmentationMetrics(num_classes=7, include_background=False)

        # Early stopping
        self.early_stopping = EarlyStopping(patience=self.config.early_stopping_patience)

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")

        for batch in pbar:
            # Move to device
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping for MPS stability
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )

            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / num_batches

        return {"train_loss": avg_loss}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation.

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        self.metrics.reset()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]")

        for batch in pbar:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item()
            num_batches += 1

            # Update metrics
            self.metrics.update(outputs, labels)

        avg_loss = total_loss / num_batches
        metric_results = self.metrics.compute()

        results = {"val_loss": avg_loss}
        results.update(metric_results)

        return results

    def train(self, num_epochs: Optional[int] = None) -> Dict[str, float]:
        """Run full training loop.

        Args:
            num_epochs: Number of epochs to train (overrides config)

        Returns:
            Best validation metrics
        """
        epochs = num_epochs or self.config.num_epochs
        best_metrics = {}

        print(f"Starting training for {epochs} epochs on {self.device}")
        print(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")

        for epoch in range(epochs):
            self.current_epoch = epoch
            start_time = time.time()

            # Training
            train_metrics = self.train_epoch()

            # Validation
            val_metrics = self.validate()

            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            epoch_time = time.time() - start_time

            # Log metrics
            self._log_metrics(train_metrics, val_metrics, epoch)

            # Print summary
            print(f"\nEpoch {epoch + 1}/{epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"  Val Dice: {val_metrics['mean_dice']:.4f}")
            print(f"  Val IoU: {val_metrics['mean_iou']:.4f}")

            # Save best model
            if val_metrics['mean_dice'] > self.best_dice:
                self.best_dice = val_metrics['mean_dice']
                best_metrics = val_metrics.copy()
                self._save_checkpoint("best_model.pth", val_metrics)
                print(f"  -> New best model (Dice: {self.best_dice:.4f})")

            # Early stopping
            if self.early_stopping(val_metrics['mean_dice']):
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        # Save final model
        self._save_checkpoint("final_model.pth", val_metrics)

        if self.writer is not None:
            self.writer.close()

        return best_metrics

    def _log_metrics(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch: int
    ):
        """Log metrics to TensorBoard."""
        if self.writer is None:
            return

        # Training metrics
        for key, value in train_metrics.items():
            self.writer.add_scalar(f"train/{key}", value, epoch)

        # Validation metrics
        for key, value in val_metrics.items():
            self.writer.add_scalar(f"val/{key}", value, epoch)

        # Learning rate
        if self.scheduler is not None:
            lr = self.scheduler.get_last_lr()[0]
            self.writer.add_scalar("train/learning_rate", lr, epoch)

    def _save_checkpoint(self, filename: str, metrics: Dict[str, float]):
        """Save model checkpoint."""
        if self.checkpoint_dir is None:
            return

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "best_dice": self.best_dice,
            # Training hyperparameters that produced this checkpoint — pinning
            # these is what makes a checkpoint self-describing so future readers
            # know what loss/gamma/weights were active.
            "training_config": {
                "focal_gamma": self.config.focal_gamma,
                "dice_weight": self.config.dice_weight,
                "focal_weight": self.config.focal_weight,
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "batch_size": self.config.batch_size,
            },
            # Resolved class weights (after data-derivation + boosts)
            "class_weights": _extract_class_weights(self.criterion),
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: Path):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint.get("epoch", 0)
        self.best_dice = checkpoint.get("best_dice", 0.0)

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print(f"Loaded checkpoint from epoch {self.current_epoch}, best dice: {self.best_dice:.4f}")
