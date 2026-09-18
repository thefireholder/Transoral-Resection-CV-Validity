#!/usr/bin/env python3
"""Script 13: Train segmentation on the unified JSR + JML619 dataset.

Mirrors scripts/09_train_jml619.py with the unified class schema
(data/unified_schema.md) and the combined temporal split.
Run scripts/12_prepare_unified.py first.
"""
import sys
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import get_config, UnifiedClassConfig, ModelConfig
from src.data.dataset import create_data_loaders_from_split
from src.transforms.augmentations import get_train_transforms, get_val_transforms
from src.models.segmentation import SegmentationModel
from src.training.losses import (
    DiceFocalLoss,
    compute_class_weights_from_masks,
    print_class_weight_report,
)
from src.training.trainer import Trainer
from src.training.metrics import SegmentationMetrics
from src.utils.device import get_device, get_dataloader_kwargs


def main():
    config = get_config()
    class_cfg = UnifiedClassConfig()

    model_cfg = ModelConfig(
        spatial_dims=config.model.spatial_dims,
        in_channels=config.model.in_channels,
        out_channels=class_cfg.num_classes,
        channels=config.model.channels,
        strides=config.model.strides,
        num_res_units=config.model.num_res_units,
        dropout=config.model.dropout,
    )

    print("=" * 60)
    print("Step 13: Train Unified (JSR + JML619) Segmentation Model")
    print("=" * 60)

    device = get_device()
    print(f"\nUsing device: {device}")

    loader_kwargs = get_dataloader_kwargs(device)
    print("\nLoading data (combined temporal split)...")
    train_loader, val_loader, _test_loader = create_data_loaders_from_split(
        config.paths.unified_frames_dir,
        config.paths.unified_masks_dir,
        config.paths.unified_split_file,
        train_transform=get_train_transforms(),
        val_transform=get_val_transforms(),
        data_config=config.data,
        batch_size=config.training.batch_size,
        num_workers=loader_kwargs["num_workers"],
    )

    print("\nCreating model...")
    model = SegmentationModel(model_cfg)
    print(f"Model parameters: {model.get_num_parameters():,}")

    class_weights, weight_stats = compute_class_weights_from_masks(
        config.paths.unified_masks_dir, class_cfg
    )
    print_class_weight_report(class_weights, weight_stats, class_cfg)
    class_weights = class_weights.to(device)

    criterion = DiceFocalLoss(
        num_classes=class_cfg.num_classes,
        class_weights=class_weights,
        dice_weight=config.training.dice_weight,
        focal_weight=config.training.focal_weight,
        focal_gamma=config.training.focal_gamma,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.training.num_epochs,
        eta_min=1e-6,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config.training,
        checkpoint_dir=config.paths.unified_checkpoints_dir,
        log_dir=config.paths.unified_logs_dir,
    )
    trainer.metrics = SegmentationMetrics(
        num_classes=class_cfg.num_classes,
        class_names=[class_cfg.id_to_class[i] for i in range(class_cfg.num_classes)],
        include_background=False,
    )

    print("\nTraining configuration:")
    print(f"  Classes: {class_cfg.num_classes}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Epochs: {config.training.num_epochs}")
    print(f"  Checkpoints: {config.paths.unified_checkpoints_dir}")

    print("\n" + "=" * 60)
    print("Starting unified training...")
    print("=" * 60)

    # Resume if a checkpoint exists (pass --fresh to ignore it). load_checkpoint
    # restores model/optimizer/scheduler and the best-Dice watermark, so
    # best_model.pth is only overwritten by genuinely better epochs.
    remaining = config.training.num_epochs
    ckpt_path = config.paths.unified_checkpoints_dir / "best_model.pth"
    if ckpt_path.exists() and "--fresh" not in sys.argv:
        trainer.load_checkpoint(ckpt_path)
        remaining = max(1, config.training.num_epochs - trainer.current_epoch)
        print(f"Resuming: {remaining} epochs remaining")

    best_metrics = trainer.train(num_epochs=remaining)

    print("\n" + "=" * 60)
    print("Unified training complete!")
    print("=" * 60)
    if best_metrics:
        print(f"\nBest validation metrics:")
        print(f"  Mean Dice: {best_metrics['mean_dice']:.4f}")
        print(f"  Mean IoU:  {best_metrics['mean_iou']:.4f}")
        for k in sorted(best_metrics):
            if k.startswith("dice_"):
                print(f"  {k}: {best_metrics[k]:.4f}")
    else:
        print("\nNo epoch in this run beat the resumed checkpoint; "
              "best_model.pth is unchanged.")


if __name__ == "__main__":
    main()
