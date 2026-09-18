"""Segmentation model definitions using MONAI."""
import torch
import torch.nn as nn
from monai.networks.nets import UNet, FlexibleUNet
from typing import Tuple, Optional, List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import ModelConfig


def create_unet(config: Optional[ModelConfig] = None) -> UNet:
    """Create a MONAI UNet model for segmentation.

    Args:
        config: Model configuration

    Returns:
        MONAI UNet model
    """
    cfg = config or ModelConfig()

    model = UNet(
        spatial_dims=cfg.spatial_dims,
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        channels=cfg.channels,
        strides=cfg.strides,
        num_res_units=cfg.num_res_units,
        dropout=cfg.dropout,
    )

    return model


def create_flexible_unet(
    out_channels: int = 7,
    in_channels: int = 3,
    backbone: str = "efficientnet-b2",
    dropout: float = 0.2,
    pretrained_weights_path: Optional[Path] = None,
) -> FlexibleUNet:
    """Create a FlexibleUNet matching the MONAI endoscopic_tool_segmentation bundle.

    When pretrained_weights_path is provided, loads encoder + decoder weights from
    that checkpoint and leaves the segmentation head freshly initialized (the
    bundle's head outputs 2 channels for binary segmentation; ours outputs N).
    """
    # MONAI's FlexibleUNet accepts pre_conv=None at runtime (the bundle config sets
    # it to null) but its type hint requires str. Build via cast to satisfy ty.
    from typing import cast, Any
    model = cast(Any, FlexibleUNet)(
        in_channels=in_channels,
        out_channels=out_channels,
        backbone=backbone,
        spatial_dims=2,
        dropout=dropout,
        pretrained=False,
        is_pad=False,
        pre_conv=None,
    )

    if pretrained_weights_path is not None:
        sd = torch.load(str(pretrained_weights_path), map_location="cpu", weights_only=False)
        sd = {k: v for k, v in sd.items() if not k.startswith("segmentation_head")}
        result = model.load_state_dict(sd, strict=False)
        unexpected = [k for k in result.unexpected_keys if not k.startswith("segmentation_head")]
        if unexpected:
            raise RuntimeError(f"Unexpected keys in pretrained weights: {unexpected[:5]}")
        head_keys = [k for k in result.missing_keys if not k.startswith("segmentation_head")]
        if head_keys:
            raise RuntimeError(f"Unexpected missing keys (only head should be missing): {head_keys[:5]}")

    return model


def get_param_groups(model: FlexibleUNet, head_lr: float, backbone_lr: float) -> List[dict]:
    """Split FlexibleUNet params into pretrained backbone (encoder+decoder) and fresh head.

    Returns the param-group list for an optimizer with differential learning rates.
    """
    head_params, backbone_params = [], []
    for name, p in model.named_parameters():
        if name.startswith("segmentation_head"):
            head_params.append(p)
        else:
            backbone_params.append(p)
    return [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": head_lr},
    ]


class SegmentationModel(nn.Module):
    """Wrapper for segmentation model with utility methods."""

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()
        self.model = create_unet(self.config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Logits tensor of shape (B, num_classes, H, W)
        """
        return self.model(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted class labels.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Predicted labels of shape (B, H, W)
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Probabilities of shape (B, num_classes, H, W)
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)

    def get_num_parameters(self) -> int:
        """Get total number of model parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_model(checkpoint_path: Path, device: torch.device) -> SegmentationModel:
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model to

    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load config if saved
    config = checkpoint.get('config', ModelConfig())
    model = SegmentationModel(config)

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model
