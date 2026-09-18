"""Device utilities for MPS/CUDA/CPU support."""
import torch
from typing import Dict, Any


def get_device() -> torch.device:
    """Get the best available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def get_dataloader_kwargs(device: torch.device) -> Dict[str, Any]:
    """Get DataLoader kwargs optimized for the device."""
    if device.type == "mps":
        # MPS requires special handling
        return {
            "num_workers": 0,
            "pin_memory": False,
        }
    elif device.type == "cuda":
        return {
            "num_workers": 4,
            "pin_memory": True,
        }
    else:
        return {
            "num_workers": 0,
            "pin_memory": False,
        }


def to_device(data: Any, device: torch.device) -> Any:
    """Move data to device, handling nested structures."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(v, device) for v in data)
    return data
