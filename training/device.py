"""Shared device resolution for training and inference."""

import torch


def resolve_device() -> torch.device:
    """Return CUDA if available, otherwise CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        return device
    print("Using CPU")
    return torch.device("cpu")
