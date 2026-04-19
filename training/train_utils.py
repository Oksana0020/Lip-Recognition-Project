"""Shared training utilities: train/eval loop, sampler, checkpoint IO."""

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm


def build_sampler(
    label_indices: List[int], num_classes: int
) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler for balanced class sampling"""
    counts = np.zeros(num_classes, dtype=np.float64)
    for idx in label_indices:
        counts[idx] += 1.0
    weights_per_class = 1.0 / np.maximum(counts, 1.0)
    sample_weights = torch.tensor(
        [weights_per_class[i] for i in label_indices], dtype=torch.float)
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(label_indices),
        replacement=True)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimiser: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, accuracy %)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for frames, labels in tqdm(loader, desc="Train", leave=False):
        frames, labels = frames.to(device), labels.to(device)
        optimiser.zero_grad()
        logits = model(frames)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        bs = frames.size(0)
        total_loss += loss.item() * bs
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += bs
    return total_loss / max(total, 1), 100.0 * correct / max(total, 1)


def eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Run one evaluation epoch. Returns (avg_loss, accuracy %)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for frames, labels in tqdm(loader, desc="Eval", leave=False):
            frames, labels = frames.to(device), labels.to(device)
            logits = model(frames)
            loss = criterion(logits, labels)
            bs = frames.size(0)
            total_loss += loss.item() * bs
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += bs
    return total_loss / max(total, 1), 100.0 * correct / max(total, 1)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimiser: optim.Optimizer,
    epoch: int,
    val_accuracy: float,
    label_map: Dict,
    extra: Dict = None
) -> None:
    """Save model checkpoint with metadata"""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimiser_state_dict": optimiser.state_dict(),
            "val_accuracy": val_accuracy,
            "label_map": label_map,
            **(extra or {})
        },
        path)
    print(f"  Checkpoint saved: {path}")


_OLD_KEY_MAP = {
    "conv_block_1": "block1",
    "conv_block_2": "block2",
    "conv_block_3": "block3",
    "conv_block_4": "block4",
    "fully_connected_classifier": "classifier",
}

_OLD_CLASSIFIER_IDX = {"classifier.0.": "classifier.1.",
                       "classifier.3.": "classifier.4."}


def _remap_state_dict(state_dict: Dict) -> Dict:
    """Rename legacy layer keys to current names for backward compat."""
    remapped = {}
    for k, v in state_dict.items():
        for old, new in _OLD_KEY_MAP.items():
            k = k.replace(old, new, 1)
        for old, new in _OLD_CLASSIFIER_IDX.items():
            if k.startswith(old):
                k = new + k[len(old):]
                break
        remapped[k] = v
    return remapped


def load_checkpoint(
    path: Path, model: nn.Module, device: torch.device
) -> Dict:
    """Load checkpoint into model and return the full dict"""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state_dict = _remap_state_dict(ckpt["model_state_dict"])
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return ckpt
