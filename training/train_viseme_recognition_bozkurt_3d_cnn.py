"""Viseme 3D CNN training (Bozkurt). Config only; see shared modules"""

import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from training.model import ThreeDimensionalCNN
from training.dataset import VisemeDataset
from training.device import resolve_device
from training.train_utils import (
    build_sampler, train_one_epoch, eval_one_epoch, save_checkpoint)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = (
    PROJECT_ROOT / "data" / "processed" / "visemes_bozkurt_mfa_balanced_npy")
CHECKPOINT_DIR = PROJECT_ROOT / "training" / "checkpoints_bozkurt_viseme"
TENSORBOARD_DIR = PROJECT_ROOT / "training" / "runs_bozkurt_viseme"
NUM_FRAMES = 8
HEIGHT = 64
WIDTH = 64
BATCH_SIZE = 32
EPOCHS = 60
LR = 0.0005
WEIGHT_DECAY = 1e-5
PATIENCE = 15
SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


def main() -> None:
    device = resolve_device()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    visemes = sorted(f.name for f in DATA_ROOT.iterdir() if f.is_dir())
    viseme_to_index = {v: i for i, v in enumerate(visemes)}
    index_to_viseme = {i: v for v, i in viseme_to_index.items()}
    num_classes = len(visemes)
    print(f"Classes: {num_classes} visemes: {visemes}")
    # dataset split
    full = VisemeDataset(
        data_root=DATA_ROOT,
        viseme_to_index=viseme_to_index,
        num_frames=NUM_FRAMES, height=HEIGHT, width=WIDTH)
    n = len(full)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    indices = list(range(n))
    random.shuffle(indices)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    train_set = Subset(full, train_idx)
    val_set = Subset(full, val_idx)
    n_test = n - n_train - n_val
    print(f"Split  train={len(train_set)}  val={len(val_set)}  test={n_test}")
    train_labels = [full.samples[i]["label_index"] for i in train_idx]
    sampler = build_sampler(train_labels, num_classes)
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    # model setup
    model = ThreeDimensionalCNN(
        num_classes=num_classes, input_channels=1).to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)
    writer = SummaryWriter(TENSORBOARD_DIR)
    best_val_acc = 0.0
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimiser, device)
        val_loss, val_acc = eval_one_epoch(
            model, val_loader, criterion, device)
        scheduler.step()
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Acc/val", val_acc, epoch)
        print(
            f"Epoch {epoch:3d}/{EPOCHS}  train={train_acc:.2f}%"
            f"  val={val_acc:.2f}%  loss={train_loss:.4f}/{val_loss:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            save_checkpoint(
                CHECKPOINT_DIR / "bozkurt_viseme_best_model.pth",
                model, optimiser, epoch, val_acc,
                label_map=viseme_to_index,
                extra={"index_to_viseme": index_to_viseme})
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    writer.close()
    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
