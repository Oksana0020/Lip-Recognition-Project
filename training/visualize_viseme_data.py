"""
Visualize viseme dataset statistics and sample frames.
"""

from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent


def scan_viseme_samples(data_root: Path) -> Dict[str, List[Path]]:
    viseme_samples: Dict[str, List[Path]] = {}
    for viseme_dir in sorted(data_root.iterdir()):
        if not viseme_dir.is_dir():
            continue
        npy_files = sorted(viseme_dir.glob("*.npy"))
        if npy_files:
            viseme_samples[viseme_dir.name] = npy_files
    return viseme_samples


def plot_viseme_stats(data_root: Path, save_path: Path) -> None:
    viseme_samples = scan_viseme_samples(data_root)
    if not viseme_samples:
        print(f"No viseme samples found in: {data_root}")
        return
    visemes = list(viseme_samples.keys())
    counts = [len(viseme_samples[v]) for v in visemes]
    plt.figure(figsize=(10, 4))
    bars = plt.bar(visemes, counts, edgecolor="black", linewidth=0.5)
    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            str(count),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.title("Viseme Sample Counts")
    plt.xlabel("Viseme class")
    plt.ylabel("Number of samples")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved viseme stats plot: {save_path}")


def load_npy_video(npy_path: Path) -> np.ndarray:
    return np.load(npy_path)


def sample_frames(frames: np.ndarray, count: int = 8) -> np.ndarray:
    if frames.ndim == 3:
        frames = np.expand_dims(frames, axis=-1)
    total = frames.shape[0]
    indices = np.linspace(0, total - 1, count, dtype=int)
    return frames[indices]


def plot_viseme_sample(
    data_root: Path,
    viseme: str | None,
    save_path: Path,
) -> None:
    viseme_samples = scan_viseme_samples(data_root)
    if not viseme_samples:
        print(f"No viseme samples found in: {data_root}")
        return
    if viseme is None:
        viseme = random.choice(list(viseme_samples.keys()))=
    samples = viseme_samples.get(viseme)
    if not samples:
        print(f"No samples found for viseme: {viseme}")
        return
    npy_path = random.choice(samples)
    frames = load_npy_video(npy_path)
    grid = sample_frames(frames, count=8)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    for ax, frame in zip(axes, grid):
        if frame.shape[-1] == 1:
            ax.imshow(frame.squeeze(), cmap="gray")
        else:
            # OpenCV stores frames as BGR, convert to RGB for correct colours
            ax.imshow(frame[..., ::-1])
        ax.axis("off")
    fig.suptitle(f"Viseme {viseme} sample: {npy_path.name}")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved viseme sample: {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize viseme dataset")
    parser.add_argument(
        "--mode",
        choices=["stats", "sample"],
        required=True,
        help="stats: count plot, sample: frame grid",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "visemes_bozkurt_mfa",
        help="Path to viseme dataset root",
    )
    parser.add_argument(
        "--viseme",
        type=str,
        default=None,
        help="Viseme class for sample mode",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "visuals" / "viseme",
        help="Output folder for plots",
    )
    args = parser.parse_args()
    if args.mode == "stats":
        plot_viseme_stats(args.data_root, args.out / "viseme_counts.png")
    else:
        plot_viseme_sample(
            args.data_root,
            args.viseme,
            args.out / f"viseme_sample_{args.viseme or 'random'}.png",
        )


if __name__ == "__main__":
    main()
