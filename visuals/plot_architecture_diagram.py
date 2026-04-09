"""
Plot a visual architecture diagram of the 3D CNN model.
Input -> Conv Block 1-4 -> AdaptiveAvgPool -> FC -> Output
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "visuals"
COLORS = {
    "input": "#4A90D9",
    "conv": "#E8A838",
    "pool": "#5BAD6F",
    "fc": "#C0616B",
    "output": "#8E6DBE",
    "arrow": "#555555",
    "bg": "#FAFAFA"
}


def build_layers(num_classes: int, task_name: str) -> list[dict]:
    return [
        {"label": "Input\n8×1×64×64",
         "sub": "(frames×ch×H×W)",
         "color": COLORS["input"]},

        {"label": "Conv3D Block 1",
         "sub": "1→32\nBN·ReLU\nPool(1,2,2)",
         "color": COLORS["conv"]},

        {"label": "Conv3D Block 2",
         "sub": "32→64\nBN·ReLU\nPool(2,2,2)",
         "color": COLORS["conv"]},

        {"label": "Conv3D Block 3",
         "sub": "64→128\nBN·ReLU\nPool(2,2,2)",
         "color": COLORS["conv"]},

        {"label": "Conv3D Block 4",
         "sub": "128→256\nBN·ReLU\nPool(2,2,2)",
         "color": COLORS["conv"]},

        {"label": "AdaptiveAvgPool3D",
         "sub": "1×4×4 → 4096",
         "color": COLORS["pool"]},

        {"label": "Fully Connected",
         "sub": "4096→512\nReLU·Dropout",
         "color": COLORS["fc"]},

        {"label": f"Output\n{num_classes}",
         "sub": task_name,
         "color": COLORS["output"]}]


def draw_pipeline(ax: plt.Axes, layers: list[dict], title: str) -> None:
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    n = len(layers)
    box_h = 0.085
    gap = (1 - n * box_h) / (n + 1)
    box_w = 0.72
    x = (1 - box_w) / 2
    ax.text(
        0.5, 0.98, title,
        ha="center", va="top",
        fontsize=12, weight="bold")
    centers = []
    for i, layer in enumerate(layers):
        y = 1 - (i + 1) * (box_h + gap)
        cy = y + box_h / 2
        centers.append(cy)
        box = FancyBboxPatch(
            (x, y), box_w, box_h,
            boxstyle="round,pad=0.01",
            facecolor=layer["color"],
            edgecolor="white",
            lw=1.5)
        ax.add_patch(box)
        ax.text(
            0.5, cy + 0.01, layer["label"],
            ha="center", va="center",
            fontsize=8, weight="bold", color="white")
        if layer.get("sub"):
            ax.text(
                0.5, cy - 0.02, layer["sub"],
                ha="center", va="center",
                fontsize=6, color="white")
    for i in range(len(centers) - 1):
        ax.annotate(
            "",
            xy=(0.5, centers[i + 1] + box_h / 2),
            xytext=(0.5, centers[i] - box_h / 2),
            arrowprops=dict(
                arrowstyle="-|>",
                color=COLORS["arrow"],
                lw=1.3))


def add_legend(fig: plt.Figure) -> None:
    items = [
        mpatches.Patch(color=COLORS["input"], label="Input"),
        mpatches.Patch(color=COLORS["conv"], label="Conv"),
        mpatches.Patch(color=COLORS["pool"], label="Pooling"),
        mpatches.Patch(color=COLORS["fc"], label="FC"),
        mpatches.Patch(color=COLORS["output"], label="Output")]
    fig.legend(
        handles=items,
        loc="lower center",
        ncol=5,
        fontsize=8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=["word", "viseme", "both"],
        default="both")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.task == "both":
        fig, axes = plt.subplots(1, 2, figsize=(10, 10))
        draw_pipeline(
            axes[0],
            build_layers(51, "word"),
            "Word Model")
        draw_pipeline(
            axes[1],
            build_layers(13, "viseme"),
            "Viseme Model")
        out = args.outdir / "architecture.png"
    elif args.task == "word":
        fig, ax = plt.subplots(figsize=(5, 10))
        draw_pipeline(
            ax,
            build_layers(51, "word"),
            "Word Model")
        out = args.outdir / "architecture_word.png"
    else:
        fig, ax = plt.subplots(figsize=(5, 10))
        draw_pipeline(
            ax,
            build_layers(13, "viseme"),
            "Viseme Model")
        out = args.outdir / "architecture_viseme.png"
    add_legend(fig)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
