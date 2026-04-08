"""
Script aims to:
1) Print 3D CNN model architecture + parameter counts
2) Plot word dataset statistics from word_statistics.json
3) Visualize a sample .npy video (8 evenly spaced frames)
"""

from __future__ import annotations
from training.train_word_recognition_3d_cnn import ThreeDimensionalCNN
import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))


def count_model_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters in a model."""
    total_parameter_count = 0
    trainable_parameter_count = 0
    for parameter in model.parameters():
        parameter_count = parameter.numel()
        total_parameter_count += parameter_count
        if parameter.requires_grad:
            trainable_parameter_count += parameter_count
    return total_parameter_count, trainable_parameter_count


def visualize_model_architecture(
    input_channels: int,
    number_of_classes: int,
    frames: int,
    height: int,
    width: int,
) -> None:
    """Print model structure, parameter counts, and output shape."""
    print("=" * 80)
    print("3D CNN model architecture")
    print("=" * 80)
    model = ThreeDimensionalCNN(
        input_channels=input_channels,
        number_of_classes=number_of_classes)
    print("\nModel:")
    print(model)
    print("\n" + "-" * 80)
    print("LAYER-WISE PARAMETER COUNT")
    print("-" * 80)
    total_parameter_count = 0
    trainable_parameter_count = 0
    for name, parameter in model.named_parameters():
        parameter_count = parameter.numel()
        total_parameter_count += parameter_count
        if parameter.requires_grad:
            trainable_parameter_count += parameter_count
        print(f"{name:55s} {parameter_count:>15,} params")
    print("-" * 80)
    non_trainable_parameter_count = (
        total_parameter_count - trainable_parameter_count)
    print(f"{'TOTAL PARAMETERS':55s} {total_parameter_count:>15,}")
    print(f"{'TRAINABLE PARAMETERS':55s} {trainable_parameter_count:>15,}")
    print(
        f"{'NON-TRAINABLE PARAMETERS':55s} "
        f"{non_trainable_parameter_count:>15,}")
    print("-" * 80)

    estimated_model_size_megabytes = (
        total_parameter_count * 4
    ) / (1024 ** 2)
    print(
        "\nEstimated model size (float32): "
        f"{estimated_model_size_megabytes:.2f} MB")
    print("\n" + "-" * 80)
    print("FORWARD PASS SHAPES")
    print("-" * 80)

    dummy_input_tensor = torch.randn(
        1,
        input_channels,
        frames,
        height,
        width)
    print(f"Input shape: {list(dummy_input_tensor.shape)}")
    with torch.no_grad():
        output_logits = model(dummy_input_tensor)
    print(f"Output shape: {list(output_logits.shape)}")
    print(f"Output represents logits for {output_logits.shape[1]} classes")


def load_word_statistics(stats_file_path: Path) -> Dict[str, Any]:
    """Load dataset statistics JSON."""
    with stats_file_path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def visualize_data_statistics(
    stats_file_path: Path,
    save_plot_path: Path,
) -> None:
    """Print and plot dataset statistics"""
    print("=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)
    if not stats_file_path.exists():
        print(f"Error: Statistics file not found: {stats_file_path}")
        print("Run preprocessing to generate word_statistics.json first.")
        return
    stats_data = load_word_statistics(stats_file_path)
    total_words = int(stats_data.get("total_words", 0))
    total_samples = int(stats_data.get("total_samples", 0))
    word_statistics: Dict[str, Dict[str, Any]] = stats_data.get(
        "word_statistics", {})

    if not word_statistics:
        print("Error: word_statistics is empty in stats JSON.")
        return
    average_samples_per_word = (
        total_samples / total_words if total_words else 0.0)
    print(f"\nTotal unique words: {total_words}")
    print(f"Total samples: {total_samples}")
    print(f"Average samples per word: {average_samples_per_word:.1f}")
    words_sorted_by_count = sorted(
        word_statistics.keys(),
        key=lambda word_label: word_statistics[word_label].get("count", 0),
        reverse=True)

    def print_word_table(title: str, word_list: List[str]) -> None:
        print("\n" + "-" * 80)
        print(title)
        print("-" * 80)
        print(
            f"{'Word':<15} {'Count':>10} "
            f"{'Avg Duration':>15} {'Avg Frames':>12}")
        print("-" * 80)
        for word_label in word_list:
            entry = word_statistics[word_label]
            count = int(entry.get("count", 0))
            avg_duration = float(entry.get("avg_duration", 0.0))
            avg_frames = float(entry.get("avg_frames", 0.0))
            print(
                f"{word_label:<15} {count:>10} "
                f"{avg_duration:>14.3f}s {avg_frames:>12.1f}")

    print_word_table(
        "TOP 10 WORDS BY SAMPLE COUNT",
        words_sorted_by_count[:10])
    print_word_table(
        "BOTTOM 10 WORDS BY SAMPLE COUNT",
        words_sorted_by_count[-10:])
    counts = [
        int(word_statistics[word_label].get("count", 0))
        for word_label in words_sorted_by_count]
    avg_durations = [
        float(word_statistics[word_label].get("avg_duration", 0.0))
        for word_label in words_sorted_by_count]
    avg_frames = [
        float(word_statistics[word_label].get("avg_frames", 0.0))
        for word_label in words_sorted_by_count]
    plt.figure(figsize=(14, 10))
    plt.subplot(2, 2, 1)
    plt.bar(
        range(len(words_sorted_by_count)),
        counts,
        edgecolor="black",
        linewidth=0.5)
    plt.title("Sample Count Distribution Across Words")
    plt.xlabel("Words (sorted by count)")
    plt.ylabel("Number of Samples")
    plt.grid(True, alpha=0.3)
    plt.subplot(2, 2, 2)
    plt.hist(avg_durations, bins=30, edgecolor="black", linewidth=1)
    plt.title("Distribution of Average Word Durations")
    plt.xlabel("Average Duration (seconds)")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.subplot(2, 2, 3)
    plt.hist(avg_frames, bins=30, edgecolor="black", linewidth=1)
    plt.title("Distribution of Average Frame Counts")
    plt.xlabel("Average Number of Frames")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    top_words = words_sorted_by_count[:15]
    top_counts = [
        int(word_statistics[word_label].get("count", 0))
        for word_label in top_words]
    y_positions = np.arange(len(top_words))
    plt.subplot(2, 2, 4)
    plt.barh(
        y_positions,
        top_counts,
        edgecolor="black",
        linewidth=1)
    plt.yticks(y_positions, top_words)
    plt.gca().invert_yaxis()
    plt.title("Top 15 Words by Sample Count")
    plt.xlabel("Sample Count")
    plt.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    save_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_plot_path, dpi=150, bbox_inches="tight")
    print(f"\nStatistics plot saved to: {save_plot_path}")
    plt.show()


def pick_word_folder(
    data_root: Path,
    requested_word: str | None,
) -> Path | None:
    """Pick requested wordfolder or choose one at random"""
    if requested_word:
        candidate = data_root / requested_word
        return candidate if candidate.exists() else None
    word_folders = [path for path in data_root.iterdir() if path.is_dir()]
    if not word_folders:
        return None
    return random.choice(word_folders)


def visualize_sample_video(
    data_root: Path,
    requested_word: str | None,
    save_plot_path: Path,
    assume_bgr_frames: bool,
) -> None:
    """Visualize 8 evenly spaced frames from a sample clip."""
    print("=" * 80)
    print("SAMPLE VIDEO VISUALIZATION")
    print("=" * 80)
    word_folder_path = pick_word_folder(data_root, requested_word)
    if word_folder_path is None:
        print(f"Error: No word folder found under: {data_root}")
        return
    word_label = word_folder_path.name
    print(f"\nSelected word: '{word_label}'")
    numpy_files = list(word_folder_path.glob("*.npy"))
    if not numpy_files:
        print(f"Error: No .npy files found for word '{word_label}'")
        return

    video_file_path = random.choice(numpy_files)
    metadata_file_path = video_file_path.with_suffix(".json")
    print(f"\nLoading sample: {video_file_path.name}")
    video_frames = np.load(video_file_path)
    num_frames = int(video_frames.shape[0])
    print(f"Video shape: {video_frames.shape}")

    if metadata_file_path.exists():
        with metadata_file_path.open("r", encoding="utf-8") as file_handle:
            metadata = json.load(file_handle)
        print("\nMetadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")

    number_of_display_frames = min(8, num_frames)
    frame_indices = np.linspace(
        0,
        num_frames - 1,
        number_of_display_frames,
        dtype=int)

    figure, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes_flat = axes.flatten()
    for plot_index, frame_index in enumerate(frame_indices):
        frame = video_frames[frame_index]
        if assume_bgr_frames and frame.ndim == 3 and frame.shape[-1] == 3:
            frame = frame[:, :, ::-1]
        axes_flat[plot_index].imshow(frame)
        axes_flat[plot_index].set_title(
            f"Frame {frame_index + 1}/{num_frames}",
            fontsize=12)
        axes_flat[plot_index].axis("off")

    figure.suptitle(
        f"Sample Video Frames - Word: '{word_label}'\n"
        f"{video_file_path.name}",
        fontsize=16,
        fontweight="bold")
    plt.tight_layout()
    save_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_plot_path, dpi=150, bbox_inches="tight")
    print(f"\nSample visualization saved to: {save_plot_path}")
    plt.show()


def build_argument_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser"""
    parser = argparse.ArgumentParser(
        description="Model and Data Visualization")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["architecture", "data_stats", "sample_video"],
        default="architecture",
        help="Visualization mode")
    parser.add_argument(
        "--word",
        type=str,
        default=None,
        help=(
            "Word to visualize (sample_video mode only). "
            "If omitted, chooses random."),)
    parser.add_argument(
        "--assume-bgr",
        action="store_true",
        help=(
            "If frames look color-swapped, enable this to treat them "
            "as BGR and convert to RGB for plotting"),)
    parser.add_argument("--input-channels", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=52)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    return parser


def main() -> None:
    """Run selected visualization mode"""
    parser = build_argument_parser()
    args = parser.parse_args()
    words_data_root = PROJECT_ROOT / "data" / "processed" / "words_by_label"
    stats_file_path = words_data_root / "word_statistics.json"

    if args.mode == "architecture":
        visualize_model_architecture(
            input_channels=args.input_channels,
            number_of_classes=args.num_classes,
            frames=args.frames,
            height=args.height,
            width=args.width)
        return

    if args.mode == "data_stats":
        save_plot_path = PROJECT_ROOT / "visuals" / "data_statistics.png"
        visualize_data_statistics(
            stats_file_path=stats_file_path,
            save_plot_path=save_plot_path)
        return

    if args.mode == "sample_video":
        save_plot_path = (
            PROJECT_ROOT
            / "visuals"
            / f"sample_video_{args.word or 'random'}.png")
        visualize_sample_video(
            data_root=words_data_root,
            requested_word=args.word,
            save_plot_path=save_plot_path,
            assume_bgr_frames=bool(args.assume_bgr))


if __name__ == "__main__":
    main()
