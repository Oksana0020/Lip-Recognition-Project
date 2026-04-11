"""
Script to compute per-class accuracy and save a bar chart
"""

from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_viseme_model_and_dataset(
    checkpoint_path: Path,
    data_root: Path,
) -> Tuple[torch.nn.Module, List[str], DataLoader]:
    """Load viseme model, class labels and evaluation dataloader"""
    from training.train_viseme_recognition_bozkurt_3d_cnn import (
        BozkurtVisemeLipReadingDataset,
        BozkurtVisemeTrainingConfig,
        ThreeDimensionalCNN
    )
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
    viseme_class_labels = checkpoint_data["viseme_classes"]
    training_config = BozkurtVisemeTrainingConfig()
    resolved_data_root = str(data_root)
    first_class_dir = Path(resolved_data_root) / viseme_class_labels[0]
    if first_class_dir.exists():
        npy_count = sum(
            1 for f in first_class_dir.iterdir() if f.suffix == ".npy"
        )
        if npy_count == 0:
            resolved_data_root = training_config.data_root_directory
            print(
                f"Note: supplied data-root has no .npy files; "
                f"using training config path: {resolved_data_root}"
            )
    dataset = BozkurtVisemeLipReadingDataset(
        data_root_directory=resolved_data_root,
        viseme_class_labels=viseme_class_labels,
        sequence_length_frames=training_config.sequence_length_frames,
        target_height_pixels=training_config.video_height_pixels,
        target_width_pixels=training_config.video_width_pixels,
        data_split="all",
    )
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )
    model = ThreeDimensionalCNN(
        number_of_classes=len(viseme_class_labels),
        input_channels=1,
    )
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.eval()
    return model, viseme_class_labels, dataloader


def load_word_model_and_dataset(
    checkpoint_path: Path,
    data_root: Path,
) -> Tuple[torch.nn.Module, List[str], DataLoader]:
    """Load word model, class labels and evaluation dataloader"""
    from training.train_word_recognition_3d_cnn import (
        ThreeDimensionalCNN,
        TrainingConfig,
        WordLipReadingDataset,
        load_word_dataset,
    )
    training_config = TrainingConfig()
    sample_list, word_to_index, index_to_word = load_word_dataset(data_root)
    dataset = WordLipReadingDataset(
        samples=sample_list,
        word_to_index_mapping=word_to_index,
        target_frame_count=training_config.target_frame_count,
        target_frame_height=training_config.target_frame_height,
        target_frame_width=training_config.target_frame_width,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
    model = ThreeDimensionalCNN(
        input_channels=1,
        number_of_classes=len(word_to_index),
    )
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.eval()
    class_labels = [
        index_to_word[index]
        for index in range(len(index_to_word))]
    return model, class_labels, dataloader


def compute_per_class_accuracy(
    model: torch.nn.Module,
    dataloader: DataLoader,
    class_labels: List[str],
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """Compute total, correct and accuracy for each class."""
    total_by_class = {label: 0 for label in class_labels}
    correct_by_class = {label: 0 for label in class_labels}
    model.to(device)
    with torch.no_grad():
        for input_batch, target_batch in dataloader:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            logits = model(input_batch)
            predicted_indices = torch.argmax(logits, dim=1)
            predicted_list = predicted_indices.cpu().numpy().tolist()
            target_list = target_batch.cpu().numpy().tolist()
            for predicted_index, target_index in zip(
                predicted_list,
                target_list,
            ):
                class_label = class_labels[int(target_index)]
                total_by_class[class_label] += 1
                if int(predicted_index) == int(target_index):
                    correct_by_class[class_label] += 1
    results: Dict[str, Dict[str, float]] = {}
    for class_label in class_labels:
        total_count = total_by_class[class_label]
        correct_count = correct_by_class[class_label]
        accuracy_percent = (
            (correct_count / total_count) * 100.0 if total_count else 0.0
        )
        results[class_label] = {
            "total": total_count,
            "correct": correct_count,
            "accuracy": accuracy_percent,
        }
    return results


def save_results_plot(
    results: Dict[str, Dict[str, float]],
    output_directory: Path,
    top_n: int = 20,
    task_label: str = "",
) -> None:
    """Save per-class accuracy bar chart"""
    sorted_results = sorted(
        results.items(),
        key=lambda item: item[1]["total"],
        reverse=True,
    )
    if top_n:
        sorted_results = sorted_results[:top_n]
    class_labels = [label for label, _ in sorted_results]
    accuracy_values = [stats["accuracy"] for _, stats in sorted_results]
    plt.figure(figsize=(12, 5))
    bars = plt.bar(
        class_labels,
        accuracy_values,
        edgecolor="black",
        linewidth=0.5,
    )
    for bar, accuracy in zip(bars, accuracy_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{accuracy:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Accuracy (%)")
    prefix = f"{task_label} — " if task_label else ""
    plt.title(
        f"{prefix}Per-Class Accuracy (Top {len(class_labels)} by support)"
    )
    plt.ylim(0, 110)
    plt.tight_layout()
    output_directory.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_directory / "per_class_accuracy.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def save_results_csv(
    results: Dict[str, Dict[str, float]],
    output_directory: Path,
) -> None:
    """Save per-class accuracy statistics to CSV"""
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / "per_class_accuracy.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["class", "total", "correct", "accuracy_percent"])
        for class_label, stats in results.items():
            writer.writerow(
                [
                    class_label,
                    stats["total"],
                    stats["correct"],
                    stats["accuracy"],
                ]
            )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Compute per-class accuracy.",
    )
    parser.add_argument(
        "--task",
        choices=["viseme", "word"],
        required=True,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    """Run per-class accuracy evaluation."""
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.task == "viseme":
        model, class_labels, dataloader = load_viseme_model_and_dataset(
            args.checkpoint,
            args.data_root,
        )
    else:
        model, class_labels, dataloader = load_word_model_and_dataset(
            args.checkpoint,
            args.data_root,
        )
    results = compute_per_class_accuracy(
        model=model,
        dataloader=dataloader,
        class_labels=class_labels,
        device=device,
    )
    task_label = "Viseme Level" if args.task == "viseme" else "Word Level"
    save_results_csv(results, args.outdir)
    save_results_plot(
        results, args.outdir, top_n=args.top_n, task_label=task_label
    )
    print(f"Saved per-class accuracy to: {args.outdir}")


if __name__ == "__main__":
    main()
