"""
Evaluate a trained 3D CNN checkpoint on a dataset split and export metrics
"""

from __future__ import annotations
import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from sklearn.metrics import classification_report, confusion_matrix
except Exception as exc:
    raise RuntimeError(
        "scikit-learn is required for evaluation"
    ) from exc


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))


@dataclass
class EvaluationOutputPaths:
    metrics_json_path: Path
    classification_report_path: Path
    confusion_matrix_path: Path


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for model evaluation"""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained 3D CNN model"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--task",
        type=str,
        choices=["word", "viseme"],
        required=True,
    )
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    """Return torch device requested by the user"""
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_name)


def load_word_model(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[torch.nn.Module, List[str]]:
    """Load a trained word-level model and ordered class labels."""
    from training.train_word_recognition_3d_cnn import (
        ThreeDimensionalCNN)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    word_to_index = checkpoint["word_to_index"]
    raw_index_to_word = checkpoint["index_to_word"]
    index_to_word = {
        int(index): word
        for index, word in raw_index_to_word.items()
    }
    class_count = len(word_to_index)
    model = ThreeDimensionalCNN(
        input_channels=1,
        number_of_classes=class_count,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    class_labels = [
        index_to_word[index]
        for index in range(len(index_to_word))
    ]
    return model, class_labels


def load_viseme_model(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[torch.nn.Module, List[str]]:
    """Load a trained viseme-level model and class labels"""
    from training.train_viseme_recognition_bozkurt_3d_cnn import (
        ThreeDimensionalCNN,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_labels = checkpoint["viseme_classes"]
    model = ThreeDimensionalCNN(
        number_of_classes=len(class_labels),
        input_channels=1,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_labels


class NpyClassificationDataset(Dataset):
    """Dataset that loads .npy clips from class subfolders"""
    def __init__(
        self,
        data_root: Path,
        class_labels: List[str],
        sequence_length: int,
        frame_height: int,
        frame_width: int,
        channels: int,
        use_time_first_layout: bool = False,
    ) -> None:
        self.data_root = data_root
        self.class_labels = class_labels
        self.sequence_length = sequence_length
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.channels = channels
        self.use_time_first_layout = use_time_first_layout
        self.label_to_index = {
            label: index for index, label in enumerate(class_labels)}
        self.samples = self._collect_samples()

    def _collect_samples(self) -> List[Tuple[Path, int]]:
        """Collect all .npy samples and their numeric labels"""
        collected_samples: List[Tuple[Path, int]] = []
        for class_label in self.class_labels:
            class_directory = self.data_root / class_label
            if not class_directory.exists():
                continue
            for file_path in sorted(class_directory.glob("*.npy")):
                label_index = self.label_to_index[class_label]
                collected_samples.append((file_path, label_index))
        return collected_samples

    def __len__(self) -> int:
        return len(self.samples)

    def _resize_sequence_length(self, frames: np.ndarray) -> np.ndarray:
        """Trim frames to the required sequence length."""
        current_frame_count = frames.shape[0]
        if current_frame_count >= self.sequence_length:
            sampled_indices = np.linspace(
                0,
                current_frame_count - 1,
                self.sequence_length,
                dtype=int,
            )
            return frames[sampled_indices]
        repeated_indices = [
            index % current_frame_count
            for index in range(self.sequence_length)
        ]
        return frames[repeated_indices]

    def _resize_frames(self, frames: np.ndarray) -> np.ndarray:
        """Resize every frame to the target spatial resolution"""
        import cv2
        resized_frames = []
        for frame in frames:
            resized_frame = cv2.resize(
                frame,
                (self.frame_width, self.frame_height),
            )
            resized_frames.append(resized_frame)
        return np.stack(resized_frames, axis=0)

    def _match_channel_count(self, frames: np.ndarray) -> np.ndarray:
        """Convert frames to the expected number of channels"""
        if frames.ndim == 3:
            frames = np.expand_dims(frames, axis=-1)

        if self.channels == 1 and frames.shape[-1] == 3:
            import cv2
            grayscale_frames = []
            for frame in frames:
                grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                grayscale_frames.append(grayscale_frame)

            grayscale_array = np.stack(grayscale_frames, axis=0)
            frames = np.expand_dims(grayscale_array, axis=-1)

        if self.channels == 3 and frames.shape[-1] == 1:
            frames = np.repeat(frames, 3, axis=-1)
        return frames

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        file_path, label_index = self.samples[index]
        frames = np.load(file_path)
        frames = self._resize_sequence_length(frames)
        frames = self._resize_frames(frames)
        frames = self._match_channel_count(frames)
        frames = frames.astype(np.float32) / 255.0
        frame_tensor = torch.from_numpy(frames).float()
        if self.use_time_first_layout:
            frame_tensor = frame_tensor.permute(0, 3, 1, 2)
        else:
            frame_tensor = frame_tensor.permute(3, 0, 1, 2)
        return frame_tensor, label_index


def save_confusion_matrix_plot(
    confusion: np.ndarray,
    class_labels: List[str],
    save_path: Path,
) -> None:
    """Save a row-normalized confusion matrix heatmap."""
    row_totals = confusion.sum(axis=1, keepdims=True)
    normalized_confusion = np.where(
        row_totals > 0,
        confusion / row_totals,
        0.0,
    )
    n = len(class_labels)
    figure, axis = plt.subplots(figsize=(22, 20))
    image = axis.imshow(
        normalized_confusion,
        interpolation="nearest",
        cmap="Blues",
        vmin=0,
        vmax=1,
    )
    cbar = figure.colorbar(image, ax=axis, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(labelsize=11)
    axis.set_title(
        "Confusion Matrix (row-normalised recall)",
        fontsize=16,
        pad=14,
    )
    axis.set_xlabel("Predicted", fontsize=13, labelpad=10)
    axis.set_ylabel("True", fontsize=13, labelpad=10)
    tick_positions = np.arange(n)
    axis.set_xticks(tick_positions)
    axis.set_yticks(tick_positions)
    axis.set_xticklabels(class_labels, rotation=60, ha="right", fontsize=10)
    axis.set_yticklabels(class_labels, fontsize=10)
    for true_index in range(n):
        for pred_index in range(n):
            pct = normalized_confusion[true_index, pred_index]
            if true_index == pred_index or pct < 0.05:
                continue
            text_color = "white" if pct > 0.5 else "black"
            axis.text(
                pred_index,
                true_index,
                f"{pct * 100:.0f}%",
                ha="center",
                va="center",
                color=text_color,
                fontsize=7.5,
                fontweight="bold",
            )
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[List[int], List[int], List[float]]:
    """Run model evaluation and return labels, predictions, confidences"""
    true_labels: List[int] = []
    predicted_labels: List[int] = []
    confidence_scores: List[float] = []
    model.eval()
    with torch.no_grad():
        for frame_batch, label_batch in dataloader:
            frame_batch = frame_batch.to(device)
            label_batch = label_batch.to(device)
            logits = model(frame_batch)
            probabilities = torch.softmax(logits, dim=1)
            batch_predictions = torch.argmax(probabilities, dim=1)
            batch_confidences = torch.max(probabilities, dim=1).values
            true_labels.extend(label_batch.cpu().numpy().tolist())
            predicted_labels.extend(batch_predictions.cpu().numpy().tolist())
            confidence_scores.extend(batch_confidences.cpu().numpy().tolist())
    return true_labels, predicted_labels, confidence_scores


def write_evaluation_outputs(
    output_directory: Path,
    class_labels: List[str],
    true_labels: List[int],
    predicted_labels: List[int],
) -> EvaluationOutputPaths:
    """Write report text, metrics JSON, and confusion matrix image."""
    output_directory.mkdir(parents=True, exist_ok=True)
    report_text = classification_report(
        true_labels,
        predicted_labels,
        target_names=class_labels,
        digits=4,
        zero_division=0,
    )
    report_path = output_directory / "classification_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    # Confusion matrix for only multi-character labels
    word_indices = [
        i for i, lbl in enumerate(class_labels) if len(lbl) > 1
    ]
    word_labels = [class_labels[i] for i in word_indices]
    word_index_set = set(word_indices)
    # Remap to word-only subset
    old_to_new = {old: new for new, old in enumerate(word_indices)}
    filtered_pairs = [
        (old_to_new[t], old_to_new[p])
        for t, p in zip(true_labels, predicted_labels)
        if t in word_index_set and p in word_index_set
    ]
    if filtered_pairs:
        f_true, f_pred = zip(*filtered_pairs)
        confusion = confusion_matrix(
            list(f_true), list(f_pred),
            labels=list(range(len(word_labels))),
        )
    else:
        confusion = confusion_matrix(true_labels, predicted_labels)
        word_labels = class_labels
    confusion_path = output_directory / "confusion_matrix.png"
    save_confusion_matrix_plot(confusion, word_labels, confusion_path)
    report_dict = classification_report(
        true_labels,
        predicted_labels,
        target_names=class_labels,
        digits=4,
        zero_division=0,
        output_dict=True,
    )

    metrics_payload = {
        "accuracy": float(report_dict.get("accuracy", 0.0)),
        "macro_f1": float(
            report_dict.get("macro avg", {}).get("f1-score", 0.0)
        ),
        "weighted_f1": float(
            report_dict.get("weighted avg", {}).get("f1-score", 0.0)
        ),
        "num_samples": len(true_labels),
        "class_labels": class_labels,
    }

    metrics_path = output_directory / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics_payload, metrics_file, indent=2)

    return EvaluationOutputPaths(
        metrics_json_path=metrics_path,
        classification_report_path=report_path,
        confusion_matrix_path=confusion_path,
    )


def validate_paths(checkpoint_path: Path, data_root: Path) -> None:
    """Validate required input paths."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")


def load_model_and_labels(
    task_name: str,
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[torch.nn.Module, List[str], int, bool]:
    """Load model, class labels, channels and tensor layout settings"""
    if task_name == "word":
        model, class_labels = load_word_model(checkpoint_path, device)
        return model, class_labels, 1, False

    model, class_labels = load_viseme_model(checkpoint_path, device)
    return model, class_labels, 1, True


def main() -> None:
    args = parse_arguments()
    checkpoint_path = Path(args.checkpoint)
    data_root = Path(args.data_root)
    output_directory = Path(args.outdir)
    validate_paths(checkpoint_path, data_root)
    device = resolve_device(args.device)
    print(f"Using device: {device}")
    model, class_labels, channels, use_time_first_layout = (
        load_model_and_labels(args.task, checkpoint_path, device)
    )

    if args.task == "word":
        from training.train_word_recognition_3d_cnn import (
            WordLipReadingDataset,
            load_word_dataset,
        )
        sample_list, word_to_index, _ = load_word_dataset(data_root)
        dataset = WordLipReadingDataset(
            samples=sample_list,
            word_to_index_mapping=word_to_index,
            target_frame_count=args.frames,
            target_frame_height=args.height,
            target_frame_width=args.width,
            is_training=False,
        )
    else:
        dataset = NpyClassificationDataset(
            data_root=data_root,
            class_labels=class_labels,
            sequence_length=args.frames,
            frame_height=args.height,
            frame_width=args.width,
            channels=channels,
            use_time_first_layout=use_time_first_layout,
        )

    if len(dataset) == 0:
        raise RuntimeError(
            f"No .npy samples found under {data_root}. "
            "Expected class folders."
        )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"Evaluating {args.task} model on {len(dataset)} samples...")
    true_labels, predicted_labels, _ = evaluate_model(
        model,
        dataloader,
        device,
    )

    output_paths = write_evaluation_outputs(
        output_directory,
        class_labels,
        true_labels,
        predicted_labels,
    )

    print("Saved")
    print(f"  - {output_paths.metrics_json_path}")
    print(f"  - {output_paths.classification_report_path}")
    print(f"  - {output_paths.confusion_matrix_path}")


if __name__ == "__main__":
    main()
