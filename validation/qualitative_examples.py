"""
Export qualitative examples from an evaluation run
Saves top-k predictions for each sample and exports misclassified
samples to JSON
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))


class FlatNpyDataset(Dataset):
    """Dataset that returns .npy file paths and numeric labels"""
    def __init__(self, data_root: Path, class_labels: List[str]) -> None:
        self.data_root = data_root
        self.class_labels = class_labels
        self.label_to_index = {
            label: index for index, label in enumerate(class_labels)
        }
        self.samples = self._scan_samples()

    def _scan_samples(self) -> List[Tuple[Path, int]]:
        """Collect all .npy samples from class subdirectories."""
        samples: List[Tuple[Path, int]] = []
        for class_label in self.class_labels:
            class_directory = self.data_root / class_label
            if not class_directory.exists():
                continue
            for npy_path in sorted(class_directory.glob("*.npy")):
                samples.append((npy_path, self.label_to_index[class_label]))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Path, int]:
        return self.samples[index]


def resolve_device(device_name: str) -> torch.device:
    """Resolve the requested torch device"""
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_name)


def load_model_and_labels(
    task_name: str,
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[torch.nn.Module, List[str], int, bool]:
    """Load model, class labels, channel count, and tensor layout."""
    checkpoint_data = torch.load(checkpoint_path, map_location=device)
    if task_name == "word":
        from training.train_word_recognition_3d_cnn import (
            ThreeDimensionalCNN,)
        word_to_index = checkpoint_data["word_to_index"]
        raw_index_to_word = checkpoint_data["index_to_word"]
        index_to_word = {
            int(index): word
            for index, word in raw_index_to_word.items()}
        class_labels = [
            index_to_word[index]
            for index in range(len(index_to_word))]
        model = ThreeDimensionalCNN(
            input_channels=3,
            num_classes=len(word_to_index),)
        channels = 3
        use_time_first_layout = False
    else:
        from training.train_viseme_recognition_bozkurt_3d_cnn import (
            ThreeDimensionalCNN,)
        class_labels = checkpoint_data["viseme_classes"]
        model = ThreeDimensionalCNN(
            number_of_classes=len(class_labels),
            input_channels=1,)
        channels = 1
        use_time_first_layout = True
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_labels, channels, use_time_first_layout


def load_and_preprocess_npy(
    npy_path: Path,
    frame_count: int,
    frame_height: int,
    frame_width: int,
    channels: int,
    use_time_first_layout: bool = False,
) -> torch.Tensor:
    """Load a npy clip and preprocess it to match model input"""
    import cv2
    frames = np.load(npy_path)
    current_frame_count = frames.shape[0]
    if current_frame_count >= frame_count:
        frame_indices = np.linspace(
            0,
            current_frame_count - 1,
            frame_count,
            dtype=int,)
        frames = frames[frame_indices]
    else:
        frame_indices = [
            index % current_frame_count
            for index in range(frame_count)]
        frames = frames[frame_indices]
    resized_frames = [
        cv2.resize(frame, (frame_width, frame_height))
        for frame in frames]
    frames = np.stack(resized_frames, axis=0)
    if frames.ndim == 3:
        frames = np.expand_dims(frames, axis=-1)
    if channels == 1 and frames.shape[-1] == 3:
        grayscale_frames = [
            cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            for frame in frames]
        frames = np.expand_dims(
            np.stack(grayscale_frames, axis=0),
            axis=-1,)
    if channels == 3 and frames.shape[-1] == 1:
        frames = np.repeat(frames, 3, axis=-1)
    frames = frames.astype(np.float32) / 255.0
    frame_tensor = torch.from_numpy(frames).float()
    if use_time_first_layout:
        frame_tensor = frame_tensor.permute(0, 3, 1, 2)
    else:
        frame_tensor = frame_tensor.permute(3, 0, 1, 2)
    return frame_tensor.unsqueeze(0)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Export qualitative prediction examples.",)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--task",
        type=str,
        choices=["word", "viseme"],
        required=True,)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-items", type=int, default=300)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    """Export top-k predictions and misclassified samples"""
    args = parse_arguments()
    args.outdir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    model, class_labels, channels, use_time_first_layout = (
        load_model_and_labels(
            args.task,
            args.checkpoint,
            device,))
    dataset = FlatNpyDataset(args.data_root, class_labels)
    example_rows: List[Dict[str, object]] = []
    misclassified_rows: List[Dict[str, object]] = []
    max_items = min(args.max_items, len(dataset))
    top_k = min(args.top_k, len(class_labels))
    for sample_index in range(max_items):
        npy_path, true_index = dataset[sample_index]
        video_tensor = load_and_preprocess_npy(
            npy_path=npy_path,
            frame_count=args.frames,
            frame_height=args.height,
            frame_width=args.width,
            channels=channels,
            use_time_first_layout=use_time_first_layout,
        ).to(device)
        with torch.no_grad():
            logits = model(video_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
        top_probabilities, top_indices = torch.topk(probabilities, k=top_k)
        top_k_predictions = [
            {"label": class_labels[int(class_index.item())],
                "probability": float(probability_value.item()), }
            for probability_value, class_index in zip(
                top_probabilities,
                top_indices,)]
        predicted_index = int(torch.argmax(probabilities).item())
        example_entry = {
            "npy_path": str(npy_path),
            "true_label": class_labels[true_index],
            "predicted_label": class_labels[predicted_index],
            "top_k": top_k_predictions, }
        example_rows.append(example_entry)
        if predicted_index != true_index:
            misclassified_rows.append(example_entry)
    examples_path = args.outdir / "examples_topk.json"
    misclassified_path = args.outdir / "misclassified.json"
    examples_path.write_text(
        json.dumps(example_rows, indent=2),
        encoding="utf-8",)
    misclassified_path.write_text(
        json.dumps(misclassified_rows, indent=2),
        encoding="utf-8",)
    print(f"Saved: {examples_path}")
    print(f"Saved: {misclassified_path}")
    print(
        f"Processed {len(example_rows)} samples, "
        f"misclassified {len(misclassified_rows)}")


if __name__ == "__main__":
    main()
