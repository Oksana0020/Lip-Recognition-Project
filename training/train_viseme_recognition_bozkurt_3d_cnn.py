"""
Bozkurt Viseme Recognition 3D CNN Training Script

This script trains a 3D Convolutional Neural Network to recognize
16 viseme classes using Bozkurt viseme mapping system.
 Visemes group phonemes with similar mouth shapes.
Bozkurt System: 16 viseme classes (S, V2-V16)
"""

import json
import os
import csv
import argparse
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import importlib.util

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def _load_split_dataset_utils():
    """Lazy-load dataset splitting utilities from preprocessing module."""
    preprocessing_dir = Path(__file__).parent.parent / "preprocessing"
    spec = importlib.util.spec_from_file_location(
        "dataset_splitting_utils",
        preprocessing_dir / "dataset_splitting_utils.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_split_utils = None


def split_dataset_into_train_val_test(
    full_dataset,
    train_ratio,
    val_ratio,
    test_ratio,
    random_seed,
):
    """Wrapper to call split utility function."""
    global _split_utils
    if _split_utils is None:
        _split_utils = _load_split_dataset_utils()
    return _split_utils.split_dataset_into_train_val_test(
        full_dataset, train_ratio, val_ratio, test_ratio, random_seed
    )


class BozkurtVisemeTrainingConfig:
    """Configuration for Bozkurt viseme recognition training."""

    def __init__(self):
        self.mapping_csv_path = "mapping/bozkurt_viseme_map.csv"

        preferred_data_roots = [
            "data/processed/visemes_bozkurt_mfa_balanced_npy",
            "data/processed/visemes_bozkurt_mfa_npy",
            "data/processed/visemes_bozkurt_mfa",
        ]
        self.data_root_directory = preferred_data_roots[-1]

        for candidate_root in preferred_data_roots:
            if os.path.isdir(candidate_root):
                self.data_root_directory = candidate_root
                break

        if self.data_root_directory.endswith("_balanced_npy"):
            print(
                "Using balanced precomputed viseme .npy dataset: "
                f"{self.data_root_directory}"
            )
        elif self.data_root_directory.endswith("_npy"):
            print(
                "Using precomputed viseme .npy dataset: "
                f"{self.data_root_directory}"
            )
        else:
            print(
                "Precomputed .npy viseme dataset not found; "
                "falling back to JSON+video extraction dataset"
            )

        self.phoneme_intervals_root_directory = "data/processed/phonemes_mfa"

        self.video_height_pixels = 48
        self.video_width_pixels = 48
        self.sequence_length_frames = 16
        self.color_channels = 1

        self.batch_size_samples = 8
        self.num_data_loading_workers = 0
        self.number_of_epochs = 30
        self.learning_rate = 0.001
        self.weight_decay_l2 = 0.0001
        self.reduce_lr_on_plateau_patience = 3
        self.early_stopping_patience = 8

        self.training_data_ratio = 0.7
        self.validation_data_ratio = 0.15
        self.test_data_ratio = 0.15

        self.checkpoint_save_directory = (
            "training/checkpoints_bozkurt_viseme"
        )
        self.save_checkpoint_every_n_epochs = 5
        self.keep_best_model_only = True

        self.tensorboard_log_directory = "training/runs_bozkurt_viseme"
        self.print_training_progress_every_n_batches = 10

        self.metrics_output_directory = "training/results_bozkurt_viseme"
        self.training_history_filename = "training_history.json"
        self.final_metrics_filename = "final_test_metrics.json"

        self.use_gpu_if_available = True
        self.device = self._setup_device()

        self.random_seed = 42
        self.number_of_classes = 16

    def _setup_device(self) -> torch.device:
        """Configure GPU or CPU device for training."""
        if self.use_gpu_if_available and torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")

            return device

        if self.use_gpu_if_available:
            try:
                if hasattr(torch, "xpu") and torch.xpu.is_available():
                    device = torch.device("xpu")
                    print("Using Intel XPU backend")
                    return device
            except Exception:
                pass

            try:
                torch_directml = importlib.import_module(
                    "torch_directml"
                )
                device = torch_directml.device()
                if self._directml_supports_conv3d(device):
                    print("Using DirectML GPU backend")
                    return device

                print(
                    "DirectML is available but Conv3D is not supported "
                    "for this model; falling back to CPU"
                )
            except Exception:
                pass

        device = torch.device("cpu")
        print("Using CPU for training")
        return device

    def _directml_supports_conv3d(self, device: Any) -> bool:
        """Check whether Conv3D works on DirectML backend."""
        try:
            test_input = torch.zeros(
                (1, 1, 4, 8, 8),
                dtype=torch.float32,
                device=device,
            )
            test_conv = nn.Conv3d(
                in_channels=1,
                out_channels=2,
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1),
            ).to(device)
            _ = test_conv(test_input)
            return True
        except Exception:
            return False


class BozkurtVisemeLipReadingDataset(Dataset):
    """
    Dataset loader for Bozkurt viseme lip reading videos.
    Each sample is a sequence of grayscale lip-region frames.
    """

    def __init__(
        self,
        data_root_directory: str,
        viseme_class_labels: List[str],
        sequence_length_frames: int = 50,
        target_height_pixels: int = 64,
        target_width_pixels: int = 64,
        phoneme_intervals_root_directory: Optional[str] = None,
        data_split: str = "train",
    ):
        self.data_root_directory = data_root_directory
        self.phoneme_intervals_root_directory = (
            phoneme_intervals_root_directory
        )
        self.viseme_class_labels = sorted(viseme_class_labels)
        self.sequence_length_frames = sequence_length_frames
        self.target_height_pixels = target_height_pixels
        self.target_width_pixels = target_width_pixels
        self.data_split = data_split
        self.video_path_cache: Dict[str, str] = {}

        self.viseme_label_to_index = {
            label: idx for idx, label in enumerate(self.viseme_class_labels)
        }

        self.video_samples_list = self._load_all_samples()
        print(
            f"Loaded {len(self.video_samples_list)} samples "
            f"for {data_split} split"
        )

    def _load_all_samples(self) -> List[Dict]:
        """Scan directory and load all video sample paths with labels."""
        all_samples: List[Dict] = []

        for viseme_label in self.viseme_class_labels:
            viseme_folder_path = os.path.join(
                self.data_root_directory,
                viseme_label,
            )

            if not os.path.exists(viseme_folder_path):
                print(
                    f"Warning: Folder not found for viseme "
                    f"'{viseme_label}'"
                )
                continue

            for filename in os.listdir(viseme_folder_path):
                if not filename.endswith(".json"):
                    continue

                npy_filename = filename.replace(".json", ".npy")
                npy_file_path = os.path.join(
                    viseme_folder_path,
                    npy_filename,
                )
                json_file_path = os.path.join(
                    viseme_folder_path,
                    filename,
                )

                metadata = self._safe_load_json(json_file_path)
                if metadata is None:
                    continue

                source_video_path: Optional[str] = None
                if os.path.exists(npy_file_path):
                    source_video_path = npy_file_path
                else:
                    source_video_path = self._resolve_source_video_path(
                        filename,
                        metadata,
                    )
                    if source_video_path is None:
                        continue

                sample_info = {
                    "video_file_path": source_video_path,
                    "metadata_file_path": json_file_path,
                    "start_time": float(metadata.get("start_time", 0.0)),
                    "end_time": float(metadata.get("end_time", 0.0)),
                    "viseme_label": viseme_label,
                    "viseme_class_index": (
                        self.viseme_label_to_index[viseme_label]
                    ),
                }
                all_samples.append(sample_info)

        return all_samples

    def _safe_load_json(self, json_path: str) -> Optional[Dict]:
        try:
            with open(json_path, "r", encoding="utf-8") as json_file:
                return json.load(json_file)
        except Exception as error:
            print(f"Warning: Failed to parse metadata {json_path}: {error}")
            return None

    def _resolve_source_video_path(
        self,
        metadata_filename: str,
        metadata: Dict,
    ) -> Optional[str]:
        embedded_video = metadata.get("video")
        if isinstance(embedded_video, str) and os.path.exists(embedded_video):
            return embedded_video

        if not self.phoneme_intervals_root_directory:
            return None

        try:
            base_video_id = metadata_filename.rsplit("_", 2)[0]
        except Exception:
            return None

        if base_video_id in self.video_path_cache:
            return self.video_path_cache[base_video_id]

        interval_file_path = os.path.join(
            self.phoneme_intervals_root_directory,
            f"{base_video_id}_phoneme_intervals_mfa.json",
        )
        if not os.path.exists(interval_file_path):
            return None

        interval_json = self._safe_load_json(interval_file_path)
        if interval_json is None:
            return None

        video_path = interval_json.get("video")
        if isinstance(video_path, str) and os.path.exists(video_path):
            self.video_path_cache[base_video_id] = video_path
            return video_path

        return None

    def __len__(self) -> int:
        return len(self.video_samples_list)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        sample_info = self.video_samples_list[index]

        try:
            if sample_info["video_file_path"].lower().endswith(".npy"):
                video_frames_array = np.load(
                    sample_info["video_file_path"],
                    allow_pickle=False
                )
                if video_frames_array.size == 0:
                    raise ValueError(
                        f"Empty npy file: {sample_info['video_file_path']}"
                    )
            else:
                video_frames_array = self._extract_video_segment_frames(
                    video_path=sample_info["video_file_path"],
                    start_time_seconds=sample_info["start_time"],
                    end_time_seconds=sample_info["end_time"],
                )
        except (ValueError, OSError, FileNotFoundError):
            return self._get_fallback_sample(sample_info)

        processed_frames = self._process_video_frames(video_frames_array)

        video_tensor = torch.FloatTensor(processed_frames)
        label_index = sample_info["viseme_class_index"]
        return video_tensor, label_index

    def _get_fallback_sample(
        self, sample_info: Dict
    ) -> Tuple[torch.Tensor, int]:
        """Return a zero-padded fallback for corrupted files."""
        fallback_frames = np.zeros(
            (
                self.sequence_length_frames,
                self.target_height_pixels,
                self.target_width_pixels,
                1,
            ),
            dtype=np.float32
        )
        video_tensor = torch.FloatTensor(fallback_frames)
        label_index = sample_info["viseme_class_index"]
        return video_tensor, label_index

    def _extract_video_segment_frames(
        self,
        video_path: str,
        start_time_seconds: float,
        end_time_seconds: float,
    ) -> np.ndarray:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps is None or fps <= 0:
            fps = 25.0

        start_frame_idx = max(0, int(start_time_seconds * fps))
        end_frame_idx = max(start_frame_idx, int(end_time_seconds * fps))

        frames: List[np.ndarray] = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)

        for _ in range(start_frame_idx, end_frame_idx + 1):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()

        if not frames:
            return np.zeros(
                (
                    1,
                    self.target_height_pixels,
                    self.target_width_pixels,
                    3,
                ),
                dtype=np.uint8,
            )

        return np.array(frames)

    def _process_video_frames(
        self,
        video_frames_array: np.ndarray,
    ) -> np.ndarray:
        """Resize, grayscale, normalize, and pad/sample to fixed length."""
        current_num_frames = video_frames_array.shape[0]

        if current_num_frames >= self.sequence_length_frames:
            frame_indices = np.linspace(
                0,
                current_num_frames - 1,
                self.sequence_length_frames,
                dtype=int,
            )
            selected_frames = video_frames_array[frame_indices]
        else:
            padding_needed = self.sequence_length_frames - current_num_frames
            last_frame = video_frames_array[-1:].repeat(
                padding_needed,
                axis=0,
            )
            selected_frames = np.concatenate(
                [video_frames_array, last_frame],
                axis=0,
            )

        processed_frames_list: List[np.ndarray] = []
        for frame in selected_frames:
            resized_frame = cv2.resize(
                frame,
                (self.target_width_pixels, self.target_height_pixels),
            )

            if len(resized_frame.shape) == 3:
                grayscale_frame = cv2.cvtColor(
                    resized_frame,
                    cv2.COLOR_RGB2GRAY,
                )
            else:
                grayscale_frame = resized_frame

            frame_normalized = grayscale_frame.astype(np.float32) / 255.0
            frame_with_channel = np.expand_dims(frame_normalized, axis=0)
            processed_frames_list.append(frame_with_channel)

        processed_video = np.stack(processed_frames_list, axis=0)
        return processed_video


def load_expected_viseme_classes(mapping_csv_path: str) -> List[str]:
    """Load expected viseme class labels from mapping CSV."""
    if not os.path.exists(mapping_csv_path):
        return []

    expected_classes: List[str] = []
    with open(mapping_csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            viseme_label = str(row.get("viseme_class", "")).strip()
            if viseme_label:
                expected_classes.append(viseme_label)

    return sorted(set(expected_classes))


def compute_class_weights(
    train_dataset,
    number_of_classes: int,
) -> torch.Tensor:
    """Compute inverse-frequency class weights from a train subset."""
    class_counts = np.zeros(number_of_classes, dtype=np.float64)

    subset_indices = getattr(train_dataset, "indices", None)
    base_dataset = getattr(train_dataset, "dataset", train_dataset)

    if subset_indices is None:
        subset_indices = list(range(len(base_dataset)))

    for sample_index in subset_indices:
        label_index = base_dataset.video_samples_list[sample_index][
            "viseme_class_index"
        ]
        class_counts[label_index] += 1.0

    safe_counts = np.maximum(class_counts, 1.0)
    total_count = safe_counts.sum()
    weights = total_count / (number_of_classes * safe_counts)
    normalized_weights = weights / weights.mean()

    return torch.tensor(normalized_weights, dtype=torch.float32)


def load_checkpoint_into_model(model: nn.Module, checkpoint_path: str) -> None:
    """Load model weights from a saved checkpoint path."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)


class ThreeDimensionalCNN(nn.Module):
    """
    3D CNN for viseme recognition from video sequences.
    Input shape: (batch, time, channels, height, width)
    """

    def __init__(
        self,
        number_of_classes: int,
        input_channels: int = 1,
        dropout_probability: float = 0.5,
    ):
        super(ThreeDimensionalCNN, self).__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv3d(
                in_channels=input_channels,
                out_channels=32,
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1),
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv3d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1),
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv3d(
                in_channels=64,
                out_channels=128,
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1),
            ),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        )

        self.conv_block_4 = nn.Sequential(
            nn.Conv3d(
                in_channels=128,
                out_channels=256,
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1),
            ),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 4, 4))

        self.fully_connected_classifier = nn.Sequential(
            nn.Linear(256 * 1 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_probability),
            nn.Linear(512, number_of_classes),
        )

    def forward(self, input_video_tensor: torch.Tensor) -> torch.Tensor:
        video_batch = input_video_tensor.permute(0, 2, 1, 3, 4)
        features_1 = self.conv_block_1(video_batch)
        features_2 = self.conv_block_2(features_1)
        features_3 = self.conv_block_3(features_2)
        features_4 = self.conv_block_4(features_3)
        pooled_features = self.adaptive_pool(features_4)

        flattened_features = pooled_features.view(
            pooled_features.size(0),
            -1,
        )
        class_logits = self.fully_connected_classifier(flattened_features)
        return class_logits


def train_one_epoch(
    model: nn.Module,
    train_dataloader: DataLoader,
    loss_function: nn.Module,
    optimizer: optim.Optimizer,
    device: Any,
    epoch_number: int,
    print_every_n_batches: int,
) -> Dict[str, float]:
    """Train model for one epoch and return average loss and accuracy."""
    model.train()

    total_loss_sum = 0.0
    correct_predictions_count = 0
    total_samples_count = 0

    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch_number} [Train]")

    for batch_index, (video_batch, label_batch) in enumerate(progress_bar):
        video_batch = video_batch.to(device)
        label_batch = label_batch.to(device)

        optimizer.zero_grad()
        prediction_logits = model(video_batch)
        loss = loss_function(prediction_logits, label_batch)
        loss.backward()
        optimizer.step()

        predicted_classes = torch.argmax(prediction_logits, dim=1)
        correct_predictions = (predicted_classes == label_batch).sum().item()

        total_loss_sum += loss.item() * video_batch.size(0)
        correct_predictions_count += correct_predictions
        total_samples_count += video_batch.size(0)

        if (batch_index + 1) % print_every_n_batches == 0:
            current_avg_loss = total_loss_sum / total_samples_count
            current_avg_accuracy = (
                100.0 * correct_predictions_count / total_samples_count
            )
            progress_bar.set_postfix(
                {
                    "loss": f"{current_avg_loss:.4f}",
                    "acc": f"{current_avg_accuracy:.2f}%",
                }
            )

    if total_samples_count == 0:
        return {"loss": 0.0, "accuracy": 0.0}

    epoch_average_loss = total_loss_sum / total_samples_count
    epoch_average_accuracy = 100.0 * correct_predictions_count / (
        total_samples_count
    )

    return {
        "loss": epoch_average_loss,
        "accuracy": epoch_average_accuracy,
    }


def validate_model(
    model: nn.Module,
    val_dataloader: DataLoader,
    loss_function: nn.Module,
    device: Any,
    epoch_number: int,
) -> Dict[str, float]:
    """Validate model and return average loss and accuracy."""
    model.eval()

    total_loss_sum = 0.0
    correct_predictions_count = 0
    total_samples_count = 0

    progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch_number} [Val]")

    with torch.no_grad():
        for video_batch, label_batch in progress_bar:
            video_batch = video_batch.to(device)
            label_batch = label_batch.to(device)

            prediction_logits = model(video_batch)
            loss = loss_function(prediction_logits, label_batch)

            predicted_classes = torch.argmax(prediction_logits, dim=1)
            correct_predictions = (
                predicted_classes == label_batch
            ).sum().item()

            total_loss_sum += loss.item() * video_batch.size(0)
            correct_predictions_count += correct_predictions
            total_samples_count += video_batch.size(0)

    if total_samples_count == 0:
        return {"loss": 0.0, "accuracy": 0.0}

    val_average_loss = total_loss_sum / total_samples_count
    val_average_accuracy = 100.0 * correct_predictions_count / (
        total_samples_count
    )

    print(
        "Validation - "
        f"Loss: {val_average_loss:.4f}, "
        f"Accuracy: {val_average_accuracy:.2f}%"
    )

    return {
        "loss": val_average_loss,
        "accuracy": val_average_accuracy,
    }


def save_model_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch_number: int,
    val_accuracy: float,
    save_directory: str,
    checkpoint_filename: str,
    viseme_classes: List[str],
) -> None:
    """Save model checkpoint to disk."""
    os.makedirs(save_directory, exist_ok=True)
    checkpoint_path = os.path.join(save_directory, checkpoint_filename)

    checkpoint_data = {
        "epoch": epoch_number,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_accuracy": val_accuracy,
        "viseme_classes": viseme_classes,
    }

    torch.save(checkpoint_data, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def save_training_results(
    output_directory: str,
    training_history: List[Dict[str, float]],
    final_test_metrics: Dict[str, float],
    viseme_classes: List[str],
) -> None:
    """Save training history and final test metrics to JSON files."""
    os.makedirs(output_directory, exist_ok=True)

    history_path = os.path.join(output_directory, "training_history.json")
    metrics_path = os.path.join(output_directory, "final_test_metrics.json")

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "viseme_classes": viseme_classes,
                "epochs": training_history,
            },
            f,
            indent=2,
        )

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "viseme_classes": viseme_classes,
                "test_metrics": final_test_metrics,
            },
            f,
            indent=2,
        )

    print(f"Training history saved to: {history_path}")
    print(f"Final test metrics saved to: {metrics_path}")


def main_training_pipeline(
    resume_checkpoint: Optional[str] = None,
    fresh_start: bool = False,
):
    """Run the complete training pipeline for Bozkurt viseme recognition."""
    print("=" * 80)
    print("BOZKURT VISEME RECOGNITION - 3D CNN TRAINING")
    print("=" * 80)

    config = BozkurtVisemeTrainingConfig()

    if fresh_start:
        for path_to_reset in (
            config.checkpoint_save_directory,
            config.metrics_output_directory,
            config.tensorboard_log_directory,
        ):
            if os.path.isdir(path_to_reset):
                shutil.rmtree(path_to_reset)
                print(f"Reset previous run artifacts: {path_to_reset}")

    os.makedirs(config.checkpoint_save_directory, exist_ok=True)
    os.makedirs(config.metrics_output_directory, exist_ok=True)

    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)

    bozkurt_viseme_classes = sorted(
        [
            folder_name
            for folder_name in os.listdir(config.data_root_directory)
            if os.path.isdir(
                os.path.join(config.data_root_directory, folder_name)
            )
        ]
    )

    if len(bozkurt_viseme_classes) == 0:
        raise RuntimeError(
            "No viseme class folders found in "
            f"{config.data_root_directory}"
        )

    expected_viseme_classes = load_expected_viseme_classes(
        config.mapping_csv_path
    )
    missing_viseme_classes = sorted(
        set(expected_viseme_classes) - set(bozkurt_viseme_classes)
    )

    if missing_viseme_classes:
        print(
            "Warning: Expected viseme classes missing from dataset: "
            f"{missing_viseme_classes}"
        )

    config.number_of_classes = len(bozkurt_viseme_classes)

    print(f"\nNumber of viseme classes: {len(bozkurt_viseme_classes)}")
    print(f"Viseme classes: {bozkurt_viseme_classes}")

    print("\n" + "-" * 80)
    print("Loading dataset...")
    print("-" * 80)

    full_dataset = BozkurtVisemeLipReadingDataset(
        data_root_directory=config.data_root_directory,
        viseme_class_labels=bozkurt_viseme_classes,
        sequence_length_frames=config.sequence_length_frames,
        target_height_pixels=config.video_height_pixels,
        target_width_pixels=config.video_width_pixels,
        phoneme_intervals_root_directory=(
            config.phoneme_intervals_root_directory
        ),
    )

    if len(full_dataset) == 0:
        raise RuntimeError(
            "No valid training samples found. "
            "Expected either .npy files in viseme folders or "
            "resolvable source videos via phoneme interval JSON files."
        )

    (
        train_dataset,
        val_dataset,
        test_dataset,
    ) = split_dataset_into_train_val_test(
        full_dataset=full_dataset,
        train_ratio=config.training_data_ratio,
        val_ratio=config.validation_data_ratio,
        test_ratio=config.test_data_ratio,
        random_seed=config.random_seed,
    )

    default_workers = max(0, int(config.num_data_loading_workers))
    use_pin_memory = isinstance(config.device, torch.device) and (
        config.device.type == "cuda"
    )

    common_loader_kwargs = {
        "batch_size": config.batch_size_samples,
        "num_workers": default_workers,
        "pin_memory": use_pin_memory,
    }
    if default_workers > 0:
        common_loader_kwargs["persistent_workers"] = True

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        **common_loader_kwargs,
    )
    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        **common_loader_kwargs,
    )
    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        **common_loader_kwargs,
    )

    print(
        "DataLoader workers: "
        f"{default_workers}, pin_memory={use_pin_memory}"
    )

    print("\n" + "-" * 80)
    print("Initializing model...")
    print("-" * 80)

    model = ThreeDimensionalCNN(
        number_of_classes=config.number_of_classes,
        input_channels=config.color_channels,
    ).to(config.device)

    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")

    class_weights = compute_class_weights(
        train_dataset=train_dataset,
        number_of_classes=config.number_of_classes,
    ).to(config.device)
    print(f"Class weights: {class_weights.cpu().numpy().round(3).tolist()}")

    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay_l2,
    )
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=config.reduce_lr_on_plateau_patience,
    )

    tensorboard_writer = SummaryWriter(
        log_dir=config.tensorboard_log_directory,
    )

    print("\n" + "-" * 80)
    print("Starting training...")
    print("-" * 80)

    best_validation_accuracy = 0.0
    best_checkpoint_path = os.path.join(
        config.checkpoint_save_directory,
        "bozkurt_viseme_best_model.pth",
    )
    start_epoch = 1
    epochs_without_improvement = 0
    training_history: List[Dict[str, float]] = []

    if resume_checkpoint:
        if not os.path.exists(resume_checkpoint):
            raise FileNotFoundError(
                f"Resume checkpoint not found: {resume_checkpoint}"
            )

        checkpoint = torch.load(resume_checkpoint, map_location="cpu")
        checkpoint_state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(checkpoint_state_dict)

        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)

        checkpoint_epoch = int(checkpoint.get("epoch", 0))
        start_epoch = checkpoint_epoch + 1
        best_validation_accuracy = float(
            checkpoint.get("val_accuracy", 0.0)
        )

        checkpoint_viseme_classes = checkpoint.get("viseme_classes")
        if isinstance(checkpoint_viseme_classes, list):
            if checkpoint_viseme_classes != bozkurt_viseme_classes:
                print(
                    "Warning checkpoint viseme class order does not "
                    "match current dataset order"
                )

        print(
            "Resumed training from checkpoint "
            f"{resume_checkpoint} at epoch {checkpoint_epoch}"
        )

    if start_epoch > config.number_of_epochs:
        print(
            "Checkpoint already reached/exceeded target epochs; "
            "skipping training loop"
        )

    for epoch in range(start_epoch, config.number_of_epochs + 1):
        print(f"\nEpoch {epoch}/{config.number_of_epochs}")
        print("-" * 40)

        train_metrics = train_one_epoch(
            model=model,
            train_dataloader=train_dataloader,
            loss_function=loss_function,
            optimizer=optimizer,
            device=config.device,
            epoch_number=epoch,
            print_every_n_batches=(
                config.print_training_progress_every_n_batches
            ),
        )

        val_metrics = validate_model(
            model=model,
            val_dataloader=val_dataloader,
            loss_function=loss_function,
            device=config.device,
            epoch_number=epoch,
        )

        tensorboard_writer.add_scalar(
            "Loss/train",
            train_metrics["loss"],
            epoch,
        )
        tensorboard_writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
        tensorboard_writer.add_scalar(
            "Accuracy/train",
            train_metrics["accuracy"],
            epoch,
        )
        tensorboard_writer.add_scalar(
            "Accuracy/val",
            val_metrics["accuracy"],
            epoch,
        )
        tensorboard_writer.add_scalar(
            "LearningRate",
            optimizer.param_groups[0]["lr"],
            epoch,
        )

        training_history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_metrics["loss"]),
                "train_accuracy": float(train_metrics["accuracy"]),
                "val_loss": float(val_metrics["loss"]),
                "val_accuracy": float(val_metrics["accuracy"]),
            }
        )

        if epoch % config.save_checkpoint_every_n_epochs == 0:
            checkpoint_filename = f"bozkurt_viseme_epoch_{epoch}.pth"
            save_model_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch_number=epoch,
                val_accuracy=val_metrics["accuracy"],
                save_directory=config.checkpoint_save_directory,
                checkpoint_filename=checkpoint_filename,
                viseme_classes=bozkurt_viseme_classes,
            )

        if val_metrics["accuracy"] > best_validation_accuracy:
            best_validation_accuracy = val_metrics["accuracy"]
            epochs_without_improvement = 0
            save_model_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch_number=epoch,
                val_accuracy=val_metrics["accuracy"],
                save_directory=config.checkpoint_save_directory,
                checkpoint_filename="bozkurt_viseme_best_model.pth",
                viseme_classes=bozkurt_viseme_classes,
            )
            print(
                "New best model saved! "
                f"Validation accuracy: "
                f"{best_validation_accuracy:.2f}%"
            )
        else:
            epochs_without_improvement += 1

        lr_scheduler.step(val_metrics["accuracy"])

        if epochs_without_improvement >= config.early_stopping_patience:
            print(
                "Early stopping triggered after "
                f"{epochs_without_improvement} epochs without improvement"
            )
            break

    print("\n" + "=" * 80)
    print("Evaluating on test set...")
    print("=" * 80)

    if os.path.exists(best_checkpoint_path):
        load_checkpoint_into_model(model, best_checkpoint_path)
        model = model.to(config.device)
        print("Loaded best checkpoint for final test evaluation")

    test_metrics = validate_model(
        model=model,
        val_dataloader=test_dataloader,
        loss_function=loss_function,
        device=config.device,
        epoch_number=config.number_of_epochs,
    )

    final_test_metrics = {
        "loss": float(test_metrics["loss"]),
        "accuracy": float(test_metrics["accuracy"]),
        "best_validation_accuracy": float(best_validation_accuracy),
    }

    print("\nFinal Test Results:")
    print(f"  Loss: {final_test_metrics['loss']:.4f}")
    print(f"  Accuracy: {final_test_metrics['accuracy']:.2f}%")

    save_training_results(
        output_directory=config.metrics_output_directory,
        training_history=training_history,
        final_test_metrics=final_test_metrics,
        viseme_classes=bozkurt_viseme_classes,
    )

    tensorboard_writer.close()

    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print("=" * 80)

    return model


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(
        description="Train Bozkurt viseme recognition model"
    )
    argument_parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file for resuming training",
    )
    argument_parser.add_argument(
        "--resume-best",
        action="store_true",
        help=(
            "Resume from training/checkpoints_bozkurt_viseme/"
            "bozkurt_viseme_best_model.pth"
        ),
    )
    argument_parser.add_argument(
        "--fresh-start",
        action="store_true",
        help=(
            "Delete prior checkpoints/results/tensorboard logs "
            "and start from epoch 1"
        ),
    )

    cli_args = argument_parser.parse_args()

    selected_resume_checkpoint = cli_args.resume_checkpoint
    if cli_args.resume_best:
        selected_resume_checkpoint = (
            "training/checkpoints_bozkurt_viseme/"
            "bozkurt_viseme_best_model.pth"
        )

    if cli_args.fresh_start and selected_resume_checkpoint:
        print(
            "Fresh start requested; ignoring resume checkpoint "
            f"{selected_resume_checkpoint}"
        )
        selected_resume_checkpoint = None

    main_training_pipeline(
        resume_checkpoint=selected_resume_checkpoint,
        fresh_start=cli_args.fresh_start,
    )
