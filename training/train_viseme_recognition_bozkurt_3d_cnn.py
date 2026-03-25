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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def split_dataset_into_train_val_test(
    full_dataset, train_ratio, val_ratio, test_ratio, random_seed
):
    """Split dataset; loads split utils from the preprocessing module."""
    spec = importlib.util.spec_from_file_location(
        "dataset_splitting_utils",
        Path(__file__).parent.parent
        / "preprocessing"
        / "dataset_splitting_utils.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.split_dataset_into_train_val_test(
        full_dataset, train_ratio, val_ratio, test_ratio, random_seed
    )


class BozkurtVisemeTrainingConfig:
    """Configuration for Bozkurt viseme recognition training."""
    def __init__(self):
        self.mapping_csv_path = "mapping/bozkurt_viseme_map.csv"
        self.data_root_directory = (
            "data/processed/visemes_bozkurt_mfa_balanced_npy"
        )
        self.video_height_pixels = 64
        self.video_width_pixels = 64
        self.sequence_length_frames = 8
        self.color_channels = 1
        self.batch_size_samples = 32
        self.num_data_loading_workers = 2
        self.number_of_epochs = 60
        self.learning_rate = 0.0005
        self.weight_decay_l2 = 0.00001
        self.reduce_lr_on_plateau_patience = 5
        self.early_stopping_patience = 15
        self.training_data_ratio = 0.7
        self.validation_data_ratio = 0.15
        self.test_data_ratio = 0.15
        self.checkpoint_save_directory = (
            "training/checkpoints_bozkurt_viseme"
        )
        self.save_checkpoint_every_n_epochs = 10
        self.tensorboard_log_directory = "training/runs_bozkurt_viseme"
        self.print_training_progress_every_n_batches = 10
        self.metrics_output_directory = "training/results_bozkurt_viseme"
        self.device = self._setup_device()
        self.random_seed = 42
        self.number_of_classes = 16

    def _setup_device(self) -> torch.device:
        """Configure GPU or CPU device for training."""
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
        try:
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                print("Using Intel XPU backend")
                return torch.device("xpu")
        except Exception:
            pass
        try:
            torch_directml = importlib.import_module("torch_directml")
            dml_dev = torch_directml.device()
            # Test Conv3D support before committing to DirectML
            _t = torch.zeros((1, 1, 4, 8, 8), device=dml_dev)
            nn.Conv3d(1, 2, (3, 3, 3), padding=(1, 1, 1)).to(dml_dev)(_t)
            print("Using DirectML GPU backend")
            return dml_dev
        except Exception:
            pass
        print("Using CPU for training")
        return torch.device("cpu")


class BozkurtVisemeLipReadingDataset(Dataset):
    """
    Dataset loader for Bozkurt viseme lip reading videos.
    Each sample is a sequence of grayscale lip-region frames.
    """

    def __init__(
        self,
        data_root_directory: str,
        viseme_class_labels: List[str],
        sequence_length_frames: int = 8,
        target_height_pixels: int = 64,
        target_width_pixels: int = 64,
        data_split: str = "train",
    ):
        self.data_root_directory = data_root_directory
        self.viseme_class_labels = sorted(viseme_class_labels)
        self.sequence_length_frames = sequence_length_frames
        self.target_height_pixels = target_height_pixels
        self.target_width_pixels = target_width_pixels
        self.data_split = data_split
        self.viseme_label_to_index = {
            label: idx for idx, label in enumerate(self.viseme_class_labels)
        }
        self.video_samples_list = self._load_all_samples()
        print(
            f"Loaded {len(self.video_samples_list)} samples "
            f"({data_split})"
        )

    def _load_all_samples(self) -> List[Dict]:
        """Index all .npy clip files in the dataset directory."""
        all_samples: List[Dict] = []
        for viseme_label in self.viseme_class_labels:
            folder = os.path.join(
                self.data_root_directory, viseme_label
            )
            if not os.path.exists(folder):
                print(f"Warning: folder missing for '{viseme_label}'")
                continue
            for filename in sorted(os.listdir(folder)):
                if filename.endswith(".npy"):
                    all_samples.append({
                        "video_file_path": os.path.join(
                            folder, filename
                        ),
                        "viseme_label": viseme_label,
                        "viseme_class_index": (
                            self.viseme_label_to_index[viseme_label]
                        ),
                    })
        return all_samples

    def __len__(self) -> int:
        return len(self.video_samples_list)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        sample_info = self.video_samples_list[index]
        try:
            frames = np.load(
                sample_info["video_file_path"], allow_pickle=False
            )
            if frames.size == 0:
                raise ValueError("empty npy")
        except (ValueError, OSError, FileNotFoundError):
            return self._get_fallback_sample(sample_info)
        return (
            torch.FloatTensor(self._process_video_frames(frames)),
            sample_info["viseme_class_index"],
        )

    def _get_fallback_sample(
        self, sample_info: Dict
    ) -> Tuple[torch.Tensor, int]:
        """Return zeros for corrupted/missing files."""
        return (
            torch.zeros(
                self.sequence_length_frames,
                1,
                self.target_height_pixels,
                self.target_width_pixels,
            ),
            sample_info["viseme_class_index"],
        )

    def _process_video_frames(
        self,
        video_frames_array: np.ndarray,
    ) -> np.ndarray:
        """Resize, grayscale, normalize, pad/sample to fixed length."""
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
            # Cyclic loop padding to repeat clip from the beginning
            # until reach required sequence length.
            loop_indices = [
                i % current_num_frames
                for i in range(self.sequence_length_frames)
            ]
            selected_frames = video_frames_array[loop_indices]

        is_training = self.data_split == "train"
        do_hflip = is_training and np.random.random() < 0.5
        do_brightness = is_training and np.random.random() < 0.4
        brightness_factor = (
            np.random.uniform(0.80, 1.20) if do_brightness else 1.0
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

            if do_hflip:
                grayscale_frame = np.fliplr(grayscale_frame)

            frame_normalized = grayscale_frame.astype(np.float32) / 255.0

            # Brightness augmentation
            if do_brightness:
                frame_normalized = np.clip(
                    frame_normalized * brightness_factor, 0.0, 1.0
                )

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
    """
    Compute per-class weights from a train subset.
    Use sqrt(inverse-frequency) weights capped at 4x max/min ratio.
    """
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
    inv_freq = total_count / (number_of_classes * safe_counts)
    sqrt_weights = np.sqrt(inv_freq)
    min_w = sqrt_weights.min()
    sqrt_weights = np.clip(sqrt_weights, min_w, min_w * 4.0)
    normalized_weights = sqrt_weights / sqrt_weights.mean()
    return torch.tensor(normalized_weights, dtype=torch.float32)


def build_weighted_sampler(
    train_dataset,
    number_of_classes: int,
) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler so every batch has roughly
    equal class representation regardless of raw class counts.
    """
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
    class_weights = 1.0 / safe_counts
    sample_weights = np.array(
        [
            class_weights[
                base_dataset.video_samples_list[i]["viseme_class_index"]
            ]
            for i in subset_indices
        ],
        dtype=np.float64,
    )

    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(subset_indices),
        replacement=True,
    )


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
    )

    if len(full_dataset) == 0:
        raise RuntimeError(
            f"No .npy samples found in {config.data_root_directory}"
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

    # WeightedRandomSampler forces each batch to be class-balanced
    train_sampler = build_weighted_sampler(
        train_dataset, config.number_of_classes
    )
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
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

    class_weights_tensor = compute_class_weights(
        train_dataset, config.number_of_classes
    ).to(config.device)
    loss_function = nn.CrossEntropyLoss(weight=class_weights_tensor)
    print(
        "Loss: CrossEntropyLoss with sqrt+cap4x class weights"
    )
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
            "Checkpoint already reached target epochs; "
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
