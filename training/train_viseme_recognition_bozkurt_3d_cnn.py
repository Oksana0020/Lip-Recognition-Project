"""
Bozkurt Viseme Recognition 3D CNN Training Script

This script trains a 3D Convolutional Neural Network to recognize
16 viseme classes using Bozkurt viseme mapping system.
 Visemes group phonemes with similar mouth shapes.
Bozkurt System: 16 viseme classes (S, V2-V16)
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
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
        self.data_root_directory = "data/processed/visemes_bozkurt"

        self.video_height_pixels = 64
        self.video_width_pixels = 64
        self.sequence_length_frames = 50
        self.color_channels = 1

        self.batch_size_samples = 32
        self.number_of_epochs = 50
        self.learning_rate = 0.001
        self.weight_decay_l2 = 0.0001

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
        else:
            device = torch.device("cpu")
            print("Using CPU for training")
        return device


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

                if not os.path.exists(npy_file_path):
                    continue

                sample_info = {
                    "video_file_path": npy_file_path,
                    "metadata_file_path": json_file_path,
                    "viseme_label": viseme_label,
                    "viseme_class_index": (
                        self.viseme_label_to_index[viseme_label]
                    ),
                }
                all_samples.append(sample_info)

        return all_samples

    def __len__(self) -> int:
        return len(self.video_samples_list)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        sample_info = self.video_samples_list[index]
        video_frames_array = np.load(sample_info["video_file_path"])
        processed_frames = self._process_video_frames(video_frames_array)

        video_tensor = torch.FloatTensor(processed_frames)
        label_index = sample_info["viseme_class_index"]
        return video_tensor, label_index

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
    device: torch.device,
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
    device: torch.device,
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
            correct_predictions = (predicted_classes == label_batch).sum().item()

            total_loss_sum += loss.item() * video_batch.size(0)
            correct_predictions_count += correct_predictions
            total_samples_count += video_batch.size(0)

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
) -> None:
    """Save model checkpoint to disk."""
    os.makedirs(save_directory, exist_ok=True)
    checkpoint_path = os.path.join(save_directory, checkpoint_filename)

    checkpoint_data = {
        "epoch": epoch_number,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_accuracy": val_accuracy,
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


def main_training_pipeline():
    """Run the complete training pipeline for Bozkurt viseme recognition."""
    print("=" * 80)
    print("BOZKURT VISEME RECOGNITION - 3D CNN TRAINING")
    print("=" * 80)

    config = BozkurtVisemeTrainingConfig()

    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)

    bozkurt_viseme_classes = [
        "S",
        "V2", "V3", "V4", "V5", "V6", "V7", "V8",
        "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16",
    ]

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

    train_dataset, val_dataset, test_dataset = split_dataset_into_train_val_test(
        full_dataset=full_dataset,
        train_ratio=config.training_data_ratio,
        val_ratio=config.validation_data_ratio,
        test_ratio=config.test_data_ratio,
        random_seed=config.random_seed,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size_samples,
        shuffle=True,
        num_workers=0,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size_samples,
        shuffle=False,
        num_workers=0,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size_samples,
        shuffle=False,
        num_workers=0,
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

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay_l2,
    )

    tensorboard_writer = SummaryWriter(
        log_dir=config.tensorboard_log_directory,
    )

    print("\n" + "-" * 80)
    print("Starting training...")
    print("-" * 80)

    best_validation_accuracy = 0.0
    training_history: List[Dict[str, float]] = []

    for epoch in range(1, config.number_of_epochs + 1):
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

        tensorboard_writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
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
            )

        if val_metrics["accuracy"] > best_validation_accuracy:
            best_validation_accuracy = val_metrics["accuracy"]
            save_model_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch_number=epoch,
                val_accuracy=val_metrics["accuracy"],
                save_directory=config.checkpoint_save_directory,
                checkpoint_filename="bozkurt_viseme_best_model.pth",
            )
            print(
                "New best model saved! "
                f"Validation accuracy: "
                f"{best_validation_accuracy:.2f}%"
            )

    print("\n" + "=" * 80)
    print("Evaluating on test set...")
    print("=" * 80)

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
    main_training_pipeline()
