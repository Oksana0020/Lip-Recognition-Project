"""
Word-level Lip Reading Training Script using 3D CNN.
Trains 3D CNN to classify GRID words from lip video clips
saved as .npy arrays.
"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class TrainingConfig:
    data_root_directory: Path = (
        PROJECT_ROOT / "data" / "processed" / "words_by_label")
    target_frame_count: int = 8
    target_frame_height: int = 64
    target_frame_width: int = 64
    input_channels: int = 1
    batch_size: int = 64
    number_of_epochs: int = 20
    learning_rate: float = 0.001
    training_split_ratio: float = 0.70
    validation_split_ratio: float = 0.15
    test_split_ratio: float = 0.15
    random_seed: int = 42
    device: torch.device = torch.device("cpu")


def resolve_training_device() -> torch.device:
    """Use CUDA if available; otherwise use CPU"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        return device
    print("Using CPU for training")
    return torch.device("cpu")


class WordLipReadingDataset(Dataset):
    """
    Each item returns:
      frames_tensor: [C, T, H, W]
      label_index: int
    """

    def __init__(
        self,
        samples: List[Dict],
        word_to_index_mapping: Dict[str, int],
        target_frame_count: int,
        target_frame_height: int,
        target_frame_width: int,
    ) -> None:
        self.samples = samples
        self.word_to_index_mapping = word_to_index_mapping
        self.target_frame_count = target_frame_count
        self.target_frame_height = target_frame_height
        self.target_frame_width = target_frame_width

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, sample_index: int) -> Tuple[torch.Tensor, int]:
        sample_info = self.samples[sample_index]
        frames_file_path = Path(sample_info["frames_path"])
        word_label = sample_info["word"]
        raw_frames = np.load(frames_file_path, allow_pickle=False)
        processed_frames = self._process_video_frames(raw_frames)
        frames_tensor = torch.from_numpy(processed_frames).float()
        frames_tensor = frames_tensor.permute(3, 0, 1, 2)
        word_class_index = self.word_to_index_mapping[word_label]
        return frames_tensor, word_class_index

    def _process_video_frames(self, video_frames: np.ndarray) -> np.ndarray:
        """Resample, center crop lower face region, resize, grayscale."""
        current_frame_count = int(video_frames.shape[0])
        if current_frame_count > self.target_frame_count:
            frame_indices = np.linspace(
                0,
                current_frame_count - 1,
                self.target_frame_count,
                dtype=int)
            video_frames = video_frames[frame_indices]
        elif current_frame_count < self.target_frame_count:
            padded_frames = list(video_frames)
            while len(padded_frames) < self.target_frame_count:
                padded_frames.append(video_frames[-1])
            video_frames = np.stack(padded_frames, axis=0)
        resized_frames: List[np.ndarray] = []

        for frame_index in range(self.target_frame_count):
            frame = video_frames[frame_index]
            frame_height, frame_width = frame.shape[:2]
            crop_y_top = max(0, int(frame_height * 0.66))
            crop_y_bottom = min(frame_height, int(frame_height * 0.92))
            crop_x_left = max(0, int(frame_width * 0.33))
            crop_x_right = min(frame_width, int(frame_width * 0.67))
            frame = frame[crop_y_top:crop_y_bottom, crop_x_left:crop_x_right]
            resized_frame = cv2.resize(
                frame,
                (self.target_frame_width, self.target_frame_height))
            if resized_frame.ndim == 3:
                grayscale_frame = cv2.cvtColor(
                    resized_frame,
                    cv2.COLOR_BGR2GRAY,
                )
            else:
                grayscale_frame = resized_frame
            resized_frames.append(grayscale_frame[..., np.newaxis])
        processed_frames = np.asarray(resized_frames, dtype=np.float32) / 255.0
        return processed_frames


class ThreeDimensionalCNN(nn.Module):
    """3D CNN for word-level classification"""

    def __init__(self, input_channels: int, number_of_classes: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv3d(
                in_channels=input_channels,
                out_channels=32,
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))
        self.conv_block_2 = nn.Sequential(
            nn.Conv3d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.conv_block_3 = nn.Sequential(
            nn.Conv3d(
                in_channels=64,
                out_channels=128,
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.conv_block_4 = nn.Sequential(
            nn.Conv3d(
                in_channels=128,
                out_channels=256,
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 1 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(512, number_of_classes))

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        features = self.conv_block_1(input_tensor)
        features = self.conv_block_2(features)
        features = self.conv_block_3(features)
        features = self.conv_block_4(features)
        features = self.adaptive_pool(features)
        return self.classifier(features)


def load_word_dataset(
    data_root: Path,
) -> Tuple[List[Dict], Dict[str, int], Dict[int, str]]:
    """Load all word clips from label folders."""
    print("Loading word dataset")
    samples: List[Dict] = []
    word_labels = set()
    for word_folder in sorted(data_root.iterdir()):
        if not word_folder.is_dir():
            continue
        word_label = word_folder.name
        non_word_entries = {
            "extraction_summary.json",
            "word_statistics.json",
            "sp"}
        if word_label in non_word_entries:
            continue
        word_labels.add(word_label)
        for numpy_file in word_folder.glob("*.npy"):
            samples.append(
                {
                    "frames_path": str(numpy_file),
                    "word": word_label,
                })

    sorted_word_labels = sorted(word_labels)
    word_to_index = {
        word: index for index, word in enumerate(sorted_word_labels)}
    index_to_word = {index: word for word, index in word_to_index.items()}
    print(
        f"Loaded {len(samples)} samples from "
        f"{len(sorted_word_labels)} unique words")
    return samples, word_to_index, index_to_word


def build_data_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Build DataLoader for dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0)


def run_one_epoch_train(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Run one training epoch."""
    model.train()
    total_loss_sum = 0.0
    total_correct_predictions = 0
    total_samples_count = 0
    progress_bar = tqdm(data_loader, desc="Train", leave=False)
    for batch_frames, batch_labels in progress_bar:
        batch_frames = batch_frames.to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()
        predictions = model(batch_frames)
        loss = loss_function(predictions, batch_labels)
        loss.backward()
        optimizer.step()
        batch_size = int(batch_frames.size(0))
        total_loss_sum += float(loss.item()) * batch_size
        predicted_classes = torch.argmax(predictions, dim=1)
        total_correct_predictions += int(
            (predicted_classes == batch_labels).sum().item())
        total_samples_count += batch_size
    average_loss = total_loss_sum / max(1, total_samples_count)
    accuracy = 100.0 * total_correct_predictions / max(1, total_samples_count)
    return average_loss, accuracy


def run_one_epoch_eval(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Run one evaluation epoch."""
    model.eval()
    total_loss_sum = 0.0
    total_correct_predictions = 0
    total_samples_count = 0
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Eval", leave=False)
        for batch_frames, batch_labels in progress_bar:
            batch_frames = batch_frames.to(device)
            batch_labels = batch_labels.to(device)
            predictions = model(batch_frames)
            loss = loss_function(predictions, batch_labels)
            batch_size = int(batch_frames.size(0))
            total_loss_sum += float(loss.item()) * batch_size
            predicted_classes = torch.argmax(predictions, dim=1)
            total_correct_predictions += int(
                (predicted_classes == batch_labels).sum().item())
            total_samples_count += batch_size
    average_loss = total_loss_sum / max(1, total_samples_count)
    accuracy = 100.0 * total_correct_predictions / max(1, total_samples_count)
    return average_loss, accuracy


def main() -> None:
    """Train word-level 3D CNN."""
    print("=" * 80)
    print("WORD-LEVEL LIP READING 3D CNN TRAINING")
    print("=" * 80)
    config = TrainingConfig()
    config.device = resolve_training_device()
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)
    all_samples, word_to_index, _ = load_word_dataset(config.data_root_directory)
    number_of_classes = len(word_to_index)
    all_samples_shuffled = list(all_samples)
    random.shuffle(all_samples_shuffled)
    total_sample_count = len(all_samples_shuffled)
    train_sample_count = int(total_sample_count * config.training_split_ratio)
    validation_sample_count = int(
        total_sample_count * config.validation_split_ratio)
    train_samples = all_samples_shuffled[:train_sample_count]
    validation_samples = all_samples_shuffled[
        train_sample_count:train_sample_count + validation_sample_count]
    test_samples = all_samples_shuffled[
        train_sample_count + validation_sample_count]

    print(
        f"Split: Train={len(train_samples)}, "
        f"Val={len(validation_samples)}, "
        f"Test={len(test_samples)}")

    train_dataset = WordLipReadingDataset(
        samples=train_samples,
        word_to_index_mapping=word_to_index,
        target_frame_count=config.target_frame_count,
        target_frame_height=config.target_frame_height,
        target_frame_width=config.target_frame_width)
    
    validation_dataset = WordLipReadingDataset(
        samples=validation_samples,
        word_to_index_mapping=word_to_index,
        target_frame_count=config.target_frame_count,
        target_frame_height=config.target_frame_height,
        target_frame_width=config.target_frame_width)
    
    test_dataset = WordLipReadingDataset(
        samples=test_samples,
        word_to_index_mapping=word_to_index,
        target_frame_count=config.target_frame_count,
        target_frame_height=config.target_frame_height,
        target_frame_width=config.target_frame_width)
    
    train_loader = build_data_loader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True)
    
    validation_loader = build_data_loader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False)
    
    test_loader = build_data_loader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False)
    
    model = ThreeDimensionalCNN(
        input_channels=config.input_channels,
        number_of_classes=number_of_classes,
    ).to(config.device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    for epoch in range(1, config.number_of_epochs + 1):
        print(f"\nEpoch {epoch}/{config.number_of_epochs}")
        print("-" * 80)
        train_loss, train_accuracy = run_one_epoch_train(
            model=model,
            data_loader=train_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            device=config.device)
        validation_loss, validation_accuracy = run_one_epoch_eval(
            model=model,
            data_loader=validation_loader,
            loss_function=loss_function,
            device=config.device)
        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2f}%")
        print(
            f"Val   Loss: {validation_loss:.4f} | "
            f"Val   Acc: {validation_accuracy:.2f}%")
    print("\n" + "=" * 80)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 80)
    test_loss, test_accuracy = run_one_epoch_eval(
        model=model,
        data_loader=test_loader,
        loss_function=loss_function,
        device=config.device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
