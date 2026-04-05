"""
Word-level Lip Reading Training Script using 3D CNN.
Trains 3D CNN to classify GRID words from lip video clips
saved as .npy arrays.
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class TrainingConfig:
    data_root_directory: Path = (
        PROJECT_ROOT / "data" / "processed" / "words_by_label")
    checkpoint_save_directory: Path = (
        PROJECT_ROOT / "training" / "checkpoints_words")
    tensorboard_log_directory: Path = PROJECT_ROOT / "training" / "runs_words"
    target_frame_count: int = 8
    target_frame_height: int = 64
    target_frame_width: int = 64
    input_channels: int = 1
    batch_size: int = 64
    number_of_epochs: int = 60
    learning_rate: float = 0.001
    weight_decay: float = 0.00001
    num_workers: int = 0
    training_split_ratio: float = 0.70
    validation_split_ratio: float = 0.15
    test_split_ratio: float = 0.15
    random_seed: int = 42
    patience_epochs: int = 10
    save_checkpoint_every_n_epochs: int = 5
    device: torch.device = torch.device("cpu")


def resolve_training_device() -> torch.device:
    """Use CUDA when genuinely available; otherwise stay on CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        return device
    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            print(
                "Intel XPU detected, but this 3D CNN pipeline is configured "
                "for CPU unless real CUDA is available.")
    except Exception:
        pass
    print("Using CPU for training (real CUDA backend not available)")
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
        is_training: bool = False
    ):
        self.samples = samples
        self.word_to_index_mapping = word_to_index_mapping
        self.target_frame_count = target_frame_count
        self.target_frame_height = target_frame_height
        self.target_frame_width = target_frame_width
        self.is_training = is_training
        bbox_json_path = (
            PROJECT_ROOT / "data" / "processed" / "words_lip_bboxes.json")
        if bbox_json_path.exists():
            with open(bbox_json_path, encoding="utf-8") as bbox_file:
                raw_bbox: Dict[str, list] = json.load(bbox_file)
            self.lip_bbox_lookup: Dict[str, list] = {
                key.lower(): value for key, value in raw_bbox.items()}
            print(
                f"Loaded {len(self.lip_bbox_lookup):,} precomputed lip bboxes")
        else:
            self.lip_bbox_lookup = {}
            print("WARNING: words_lip_bboxes.json not found; using fixed crop")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, sample_index: int) -> Tuple[torch.Tensor, int]:
        sample_info = self.samples[sample_index]
        frames_file_path = Path(sample_info["frames_path"])
        word_label = sample_info["word"]
        raw_frames = np.load(frames_file_path, allow_pickle=False)
        processed_frames = self._process_video_frames(
            raw_frames, frames_file_path)

        if self.is_training:
            if random.random() < 0.5:
                processed_frames = processed_frames[:, :, ::-1, :].copy()

            if random.random() < 0.4:
                factor = random.uniform(0.8, 1.2)
                processed_frames = np.clip(
                    processed_frames * factor,
                    0.0,
                    1.0)

        frames_tensor = torch.from_numpy(processed_frames).float()
        frames_tensor = frames_tensor.permute(3, 0, 1, 2)
        word_class_index = self.word_to_index_mapping[word_label]
        return frames_tensor, word_class_index

    def _process_video_frames(
        self,
        video_frames: np.ndarray,
        frames_file_path: Path
    ) -> np.ndarray:
        """Resample, crop lip region, resize, and grayscale frames."""
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
            source_frames = list(video_frames)
            while len(padded_frames) < self.target_frame_count:
                padded_frames.extend(
                    source_frames[
                        : self.target_frame_count - len(padded_frames)])
            video_frames = np.stack(padded_frames, axis=0)

        lookup_key = str(frames_file_path.resolve()).lower()
        clip_bbox = self.lip_bbox_lookup.get(lookup_key)

        if clip_bbox is not None:
            dlib_x0, dlib_y0, dlib_x1, dlib_y1 = clip_bbox
        else:
            dlib_x0, dlib_y0, dlib_x1, dlib_y1 = None, None, None, None

        resized_frames: List[np.ndarray] = []

        for temporal_frame_index in range(self.target_frame_count):
            frame = video_frames[temporal_frame_index]
            frame_height, frame_width = frame.shape[:2]

            if dlib_x0 is not None:
                crop_x_left = max(0, dlib_x0)
                crop_y_top = max(0, dlib_y0)
                crop_x_right = min(frame_width, dlib_x1)
                crop_y_bottom = min(frame_height, dlib_y1)
            else:
                crop_y_top = max(0, int(frame_height * 0.66))
                crop_y_bottom = min(
                    frame_height, int(frame_height * 0.92))
                crop_x_left = max(0, int(frame_width * 0.33))
                crop_x_right = min(
                    frame_width, int(frame_width * 0.67))

            frame = frame[
                crop_y_top:crop_y_bottom, crop_x_left:crop_x_right]
            resized_frame = cv2.resize(
                frame,
                (self.target_frame_width, self.target_frame_height))

            if resized_frame.ndim == 3:
                grayscale_resized_frame = cv2.cvtColor(
                    resized_frame,
                    cv2.COLOR_BGR2GRAY)
            else:
                grayscale_resized_frame = resized_frame

            resized_frames.append(grayscale_resized_frame[..., np.newaxis])

        processed_frames = np.asarray(resized_frames, dtype=np.float32) / 255.0
        return processed_frames


class ThreeDimensionalCNN(nn.Module):
    """
    3D CNN for word-level classification.
    Architecture mirrors the viseme model (32→64→128→256 channels,
    only output layer changed to 52 classes instead of 13
    """

    def __init__(self, input_channels: int, number_of_classes: int):
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
    data_root: Path
) -> Tuple[List[Dict], Dict[str, int], Dict[int, str]]:
    print("Loading word dataset")
    samples: List[Dict] = []
    word_labels: set[str] = set()
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


def build_weighted_sampler(
    samples: List[Dict],
    word_to_index: Dict[str, int]
) -> WeightedRandomSampler:
    """
    build WeightedRandomSampler so every batch has
    equal class representations
    """
    num_classes = len(word_to_index)
    class_counts = np.zeros(num_classes, dtype=np.float64)

    for sample_info in samples:
        class_counts[word_to_index[sample_info["word"]]] += 1.0

    safe_counts = np.maximum(class_counts, 1.0)
    class_weights = 1.0 / safe_counts
    sample_weights = np.array(
        [class_weights[word_to_index[sample["word"]]] for sample in samples],
        dtype=np.float64)

    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(samples),
        replacement=True)


def build_data_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    shuffle: bool = False,
    sampler: WeightedRandomSampler = None
) -> DataLoader:
    loader_kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda"}

    if sampler is not None:
        loader_kwargs["sampler"] = sampler
    else:
        loader_kwargs["shuffle"] = shuffle

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    return DataLoader(dataset, **loader_kwargs)


def run_one_epoch_train(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        batch_size = int(batch_frames.size(0))
        total_loss_sum += float(loss.item()) * batch_size
        predicted_classes = torch.argmax(predictions, dim=1)
        total_correct_predictions += int(
            (predicted_classes == batch_labels).sum().item())
        total_samples_count += batch_size
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    average_loss = total_loss_sum / max(1, total_samples_count)
    accuracy = 100.0 * total_correct_predictions / max(1, total_samples_count)
    return average_loss, accuracy


def run_one_epoch_eval(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
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


def save_model_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    train_loss: float,
    validation_loss: float,
    validation_accuracy: float,
    save_path: Path,
    word_to_index: Dict[str, int],
    index_to_word: Dict[int, str],
    config: TrainingConfig
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "validation_accuracy": validation_accuracy,
        "word_to_index": word_to_index,
        "index_to_word": index_to_word,
        "config": {
            "target_frame_count": config.target_frame_count,
            "target_frame_height": config.target_frame_height,
            "target_frame_width": config.target_frame_width,
            "input_channels": config.input_channels,
            "batch_size": config.batch_size}}
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: {save_path}")


def main() -> None:
    print("=" * 80)
    print("WORD-LEVEL LIP READING 3D CNN TRAINING")
    print("=" * 80)
    config = TrainingConfig()
    config.device = resolve_training_device()
    config.checkpoint_save_directory.mkdir(parents=True, exist_ok=True)
    config.tensorboard_log_directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)
    all_samples, word_to_index, index_to_word = load_word_dataset(
        config.data_root_directory)
    number_of_classes = len(word_to_index)
    print(f"Number of classes inferred from folders: {number_of_classes}")
    dataset_split_rng = random.Random(config.random_seed)
    all_samples_shuffled = list(all_samples)
    dataset_split_rng.shuffle(all_samples_shuffled)
    total_sample_count = len(all_samples_shuffled)
    train_sample_count = int(total_sample_count * config.training_split_ratio)
    validation_sample_count = int(
        total_sample_count * config.validation_split_ratio)
    train_samples = all_samples_shuffled[:train_sample_count]
    validation_samples = all_samples_shuffled[
        train_sample_count:train_sample_count + validation_sample_count]
    test_samples = all_samples_shuffled[
        train_sample_count + validation_sample_count:]

    print(
        f"Split: Train={len(train_samples)}, "
        f"Val={len(validation_samples)}, "
        f"Test={len(test_samples)}")
    print(
        "Class imbalance handled by WeightedRandomSampler only "
        "(no loss weights)")

    train_dataset = WordLipReadingDataset(
        samples=train_samples,
        word_to_index_mapping=word_to_index,
        target_frame_count=config.target_frame_count,
        target_frame_height=config.target_frame_height,
        target_frame_width=config.target_frame_width,
        is_training=True)

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

    train_sampler = build_weighted_sampler(train_samples, word_to_index)
    print("WeightedRandomSampler built for training loader")

    train_loader = build_data_loader(
        train_dataset,
        sampler=train_sampler,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        device=config.device)

    validation_loader = build_data_loader(
        validation_dataset,
        shuffle=False,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        device=config.device)

    test_loader = build_data_loader(
        test_dataset,
        shuffle=False,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        device=config.device)

    print(f"\nUsing device: {config.device}")
    print(
        f"Frame config: T={config.target_frame_count}, "
        f"H={config.target_frame_height}, W={config.target_frame_width}")
    print(
        f"DataLoader config: batch_size={config.batch_size}, "
        f"workers={config.num_workers}")

    model = ThreeDimensionalCNN(
        input_channels=config.input_channels,
        number_of_classes=number_of_classes).to(config.device)

    total_parameters = sum(
        parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad)
    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")

    loss_function = nn.CrossEntropyLoss()
    print("Loss: CrossEntropyLoss (no smoothing) + WeightedRandomSampler")

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay)

    learning_rate_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
        verbose=True)

    tensorboard_writer = SummaryWriter(str(config.tensorboard_log_directory))
    best_validation_accuracy = 0.0
    epochs_without_improvement = 0

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

        learning_rate_scheduler.step(validation_loss)
        tensorboard_writer.add_scalar("Loss/train", train_loss, epoch)
        tensorboard_writer.add_scalar(
            "Loss/validation",
            validation_loss,
            epoch)
        tensorboard_writer.add_scalar("Accuracy/train", train_accuracy, epoch)
        tensorboard_writer.add_scalar(
            "Accuracy/validation",
            validation_accuracy,
            epoch)
        tensorboard_writer.add_scalar(
            "Learning_Rate",
            optimizer.param_groups[0]["lr"],
            epoch)

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2f}%")
        print(
            f"Val   Loss: {validation_loss:.4f} | "
            f"Val   Acc: {validation_accuracy:.2f}%")

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            epochs_without_improvement = 0
            best_model_path = (
                config.checkpoint_save_directory / "best_model_words.pth")
            save_model_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                validation_loss=validation_loss,
                validation_accuracy=validation_accuracy,
                save_path=best_model_path,
                word_to_index=word_to_index,
                index_to_word=index_to_word,
                config=config)
            print(f"New best model! Val Acc: {validation_accuracy:.2f}%")
        else:
            epochs_without_improvement += 1

        if epoch % config.save_checkpoint_every_n_epochs == 0:
            checkpoint_path = (
                config.checkpoint_save_directory
                / f"checkpoint_epoch_{epoch}.pth")
            save_model_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                validation_loss=validation_loss,
                validation_accuracy=validation_accuracy,
                save_path=checkpoint_path,
                word_to_index=word_to_index,
                index_to_word=index_to_word,
                config=config)

        if epochs_without_improvement >= config.patience_epochs:
            print("\nEarly stopping triggered.")
            print(f"No improvement for {config.patience_epochs} epochs.")
            break

    print("\n" + "=" * 80)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 80)

    best_model_checkpoint_path = (
        config.checkpoint_save_directory / "best_model_words.pth")
    if best_model_checkpoint_path.exists():
        best_checkpoint = torch.load(
            best_model_checkpoint_path,
            map_location=config.device)
        model.load_state_dict(best_checkpoint["model_state_dict"])
    else:
        print(
            "WARNING: best_model_words.pth not found; "
            "evaluating current model")

    test_loss, test_accuracy = run_one_epoch_eval(
        model=model,
        data_loader=test_loader,
        loss_function=loss_function,
        device=config.device)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    tensorboard_writer.close()
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
