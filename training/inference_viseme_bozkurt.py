"""
Bozkurt Viseme Recognition Inference Script
Loads a trained 3D CNN model
and runs inference on a single lip-region video saved as .npy
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn


BOZKURT_VISEME_CLASSES: List[str] = [
    "S",
    "V2", "V3", "V4", "V5", "V6", "V7", "V8",
    "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16",
]


class ThreeDimensionalCNN(nn.Module):
    def __init__(
        self,
        number_of_classes: int,
        input_channels: int = 1,
        dropout_probability: float = 0.5,
    ):
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv3d(input_channels, 32, (3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2)),
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv3d(32, 64, (3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv3d(64, 128, (3, 3, 3), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),
        )

        self.conv_block_4 = nn.Sequential(
            nn.Conv3d(128, 256, (3, 3, 3), padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),
        )

        self.pool = nn.AdaptiveAvgPool3d((1, 4, 4))

        self.fc = nn.Sequential(
            nn.Linear(256 * 1 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_probability),
            nn.Linear(512, number_of_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3, 4)

        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)

        x = self.pool(x)
        x = x.view(x.size(0), -1)

        return self.fc(x)


def choose_device(device_choice: str) -> torch.device:
    if device_choice == "cuda":
        return torch.device("cuda")
    if device_choice == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint_state_dict(checkpoint_path: Path, device: torch.device) -> dict:
    checkpoint = torch.load(str(checkpoint_path), map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]

    if isinstance(checkpoint, dict):
        return checkpoint

    raise ValueError("Unsupported checkpoint format")


def load_trained_model(
    checkpoint_path: Path,
    number_of_classes: int,
    device: torch.device,
) -> nn.Module:

    model = ThreeDimensionalCNN(
        number_of_classes=number_of_classes,
        input_channels=1,
        dropout_probability=0.5,
    ).to(device)

    state_dict = load_checkpoint_state_dict(checkpoint_path, device)
    model.load_state_dict(state_dict)
    model.eval()

    return model


def preprocess_video_frames(
    video_path: Path,
    sequence_length: int = 50,
    height: int = 64,
    width: int = 64,
) -> torch.Tensor:

    frames = np.load(str(video_path))

    if frames.size == 0:
        raise ValueError(f"Empty video: {video_path}")

    num_frames = frames.shape[0]

    if num_frames >= sequence_length:
        idx = np.linspace(0, num_frames - 1, sequence_length, dtype=int)
        frames = frames[idx]
    else:
        pad = sequence_length - num_frames
        frames = np.concatenate([frames, frames[-1:].repeat(pad, axis=0)])

    processed = []

    for frame in frames:

        frame = cv2.resize(frame, (width, height))

        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        frame = frame.astype(np.float32) / 255.0
        frame = np.expand_dims(frame, axis=0)

        processed.append(frame)

    video = np.stack(processed, axis=0)
    video = torch.FloatTensor(video).unsqueeze(0)

    return video


def predict_viseme(
    model: nn.Module,
    video_tensor: torch.Tensor,
    device: torch.device,
    labels: List[str],
) -> Tuple[str, float, np.ndarray]:

    video_tensor = video_tensor.to(device)

    with torch.no_grad():
        logits = model(video_tensor)
        probs = torch.softmax(logits, dim=1)

    confidence, index = torch.max(probs, dim=1)

    predicted = labels[index.item()]
    confidence = confidence.item()
    probabilities = probs.cpu().numpy()[0]

    return predicted, confidence, probabilities


def print_top_k(
    predicted: str,
    confidence: float,
    probabilities: np.ndarray,
    labels: List[str],
    top_k: int,
) -> None:

    print("\nBozkurt Viseme Inference")
    print(f"Prediction: {predicted}")
    print(f"Confidence: {confidence * 100:.2f}%")

    top = np.argsort(probabilities)[::-1][:top_k]

    print(f"\nTop {top_k} predictions:")

    for i, idx in enumerate(top, 1):
        label = labels[int(idx)]
        prob = probabilities[int(idx)]
        print(f"{i}. {label}  {prob * 100:.2f}%")


def main() -> None:

    parser = argparse.ArgumentParser(description="Bozkurt viseme inference")

    parser.add_argument("--video", required=True)
    parser.add_argument("--checkpoint", required=True)

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )

    parser.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()

    device = choose_device(args.device)

    video_path = Path(args.video)
    checkpoint_path = Path(args.checkpoint)

    if not video_path.exists():
        raise FileNotFoundError(video_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    model = load_trained_model(
        checkpoint_path,
        number_of_classes=len(BOZKURT_VISEME_CLASSES),
        device=device,
    )

    video_tensor = preprocess_video_frames(video_path)

    predicted, confidence, probs = predict_viseme(
        model,
        video_tensor,
        device,
        BOZKURT_VISEME_CLASSES,
    )

    print_top_k(predicted, confidence, probs, BOZKURT_VISEME_CLASSES, args.top_k)


if __name__ == "__main__":
    main()
