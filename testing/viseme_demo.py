"""
Viseme-Level Inference Demo
Load a sample lip clip, run it through the trained
Bozkurt viseme 3D CNN model and print the top-3
predicted viseme classes with confidence bars.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import cv2
from training.model import ThreeDimensionalCNN
from training.train_utils import _remap_state_dict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
SEQUENCE_LENGTH = 8
VIDEO_HEIGHT = 64
VIDEO_WIDTH = 64
VISEME_DESCRIPTIONS = {
    "V2":  "Open vowels     (AY, AH)",
    "V3":  "Mid vowels      (EY, EH, AE)",
    "V6":  "Rounded lips    (UW, UH, W)",
    "V7":  "Back rounded    (AO, AA, OY, OW)",
    "V8":  "Diphthong       (AW)",
    "V9":  "Velar           (G, HH, K, NG)",
    "V10": "R-coloured      (R)",
    "V11": "Alveolar        (L, D, N, T)",
    "V12": "Alveolar fric.  (S, Z)",
    "V13": "Post-alveolar   (CH, SH, JH, ZH)",
    "V14": "Dental fric.    (TH, DH)",
    "V15": "Labiodental     (F, V)",
    "V16": "Bilabial        (M, B, P)"}


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[ThreeDimensionalCNN, List[str]]:
    """Load the trained viseme model and class labels from checkpoint."""
    checkpoint = torch.load(
        str(checkpoint_path),
        map_location=device,
        weights_only=False)
    viseme_classes: List[str] = checkpoint["viseme_classes"]
    model = ThreeDimensionalCNN(
        num_classes=len(viseme_classes),
        input_channels=1)
    model.load_state_dict(_remap_state_dict(checkpoint["model_state_dict"]))
    model.to(device)
    model.eval()
    return model, viseme_classes


def preprocess_clip(
    npy_path: Path,
    sequence_length: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Load a .npy lip clip and prepare it for model input."""
    frames = np.load(str(npy_path), allow_pickle=False)
    if frames.size == 0:
        raise ValueError(f"Empty clip: {npy_path}")
    num_frames = frames.shape[0]
    if num_frames >= sequence_length:
        indices = np.linspace(
            0, num_frames - 1, sequence_length, dtype=int)
        frames = frames[indices]
    else:
        pad_count = sequence_length - num_frames
        frames = np.concatenate(
            [frames, np.repeat(frames[-1:], pad_count, axis=0)])
    processed = []
    for frame in frames:
        frame = cv2.resize(frame, (width, height))
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = frame.astype(np.float32) / 255.0
        processed.append(frame[np.newaxis, :, :])
    arr = np.stack(processed, axis=0)  # [T, 1, H, W]
    tensor = torch.FloatTensor(arr).permute(1, 0, 2, 3).unsqueeze(0)
    return tensor


def run_inference(
    model: ThreeDimensionalCNN,
    tensor: torch.Tensor,
    viseme_classes: List[str],
    device: torch.device,
    top_k: int = 3,
) -> List[Tuple[str, float]]:
    """Return top-k predictions for the input clip tensor"""
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    top_indices = probs.argsort()[::-1][:top_k]
    return [
        (viseme_classes[i], float(probs[i]) * 100.0)
        for i in top_indices]


def confidence_bar(percent: float, width: int = 30) -> str:
    filled = int(round(percent / 100.0 * width))
    return "█" * filled + "░" * (width - filled)


def print_results(
    predictions: List[Tuple[str, float]],
    ground_truth: str,
) -> None:
    print()
    print("  Ground truth :  " + ground_truth)
    desc = VISEME_DESCRIPTIONS.get(ground_truth, "")
    if desc:
        print("  Description  :  " + desc)
    print()
    print("  Top predictions:")
    print("  " + "─" * 52)
    for rank, (label, confidence) in enumerate(predictions, 1):
        bar = confidence_bar(confidence)
        marker = " ✓" if label == ground_truth else "  "
        print(
            f"  {rank}. {label:<4}  {bar}  {confidence:5.1f}%{marker}")
    print("  " + "─" * 52)
    top_label = predictions[0][0]
    if top_label == ground_truth:
        print("\n  ✅  Correct prediction!\n")
    else:
        print(
            f"\n  ❌  Incorrect predicted {top_label}, "
            f"expected {ground_truth}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Viseme-level inference demo")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "training/checkpoints_bozkurt_viseme/bozkurt_viseme_best_model.pth")
    parser.add_argument(
        "--clip",
        type=Path,
        required=True,
        help="Path to a .npy viseme clip")
    parser.add_argument(
        "--ground-truth",
        type=str,
        required=True,
        help="Expected viseme label, e.g. V8")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 60)
    print("   Viseme-Level Inference Demo  —  Bozkurt 3D CNN")
    print("=" * 60)
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Clip       : {args.clip}")
    print(f"  Device     : {device}")
    print("\n  Loading model...", end=" ", flush=True)
    model, viseme_classes = load_model(args.checkpoint, device)
    print(f"done  ({len(viseme_classes)} classes, "
          f"val acc 82.78%)")
    print(f"\n  Running inference on: {args.clip.name}")
    try:
        tensor = preprocess_clip(
            args.clip,
            sequence_length=SEQUENCE_LENGTH,
            height=VIDEO_HEIGHT,
            width=VIDEO_WIDTH)
        predictions = run_inference(
            model, tensor, viseme_classes, device)
        print_results(predictions, args.ground_truth)
    except Exception as exc:
        print(f"\n  Error processing clip: {exc}\n")


if __name__ == "__main__":
    main()
