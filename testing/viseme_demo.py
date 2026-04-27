
"""
Viseme-Level Inference Demo
Interactive menu that lets pick a sample lip clip,
runs it through the trained Bozkurt viseme 3D CNN model
and prints the top-3 predicted viseme classes with confidence bars
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import argparse
import os
import sys

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from training.model import ThreeDimensionalCNN  # noqa: E402
from training.train_utils import _remap_state_dict  # noqa: E402

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

# Slected sample clips one per class, chosen from the dataset
SAMPLE_CLIPS = [
    ("V2  — Open vowel (AY/AH)",
     "data/processed/visemes_bozkurt_mfa_balanced_npy/V2/bbab9s_34750_AY.npy"),
    ("V3  — Mid vowel (EY/EH/AE)",
     "data/processed/visemes_bozkurt_mfa_balanced_npy/V3/bbab8n_28000_EY.npy"),
    ("V6  — Rounded lips (UW/UH/W)",
     "data/processed/visemes_bozkurt_mfa_balanced_npy/V6/bbab8n_18500_UW.npy"),
    ("V7  — Back rounded (AO/AA)",
     "data/processed/visemes_bozkurt_mfa_balanced_npy/V7/bbae4n_37000_AO.npy"),
    ("V8  — Diphthong (AW)  highest accuracy 99%",
     "data/processed/visemes_bozkurt_mfa_balanced_npy/V8/bbab8n_34250_AW.npy"),
    ("V9  — Velar (G/K/NG)",
     "data/processed/visemes_bozkurt_mfa_balanced_npy/V9/bbac1a_42750_G.npy"),
    ("V10 — R-coloured (R)",
     "data/processed/visemes_bozkurt_mfa_balanced_npy/V10/bbaczp_36500_R.npy"),
    ("V11 — Alveolar (L/D/N/T)  largest class",
     "data/processed/visemes_bozkurt_mfa_balanced_npy/V11/bbab8n_17750_L.npy"),
    ("V12 — Alveolar fricative (S/Z)",
     "data/processed/visemes_bozkurt_mfa_balanced_npy/V12/bbab9s_39250_S.npy"),
    ("V13 — Post-alveolar (CH/SH)  hardest class 79%",
     "data/processed/visemes_bozkurt_mfa_balanced_npy"
     "/V13/bbih8n_34500_CH.npy"),
    ("V14 — Dental fricative (TH/DH)",
     "data/processed/visemes_bozkurt_mfa_balanced_npy"
     "/V14/bbaf3a_44750_TH.npy"),
    ("V15 — Labiodental (F/V)",
     "data/processed/visemes_bozkurt_mfa_balanced_npy/V15/bbae4n_33750_F.npy"),
    ("V16 — Bilabial (M/B/P)",
     "data/processed/visemes_bozkurt_mfa_balanced_npy/V16/bbbm1a_39500_M.npy")]


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


# Sample clip selection menu
def resolve_existing_clips(
    clips: List[Tuple[str, str]],
    data_root: Path,
) -> List[Tuple[str, str]]:
    """
    Return clips whose .npy file exists.
    Falls back to the first available .npy in the class folder
    if the exact file is missing.
    """
    resolved = []
    for label, rel_path in clips:
        full = PROJECT_ROOT / rel_path
        if full.exists():
            resolved.append((label, str(full)))
            continue
        # try to find any .npy in the same class folder
        class_name = label.split()[0].strip()
        class_folder = data_root / class_name
        if class_folder.exists():
            npys = sorted(class_folder.glob("*.npy"))
            if npys:
                resolved.append((label, str(npys[0])))
                continue
        print(f"  [warn] No clip found for {label} — skipping.")
    return resolved


def pick_clip(
    clips: List[Tuple[str, str]],
) -> Tuple[str, str] | None:
    """Print numbered menu and return the chosen (label, path) tuple"""
    print("\n  Available sample clips:")
    print("  " + "─" * 58)
    for i, (label, _) in enumerate(clips, 1):
        print(f"  {i:>2}. {label}")
    print("  " + "─" * 58)
    print("   0. Exit")
    print()
    while True:
        raw = input("  Enter number: ").strip()
        if raw == "0":
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(clips):
            return clips[int(raw) - 1]
        print("  Invalid choice — please enter a number from the list.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Viseme-level inference demo")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "training/checkpoints_bozkurt_viseme/bozkurt_viseme_best_model.pth")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT
        / "data/processed/visemes_bozkurt_mfa_balanced_npy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 60)
    print("   Viseme-Level Inference Demo  —  Bozkurt 3D CNN")
    print("=" * 60)
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Device     : {device}")
    print("\n  Loading model...", end=" ", flush=True)
    model, viseme_classes = load_model(args.checkpoint, device)
    print(f"done  ({len(viseme_classes)} classes, "
          f"val acc 82.78%)")
    clips = resolve_existing_clips(SAMPLE_CLIPS, args.data_root)
    if not clips:
        print("\n  No sample clips found. "
              "Check --data-root path.")
        sys.exit(1)
    while True:
        choice = pick_clip(clips)
        if choice is None:
            print("\n  Exiting demo.\n")
            break
        label, clip_path = choice
        ground_truth = label.split()[0].strip()
        print(f"\n  Running inference on: {os.path.basename(clip_path)}")
        try:
            tensor = preprocess_clip(
                Path(clip_path),
                sequence_length=SEQUENCE_LENGTH,
                height=VIDEO_HEIGHT,
                width=VIDEO_WIDTH)
            predictions = run_inference(
                model, tensor, viseme_classes, device)
            print_results(predictions, ground_truth)
        except Exception as exc:
            print(f"\n  Error processing clip: {exc}\n")


if __name__ == "__main__":
    main()
