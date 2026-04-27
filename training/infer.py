"""Unified inference for word and viseme lip-reading models."""


import sys
import numpy as np
import cv2
import torch
import argparse
import json
from pathlib import Path
from training.model import ThreeDimensionalCNN
from training.device import resolve_device
from training.train_utils import load_checkpoint


_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
PROJECT_ROOT = _PROJECT_ROOT
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.mpg', '.mpeg'}


def load_viseme_phoneme_map():
    mapping_path = PROJECT_ROOT / "mapping" / "bozkurt_viseme_map.csv"
    viseme_to_phonemes = {}
    if mapping_path.exists():
        import csv
        with open(mapping_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                viseme = row["viseme_class"].strip()
                phonemes = [p.strip() for p in row["phonemes"].split(",")
                            if p.strip()]
                viseme_to_phonemes[viseme] = phonemes
    return viseme_to_phonemes


def load_frames_from_video(path: Path) -> np.ndarray:
    """Read all BGR frames from a video file into an ndarray [T, H, W, 3]."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    bgr_frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        bgr_frames.append(frame)
    cap.release()
    if not bgr_frames:
        raise ValueError(f"No frames read from video: {path}")
    return np.stack(bgr_frames, axis=0)


def preprocess_clip(
        frames: np.ndarray,
        num_frames: int = 8,
        height: int = 64,
        width: int = 64,
        bbox=None) -> torch.Tensor:
    """Resample, crop, resize and normalise frames. Returns [1, 1, T, H, W]"""
    n = frames.shape[0]
    if n >= num_frames:
        idx = np.linspace(0, n - 1, num_frames, dtype=int)
        frames = frames[idx]
    else:
        padded = list(frames)
        while len(padded) < num_frames:
            padded.extend(list(frames)[:num_frames - len(padded)])
        frames = np.stack(padded[:num_frames], axis=0)
    result = []
    for frame in frames:
        h, w = frame.shape[:2]
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            frame = frame[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
        else:
            frame = frame[
                int(h * 0.66):int(h * 0.92),
                int(w * 0.33):int(w * 0.67)]
        frame = cv2.resize(frame, (width, height))
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result.append(frame)
    arr = np.stack(result, axis=0).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lip-reading inference")
    parser.add_argument("task", choices=["word", "viseme"])
    parser.add_argument(
        "clip", type=Path,
        help="Path to .npy clip file or video (mp4/mov/avi/mpg)")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    device = resolve_device()
    # load checkpoint, discover label map and class count
    ckpt = torch.load(
        args.checkpoint, map_location=device, weights_only=False)
    if "label_map" in ckpt:
        label_map: dict = ckpt["label_map"]
        index_to_label = {v: k for k, v in label_map.items()}
    elif "index_to_word" in ckpt:
        index_to_label = {int(k): v for k, v in ckpt["index_to_word"].items()}
        label_map = {v: k for k, v in index_to_label.items()}
    elif "viseme_classes" in ckpt:
        classes = ckpt["viseme_classes"]
        index_to_label = {i: c for i, c in enumerate(classes)}
        label_map = {c: i for i, c in enumerate(classes)}
    else:
        raise KeyError(
            "Checkpoint missing label_map"
            " / index_to_word / viseme_classes")
    num_classes = len(label_map)
    model = ThreeDimensionalCNN(num_classes=num_classes, input_channels=1)
    load_checkpoint(args.checkpoint, model, device)
    print(f"Model loaded ({num_classes} classes, task={args.task})")
    # load clip accept both .npy and video files
    if args.clip.suffix.lower() in VIDEO_EXTENSIONS:
        print(f"Loading video: {args.clip.name}")
        frames = load_frames_from_video(args.clip)
        print(f"Video frames: {frames.shape}")
    else:
        frames = np.load(args.clip, allow_pickle=False)
    print(f"Clip shape: {frames.shape}")
    # optional lip bbox lookup for word task
    bbox = None
    if (args.task == "word"
            and args.clip.suffix.lower() not in VIDEO_EXTENSIONS):
        bbox_json = (
            PROJECT_ROOT / "data" / "processed" / "words_lip_bboxes.json")
        if bbox_json.exists():
            with open(bbox_json, encoding="utf-8") as f:
                raw = json.load(f)
            raw = {k.lower(): v for k, v in raw.items()}
            key = str(args.clip.resolve()).lower()
            bbox = raw.get(key) or raw.get(key.replace("\\", "/"))
    tensor = preprocess_clip(frames, bbox=bbox).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    top_k = min(args.top_k, num_classes)
    top_probs, top_indices = probs.topk(num_classes)
    filtered = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        label = index_to_label.get(idx, str(idx))
        if args.task == "word" and len(label) == 1:
            continue  # skip one-letter words
        filtered.append((prob, idx))
        if len(filtered) == top_k:
            break
    print(f"\nTop-{len(filtered)} predictions:")
    viseme_to_phonemes = None
    if args.task == "viseme":
        viseme_to_phonemes = load_viseme_phoneme_map()
    for rank, (prob, idx) in enumerate(filtered, start=1):
        label = index_to_label.get(idx, str(idx))
        if args.task == "viseme" and viseme_to_phonemes:
            phonemes = viseme_to_phonemes.get(label, [])
            phoneme_str = ', '.join(phonemes) if phonemes else '-'
            print(
                f"  {rank}. {label:<8} {prob * 100:.2f}%   "
                f"phonemes: {phoneme_str}")
        else:
            print(f"  {rank}. {label:<15} {prob * 100:.2f}%")


if __name__ == "__main__":
    main()
