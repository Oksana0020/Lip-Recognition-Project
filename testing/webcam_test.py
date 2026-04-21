"""
Live webcam word recognition test.
Records a short clip from the laptop camera, applies the same
dlib lip-crop + grayscale preprocessing used during training,
runs inference with the word-level 3D-CNN and prints the top
predictions.
"""

from __future__ import annotations
import argparse
import msvcrt
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))
CHECKPOINT = (
    PROJECT_ROOT
    / "training"
    / "checkpoints_words"
    / "best_model_words.pth")
PREDICTOR_PATH = PROJECT_ROOT / "shape_predictor_68_face_landmarks.dat"
LIP_INDICES = list(range(48, 68))
PAD_FRAC = 0.35
TARGET_FRAMES = 8
TARGET_H = 64
TARGET_W = 64
WIN = "Webcam — press Q to quit"


def build_dlib_tools():
    """Load dlib detector and predictor if available"""
    try:
        import dlib
        if not PREDICTOR_PATH.exists():
            print(
                f"WARNING: {PREDICTOR_PATH.name} not found — "
                "using fixed centre crop.")
            return None, None
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(str(PREDICTOR_PATH))
        print("dlib loaded — lip detection active.")
        return detector, predictor
    except ImportError:
        print("WARNING: dlib not installed — using fixed centre crop.")
        return None, None


def detect_lip_bbox(frame_bgr, detector, predictor):
    """Return [x0, y0, x1, y1] lip bbox or None on failure."""
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    if not faces:
        return None
    shape = predictor(gray, faces[0])
    xs = [shape.part(i).x for i in LIP_INDICES]
    ys = [shape.part(i).y for i in LIP_INDICES]
    pad_x = int((max(xs) - min(xs)) * PAD_FRAC)
    pad_y = int((max(ys) - min(ys)) * PAD_FRAC)
    return [
        max(0, min(xs) - pad_x),
        max(0, min(ys) - pad_y),
        min(w, max(xs) + pad_x),
        min(h, max(ys) + pad_y)]


def fixed_crop(frame):
    """Fixed percentage crop fallback."""
    h, w = frame.shape[:2]
    return frame[int(h * 0.66):int(h * 0.92), int(w * 0.33):int(w * 0.67)]


def crop_lip(frame_bgr, detector, predictor, bbox_cache: list):
    """Crop lip region; reuse cached bbox if detection fails"""
    bbox = None
    if detector is not None:
        bbox = detect_lip_bbox(frame_bgr, detector, predictor)
    if bbox is not None:
        bbox_cache.clear()
        bbox_cache.append(bbox)
    elif bbox_cache:
        bbox = bbox_cache[0]
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        return frame_bgr[y0:y1, x0:x1]
    return fixed_crop(frame_bgr)


def preprocess_frames(
    frames_bgr: list,
    detector,
    predictor,
) -> torch.Tensor:
    """Resample, crop lip, grayscale, resize, normalise -> model tensor"""
    bbox_cache: list = []
    n = len(frames_bgr)
    indices = np.linspace(0, n - 1, TARGET_FRAMES, dtype=int)
    sampled = [frames_bgr[i] for i in indices]
    processed = []
    for frame in sampled:
        cropped = crop_lip(frame, detector, predictor, bbox_cache)
        resized = cv2.resize(cropped, (TARGET_W, TARGET_H))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        processed.append(gray[..., np.newaxis])
    arr = np.stack(processed, axis=0).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0)
    return tensor


def load_model(checkpoint_path: Path, device: torch.device):
    from training.model import ThreeDimensionalCNN
    from training.train_utils import _remap_state_dict
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "index_to_word" in ckpt:
        index_to_word = {int(k): v for k, v in ckpt["index_to_word"].items()}
    else:
        label_map = ckpt["label_map"]
        index_to_word = {v: k for k, v in label_map.items()}
    model = ThreeDimensionalCNN(
        num_classes=len(index_to_word),
        input_channels=1)
    model.load_state_dict(_remap_state_dict(ckpt["model_state_dict"]))
    model.to(device).eval()
    return model, index_to_word


def record_clip(cap: cv2.VideoCapture, seconds: float) -> list:
    """Show countdown then record; returns BGR frame list."""
    print("Get ready. Recording starts in 3 seconds.")
    deadline = time.time() + 3.0
    while time.time() < deadline:
        ok, frame = cap.read()
        if ok:
            preview = frame.copy()
            cv2.putText(
                preview, "GET READY...", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 200, 255), 3)
            cv2.imshow(WIN, preview)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)

    print(f"Recording for {seconds:.1f}s - say your word NOW!")
    frames = []
    deadline = time.time() + seconds
    while time.time() < deadline:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        preview = frame.copy()
        remaining = max(0.0, deadline - time.time())
        cv2.putText(
            preview, f"REC  {remaining:.1f}s", (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 220), 3)
        cv2.imshow(WIN, preview)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)
    print(f"Captured {len(frames)} frames.")
    return frames


def _enter_pressed() -> bool:
    """Non-blocking check for Enter key in terminal Windows"""
    if msvcrt.kbhit():
        ch = msvcrt.getwch()
        return ch in ("\r", "\n")
    return False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Webcam word recognition test")
    parser.add_argument("--seconds", type=float, default=1.0,
                        help="Recording duration in seconds (default: 1.0). "
                        "Speak immediately when REC appears.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument(
        "--save", type=Path, default=None,
        help="Save raw captured frames as .npy")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    device = torch.device("cpu")
    print("Loading model...")
    model, index_to_word = load_model(args.checkpoint, device)
    print(f"Ready - {len(index_to_word)} word classes.\n")
    # Load dlib at startup
    detector, predictor = build_dlib_tools()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}. Try --camera 1.")
        sys.exit(1)
    print("Camera open — live preview active.")
    print(
        f"Recording: {args.seconds:.1f}s — "
        "speak IMMEDIATELY when 'REC' appears.\n")
    print("Tips:")
    print("  • Face the camera straight on, good lighting")
    print("  • Keep your face centred in the green lip box")
    print("  • Say the word right as REC starts — don't wait")
    print("  • Press Q in the video window to quit\n")

    try:
        while True:
            # live idle preview while waiting for Enter
            print("Press ENTER to record a word...", end="", flush=True)
            bbox_cache: list = []
            while True:
                ok, frame = cap.read()
                if ok:
                    preview = frame.copy()
                    # Draw live lip bbox so user can verify detection
                    if detector is not None:
                        bbox = detect_lip_bbox(frame, detector, predictor)
                        if bbox is not None:
                            bbox_cache.clear()
                            bbox_cache.append(bbox)
                    if bbox_cache:
                        x0, y0, x1, y1 = bbox_cache[0]
                        cv2.rectangle(
                            preview, (x0, y0), (x1, y1),
                            (0, 220, 0), 2)
                    cv2.putText(
                        preview, "READY - press ENTER", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2)
                    cv2.imshow(WIN, preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    raise KeyboardInterrupt
                if _enter_pressed():
                    print()
                    break
            frames = record_clip(cap, args.seconds)
            if args.save:
                arr = np.stack(frames, axis=0)
                np.save(args.save, arr)
                print(f"Saved raw frames -> {args.save}")
            print("Processing...")
            tensor = preprocess_frames(frames, detector, predictor)
            with torch.no_grad():
                logits = model(tensor.to(device))
                probs = torch.softmax(logits, dim=1)[0]
            top_probs, top_idx = torch.topk(probs, args.top_k)
            print("\n" + "=" * 40)
            print("  TOP PREDICTIONS")
            print("=" * 40)
            for rank, (prob, idx) in enumerate(zip(top_probs.tolist(),
                                                   top_idx.tolist()), 1):
                word = index_to_word[idx]
                bar = "#" * int(prob * 30)
                print(f"  {rank}. {word:<12}  {prob*100:5.1f}%  {bar}")
            print("=" * 40)
            print(f"\n  -> '{index_to_word[top_idx[0].item()]}'\n")
    except KeyboardInterrupt:
        print("\n  Exiting.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
