"""
Script to precompute per-sample lip bounding boxes for word-level training.
Uses dlib face detection + 68-point landmark predictor to find lip region
"""

from __future__ import annotations
import json
import time
from pathlib import Path
import cv2
import dlib
import numpy as np
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "shape_predictor_68_face_landmarks.dat"
WORDS_ROOT = BASE_DIR / "data" / "processed" / "words_by_label"
OUTPUT_JSON = BASE_DIR / "data" / "processed" / "words_lip_bboxes.json"
LIP_IDX = list(range(48, 68))
PAD_FRAC = 0.35
FALLBACK_Y0_FRAC = 0.66
FALLBACK_Y1_FRAC = 0.92
FALLBACK_X0_FRAC = 0.33
FALLBACK_X1_FRAC = 0.67
SKIP_CLASSES = {"sp", "sil"}


def fixed_fallback_bbox(frame_h: int, frame_w: int) -> list[int]:
    """Return fixed percentage-based lip crop as [x0, y0, x1, y1]"""
    return [
        int(frame_w * FALLBACK_X0_FRAC),
        int(frame_h * FALLBACK_Y0_FRAC),
        int(frame_w * FALLBACK_X1_FRAC),
        int(frame_h * FALLBACK_Y1_FRAC),
    ]


def detect_lip_bbox(
    frame_bgr: np.ndarray,
    detector: dlib.fhog_object_detector,
    predictor: dlib.shape_predictor,
) -> list[int] | None:
    """
    Detect lips in a BGR frame using dlib
    Returns [x0, y0, x1, y1] with padding
    """
    frame_height, frame_width = frame_bgr.shape[:2]
    grayscale_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    detected_faces = detector(grayscale_frame, 1)
    if not detected_faces:
        return None
    largest_face_detection = max(
        detected_faces,
        key=lambda face_rect: face_rect.width() * face_rect.height(),)
    facial_landmark_shape = predictor(
        grayscale_frame, largest_face_detection)
    lip_landmark_points = np.array(
        [(facial_landmark_shape.part(i).x, facial_landmark_shape.part(i).y)
            for i in LIP_IDX],
        dtype=np.int32,)
    x_min, y_min = lip_landmark_points.min(axis=0)
    x_max, y_max = lip_landmark_points.max(axis=0)
    lip_bbox_width = x_max - x_min
    lip_bbox_height = y_max - y_min
    lip_padding_x = int(lip_bbox_width * PAD_FRAC)
    lip_padding_y = int(lip_bbox_height * PAD_FRAC)
    x0 = max(0, x_min - lip_padding_x)
    y0 = max(0, y_min - lip_padding_y)
    x1 = min(frame_width, x_max + lip_padding_x)
    y1 = min(frame_height, y_max + lip_padding_y)
    return [int(x0), int(y0), int(x1), int(y1)]


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"dlib shape predictor is not found: {MODEL_PATH}\n"
            "Download from http://dlib.net/files/"
            "shape_predictor_68_face_landmarks.dat.bz2")
    print("Loading dlib models")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(MODEL_PATH))
    print("Models loaded.\n")
    # Collect all .npy files
    all_word_clip_paths: list[Path] = []
    for word_dir in sorted(WORDS_ROOT.iterdir()):
        if not word_dir.is_dir():
            continue
        if word_dir.name in SKIP_CLASSES:
            continue
        all_word_clip_paths.extend(sorted(word_dir.glob("*.npy")))
    print(f"Total .npy clip files to process {len(all_word_clip_paths)}")
    # Load existing results
    lip_bbox_lookup: dict[str, list[int]] = {}
    if OUTPUT_JSON.exists():
        with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
            lip_bbox_lookup = json.load(f)
        already_done = len(lip_bbox_lookup)
        print(
            f"Loaded {already_done} already computed entries .\n")
    fallback_count = 0
    clips_processed_count = 0
    save_interval_clips = 500
    processing_start_time = time.time()
    for clip_path in tqdm(
            all_word_clip_paths, desc="Detecting lips", unit="clip"):
        clip_path_key = str(clip_path)
        if clip_path_key in lip_bbox_lookup:
            continue
        try:
            clip_frames = np.load(clip_path, allow_pickle=False)
        except Exception:
            lip_bbox_lookup[clip_path_key] = fixed_fallback_bbox(288, 360)
            fallback_count += 1
            clips_processed_count += 1
            continue
        clip_frame_height = clip_frames.shape[1]
        clip_frame_width = clip_frames.shape[2]
        middle_frame_index = len(clip_frames) // 2
        detected_lip_bbox = None
        candidate_indices = [
            middle_frame_index, 0, len(clip_frames) - 1]
        for candidate_frame_index in candidate_indices:
            if candidate_frame_index >= len(clip_frames):
                continue
            candidate_frame_bgr = clip_frames[candidate_frame_index]
            detected_lip_bbox = detect_lip_bbox(
                candidate_frame_bgr, detector, predictor)
            if detected_lip_bbox is not None:
                break
        if detected_lip_bbox is None:
            lip_bbox_lookup[clip_path_key] = fixed_fallback_bbox(
                clip_frame_height, clip_frame_width)
            fallback_count += 1
        else:
            lip_bbox_lookup[clip_path_key] = detected_lip_bbox
        clips_processed_count += 1
        if clips_processed_count % save_interval_clips == 0:
            OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                json.dump(lip_bbox_lookup, f)

    # Final save after processing all clips
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(lip_bbox_lookup, f, indent=2)
    total_elapsed_seconds = time.time() - processing_start_time
    successful_detections = clips_processed_count - fallback_count
    print(f"\nDone in {total_elapsed_seconds/60:.1f} minuts")
    print(f"  Processed : {clips_processed_count}")
    detection_pct = (
        100 * successful_detections / max(1, clips_processed_count))
    print(f"  Detected  : {successful_detections} ({detection_pct:.1f}%)")
    print(f"  Fallback  : {fallback_count}")
    print(f"  Saved to  : {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
