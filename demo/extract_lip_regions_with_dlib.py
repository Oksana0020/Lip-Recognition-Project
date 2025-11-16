"""
Sprint 2 – Lip Region Extraction Demo
- Extracts lip regions from GRID dataset videos using dlib facial landmarks.
- Saves both the original color crop (unscaled) and a resized 96×96 color crop.
- Prints a detailed extraction summary and exports metadata to JSON.
"""

from pathlib import Path
import cv2
import dlib
import numpy as np
import json

# Constants
LIP_IDX = list(range(48, 68))  # mouth region indices
MODEL_PATH = Path(__file__).parent.parent / "shape_predictor_68_face_landmarks.dat"
OUTPUT_DIR = Path(__file__).parent / "lip_regions"

def extract_lip_regions_from_video(video_path: Path, max_frames: int = 10):
    """Extract up to `max_frames` lip crops and save."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(MODEL_PATH))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    duration = total_frames / fps

    print(f"VIDEO: {video_path.name}")
    print(f"FPS: {fps:.2f}")
    print(f"TOTAL FRAMES: {total_frames}")
    print(f"DURATION: {duration:.3f} seconds\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"EXTRACTING LIP REGIONS (max {max_frames} frames):")
    print("-" * 60)

    results = []
    frame_id = 0
    processed = 0

    EVERY_N = 10       # process every 10th frame
    PAD_FRAC = 0.2     # 20% padding around the lip box

    while processed < max_frames:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_id += 1
        if frame_id % EVERY_N != 0:
            continue

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)
        if not faces:
            continue

        face = faces[0]
        shape = predictor(gray, face)
        pts = np.array([(shape.part(i).x, shape.part(i).y) for i in LIP_IDX], dtype=np.int32)

        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        w, h = (x_max - x_min), (y_max - y_min)

        # Padding
        px, py = int(w * PAD_FRAC), int(h * PAD_FRAC)
        x0 = max(0, x_min - px)
        y0 = max(0, y_min - py)
        x1 = min(frame_bgr.shape[1], x_max + px)
        y1 = min(frame_bgr.shape[0], y_max + py)

        # Color crops
        lip_orig = frame_bgr[y0:y1, x0:x1]
        if lip_orig.size == 0:
            continue

        lip_96 = cv2.resize(lip_orig, (96, 96), interpolation=cv2.INTER_AREA)

        frame_time = frame_id / fps
        base_name = f"lip_frame_{frame_id:06d}"

        path_orig = OUTPUT_DIR / f"{base_name}.jpg"
        path_96 = OUTPUT_DIR / f"{base_name}_96x96.jpg"
        cv2.imwrite(str(path_orig), lip_orig)
        cv2.imwrite(str(path_96), lip_96)

        results.append({
            "frame_number": frame_id,
            "time_seconds": round(frame_time, 3),
            "bbox": {
                "x_min": int(x0), "y_min": int(y0),
                "x_max": int(x1), "y_max": int(y1),
                "width": int(x1 - x0), "height": int(y1 - y0)
            },
            "paths": {
                "original": str(path_orig),
                "resized_96x96": str(path_96)
            }
        })
        processed += 1

        print(f"Frame {frame_id:6d} | Time: {frame_time:6.3f}s | Lip: {x1 - x0}x{y1 - y0} | Saved: {path_orig.name}, {path_96.name}")

    cap.release()

    # Average lip box size
    if results:
        avg_w = np.mean([r["bbox"]["width"] for r in results])
        avg_h = np.mean([r["bbox"]["height"] for r in results])
    else:
        avg_w = avg_h = 0.0

    print("\nEXTRACTION COMPLETE")
    print(f"PROCESSED: {processed} frames with lip regions")
    print(f"SAVED TO: {OUTPUT_DIR}")
    print(f"FINAL MODEL CROP: 96×96 (color)")
    print(f"AVERAGE LIP BOX: {avg_w:.1f}×{avg_h:.1f} pixels\n")

    # JSON summary
    json_path = OUTPUT_DIR.parent / "lip_extraction_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "video": str(video_path),
            "fps": fps,
            "total_frames": total_frames,
            "duration": duration,
            "processed": processed,
            "average_lip_box": [avg_w, avg_h],
            "lip_regions": results
        }, f, indent=2)

    print(f"Results saved to: {json_path.name}")
    return results


if __name__ == "__main__":
    # try first GRID video
    s1 = Path(__file__).parent.parent / "data" / "grid" / "GRID dataset full" / "s1"
    vids = sorted(s1.glob("*.mpg")) if s1.exists() else []
    if not vids:
        print("No .mpg videos found in data/grid/GRID dataset full/s1/")
    else:
        extract_lip_regions_from_video(vids[0], max_frames=10)