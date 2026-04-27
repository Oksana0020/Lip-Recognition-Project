"""
Batch preprocess all videos in testing/my_words:
Extracts lip region using dlib facial landmarks
Saves both cropped videos and .npy arrays
"""
import cv2
import dlib
import numpy as np
from pathlib import Path

LIP_IDX = list(range(48, 68))
MODEL_PATH = (
    Path(__file__).parent.parent / "shape_predictor_68_face_landmarks.dat")
INPUT_DIR = (
    Path(__file__).parent.parent / "testing" / "my_words")
CROPPED_VIDEO_DIR = (
    Path(__file__).parent.parent / "testing" / "my_words_cropped_videos")
NPY_DIR = (
    Path(__file__).parent.parent / "testing" / "my_words_npy")
CROP_SIZE = 64
CROPPED_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
NPY_DIR.mkdir(parents=True, exist_ok=True)


def extract_lip_frames(video_path, predictor, detector):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)
        if not faces:
            continue
        face = faces[0]
        shape = predictor(gray, face)
        pts = np.array(
            [(shape.part(i).x, shape.part(i).y) for i in LIP_IDX],
            dtype=np.int32)
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        w, h = (x_max - x_min), (y_max - y_min)
        px, py = int(w * 0.2), int(h * 0.2)
        x0 = max(0, x_min - px)
        y0 = max(0, y_min - py)
        x1 = min(frame_bgr.shape[1], x_max + px)
        y1 = min(frame_bgr.shape[0], y_max + py)
        lip_crop = frame_bgr[y0:y1, x0:x1]
        if lip_crop.size == 0:
            continue
        lip_resized = cv2.resize(
            lip_crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)
        frames.append(lip_resized)
    cap.release()
    return np.array(frames)


def save_video(frames, out_path, fps=25):
    if len(frames) == 0:
        return False
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()
    return True


def main():
    if not MODEL_PATH.exists():
        print(f"Model file not found: {MODEL_PATH}")
        return
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(MODEL_PATH))
    VIDEO_EXTS = [
        ".mp4", ".mov", ".avi", ".mkv", ".mpg", ".mpeg"]
    for vid in INPUT_DIR.iterdir():
        if vid.suffix.lower() not in VIDEO_EXTS:
            continue
        print(f"Processing {vid.name} ...")
        frames = extract_lip_frames(vid, predictor, detector)
        if len(frames) == 0:
            print("  No lip frames found!")
            continue
        cropped_video_path = CROPPED_VIDEO_DIR / (vid.stem + "_cropped.mp4")
        save_video(frames, cropped_video_path)
        print(
            f"  Cropped video saved: {cropped_video_path.name}")
        npy_path = NPY_DIR / (vid.stem + ".npy")
        np.save(npy_path, frames)
        print(
            f"  NPY array saved: {npy_path.name}")


if __name__ == "__main__":
    main()
