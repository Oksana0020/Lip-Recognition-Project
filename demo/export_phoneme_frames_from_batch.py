"""
Read batch MFA results and save one frame per phoneme
at the midpoint time for manual inspection.
"""

import json
from pathlib import Path

import cv2


BATCH_FILE = Path(__file__).parent / "batch_mfa_results.json"
FRAMES_ROOT = Path(__file__).parent / "phoneme_frames"


def save_frames_for_video(video_path: Path, phoneme_intervals, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    count_saved = 0
    for idx, ph in enumerate(phoneme_intervals, start=1):
        start_t = ph.get("start_time", 0.0)
        end_t = ph.get("end_time", start_t)
        mid_t = 0.5 * (start_t + end_t)

        frame_idx = int(mid_t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        label = f"{idx:03d}_{ph['phoneme']}_{mid_t:.3f}s"
        fname = f"{video_path.stem}_{label}.jpg"
        out_path = out_dir / fname

        cv2.putText(
            frame,
            f"{ph['phoneme']} @ {mid_t:.3f}s",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imwrite(str(out_path), frame)
        count_saved += 1

    cap.release()
    print(f"Saved {count_saved} frames for {video_path.name} -> {out_dir}")


def main():
    if not BATCH_FILE.exists():
        raise FileNotFoundError(
            f"{BATCH_FILE} not found. Run run_mfa_on_sample.py first."
        )

    with open(BATCH_FILE, "r", encoding="utf-8") as f:
        batch = json.load(f)

    results = batch.get("results", [])
    if not results:
        print("No results in batch_mfa_results.json")
        return

    print(f"Exporting frames for {len(results)} videos...\n")

    for i, vid_res in enumerate(results, start=1):
        video_path = Path(vid_res["video_file"])
        base_name = Path(video_path).stem
        out_dir = FRAMES_ROOT / base_name
        phoneme_intervals = vid_res.get("phoneme_intervals") or []

        print(
            "[{}/{}] {} ({} phonemes)".format(
                i, len(results), video_path.name, len(phoneme_intervals)
            )
        )
        save_frames_for_video(video_path, phoneme_intervals, out_dir)

    print(f"\nAll frames exported under: {FRAMES_ROOT}")


if __name__ == "__main__":
    main()
