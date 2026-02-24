"""
Word-level extraction pipeline for the GRID dataset:
1. Word extraction from videos using alignment files
2. Saving word-level frames and metadata

"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))


def find_all_grid_videos() -> List[Dict[str, str]]:
    """Find all video and alignment file pairs in the GRID dataset."""
    grid_data_path = BASE_DIR / "data" / "grid" / "GRID dataset full"
    matching_pairs: List[Dict[str, str]] = []

    for speaker_num in range(1, 35):
        speaker_id = f"s{speaker_num}"
        videos_path = grid_data_path / speaker_id
        alignments_path = grid_data_path / "alignments" / speaker_id

        if not videos_path.exists() or not alignments_path.exists():
            continue

        video_files = [f for f in os.listdir(videos_path) if f.endswith(".mpg")]

        for video_file in video_files:
            base_name = os.path.splitext(video_file)[0]
            alignment_file = f"{base_name}.align"
            video_path = videos_path / video_file
            alignment_path = alignments_path / alignment_file

            if alignment_path.exists():
                matching_pairs.append(
                    {
                        "video_path": str(video_path),
                        "alignment_path": str(alignment_path),
                        "base_name": base_name,
                        "speaker_id": speaker_id,
                    }
                )

    return matching_pairs


def parse_grid_alignment_file(alignment_file: Path) -> List[Dict[str, Any]]:
    """Parse a GRID .align file and return word-level alignments."""
    word_alignments: List[Dict[str, Any]] = []

    with open(alignment_file, "r", encoding="utf-8") as f:
        content = f.read().strip()

    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 3:
            continue

        start_frame = int(parts[0])
        end_frame = int(parts[1])
        word = parts[2]

        start_time = start_frame / 25000.0
        end_time = end_frame / 25000.0
        duration = end_time - start_time

        word_alignments.append(
            {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "word": word,
            }
        )

    return word_alignments


def extract_word_frames(
    video_path: str,
    word_info: Dict[str, Any],
) -> Optional[np.ndarray]:
    """Extract video frames corresponding to a word segment."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    start_frame_idx = int(word_info["start_time"] * fps)
    end_frame_idx = int(word_info["end_time"] * fps)

    frames: List[np.ndarray] = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)

    for _ in range(start_frame_idx, end_frame_idx + 1):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    if frames:
        return np.array(frames)
    return None


def save_word_segment(
    frames: np.ndarray,
    word_info: Dict[str, Any],
    video_name: str,
    speaker_id: str,
    output_root: Path,
) -> bool:
    """Save frames (.npy) and metadata (.json) for a word segment."""
    word_label = str(word_info["word"]).lower()
    if word_label == "sil":
        return False

    word_dir = output_root / word_label
    word_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{video_name}_word_{int(word_info['start_frame'])}_{word_label}"

    frames_file = word_dir / f"{filename}.npy"
    np.save(frames_file, frames)

    metadata = {
        "word": word_info["word"],
        "start_frame": word_info["start_frame"],
        "end_frame": word_info["end_frame"],
        "start_time": word_info["start_time"],
        "end_time": word_info["end_time"],
        "duration": word_info["duration"],
        "num_frames": len(frames),
        "source_video": video_name,
        "speaker_id": speaker_id,
        "frame_shape": list(frames.shape),
    }

    metadata_file = word_dir / f"{filename}.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return True


def process_single_video(
    video_info: Dict[str, str],
    output_root: Path,
) -> Dict[str, Any]:
    """Process one video and extract all non-sil word segments."""
    video_path = Path(video_info["video_path"])
    alignment_path = Path(video_info["alignment_path"])
    base_name = video_info["base_name"]
    speaker_id = video_info["speaker_id"]

    words_extracted = 0
    words_failed = 0

    try:
        word_alignments = parse_grid_alignment_file(alignment_path)

        for word_info in word_alignments:
            if str(word_info["word"]).lower() == "sil":
                continue

            frames = extract_word_frames(str(video_path), word_info)
            if frames is not None and len(frames) > 0:
                saved = save_word_segment(
                    frames=frames,
                    word_info=word_info,
                    video_name=base_name,
                    speaker_id=speaker_id,
                    output_root=output_root,
                )
                if saved:
                    words_extracted += 1
            else:
                words_failed += 1

        return {
            "video": base_name,
            "speaker_id": speaker_id,
            "status": "success",
            "words_extracted": words_extracted,
            "words_failed": words_failed,
        }

    except Exception as exc:
        return {
            "video": base_name,
            "speaker_id": speaker_id,
            "status": "failed",
            "error": str(exc),
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Word-level extraction pipeline for GRID dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/words_by_label",
        help="Output directory for word segments",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of videos to process",
    )
    args = parser.parse_args()

    output_root = BASE_DIR / args.output
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GRID Word-Level Extraction Pipeline")
    print("=" * 60)
    print(f"Output directory: {output_root}")
    if args.limit:
        print(f"Processing limit: {args.limit} videos")
    print("=" * 60 + "\n")

    print("Step 1: Finding GRID videos...")
    video_pairs = find_all_grid_videos()
    if args.limit:
        video_pairs = video_pairs[: args.limit]
    print(f"Found {len(video_pairs)} video-alignment pairs\n")

    print("Step 2: Extracting word segments from videos...")
    total_extracted = 0
    total_failed = 0
    failed_videos = 0

    for video_info in tqdm(video_pairs, desc="Processing videos"):
        result = process_single_video(video_info, output_root)

        if result["status"] == "success":
            total_extracted += int(result["words_extracted"])
            total_failed += int(result["words_failed"])
        else:
            failed_videos += 1

    print("\n" + "=" * 60)
    print("Extraction Summary")
    print("=" * 60)
    print(f"Videos processed: {len(video_pairs)}")
    print(f"Videos failed: {failed_videos}")
    print(f"Word segments extracted: {total_extracted}")
    print(f"Word segments failed: {total_failed}")
    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
