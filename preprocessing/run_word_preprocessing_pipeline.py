"""
Complete pipeline for word-level preprocessing for GRID dataset:
1. Word extraction from videos using alignment files
2. Saving word-level frame segments and metadata
3. Statistics collection and summary reporting
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
DEFAULT_TARGET_FRAME_COUNT = 40
DEFAULT_TARGET_FRAME_HEIGHT = 48
DEFAULT_TARGET_FRAME_WIDTH = 48


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

        video_files = [
            f for f in os.listdir(videos_path) if f.endswith(".mpg")
        ]

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
    """ Extract video frames corresponding to a word segment."""
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


def standardize_word_frames(
    frames: np.ndarray,
    target_frame_count: int,
    target_frame_height: int,
    target_frame_width: int,
) -> np.ndarray:
    """Resample and resize frames to a fixed clip shape."""
    current_frame_count = int(frames.shape[0])

    if current_frame_count > target_frame_count:
        frame_indices = np.linspace(
            0,
            current_frame_count - 1,
            target_frame_count,
            dtype=int,
        )
        frames = frames[frame_indices]
    elif current_frame_count < target_frame_count:
        frames_needed = target_frame_count - current_frame_count
        padding_frames = np.repeat(frames[-1:], frames_needed, axis=0)
        frames = np.concatenate([frames, padding_frames], axis=0)

    resized_frames: List[np.ndarray] = []
    for frame in frames:
        resized_frame = cv2.resize(
            frame,
            (target_frame_width, target_frame_height),
        )
        resized_frames.append(resized_frame)

    return np.asarray(resized_frames, dtype=np.uint8)


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
    filename = (
        f"{video_name}_word_{int(word_info['start_frame'])}_{word_label}"
    )
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
        "preprocessed_for_training": True
    }

    metadata_file = word_dir / f"{filename}.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return True


def process_single_video(
    video_info: Dict[str, str],
    output_root: Path,
    target_frame_count: int,
    target_frame_height: int,
    target_frame_width: int,
) -> Dict[str, Any]:
    """Process one video and extract all non-sil word segments"""
    video_path = Path(video_info["video_path"])
    alignment_path = Path(video_info["alignment_path"])
    base_name = video_info["base_name"]
    speaker_id = video_info["speaker_id"]

    try:
        word_alignments = parse_grid_alignment_file(alignment_path)

        words_extracted = 0
        words_failed = 0

        for word_info in word_alignments:
            if str(word_info["word"]).lower() == "sil":
                continue

            frames = extract_word_frames(str(video_path), word_info)
            if frames is not None and len(frames) > 0:
                frames = standardize_word_frames(
                    frames=frames,
                    target_frame_count=target_frame_count,
                    target_frame_height=target_frame_height,
                    target_frame_width=target_frame_width,
                )
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

        total_words = sum(
            1 for w in word_alignments if str(w["word"]).lower() != "sil"
        )

        return {
            "video": base_name,
            "speaker_id": speaker_id,
            "status": "success",
            "words_extracted": words_extracted,
            "words_failed": words_failed,
            "total_words": total_words,
        }

    except Exception as exc:
        return {
            "video": base_name,
            "speaker_id": speaker_id,
            "status": "failed",
            "error": str(exc),
        }


def collect_word_statistics(output_root: Path) -> Dict[str, Any]:
    """Compute and write summary statistics for extracted word segments."""
    print("\nCollecting word statistics...")

    word_stats: Dict[str, Any] = {}
    total_samples = 0

    word_dirs = [
        d
        for d in output_root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ]

    for word_dir in word_dirs:
        word_label = word_dir.name
        npy_files = list(word_dir.glob("*.npy"))
        json_files = list(word_dir.glob("*.json"))

        durations: List[float] = []
        frame_counts: List[int] = []
        speakers = set()

        for json_file in json_files:
            with open(json_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            durations.append(float(metadata.get("duration", 0.0)))
            frame_counts.append(int(metadata.get("num_frames", 0)))
            speakers.add(metadata.get("speaker_id", "unknown"))

        word_stats[word_label] = {
            "count": len(npy_files),
            "avg_duration": float(np.mean(durations)) if durations else 0.0,
            "min_duration": float(np.min(durations)) if durations else 0.0,
            "max_duration": float(np.max(durations)) if durations else 0.0,
            "avg_frames": float(np.mean(frame_counts))
            if frame_counts else 0.0,
            "min_frames": int(np.min(frame_counts)) if frame_counts else 0,
            "max_frames": int(np.max(frame_counts)) if frame_counts else 0,
            "num_speakers": len(speakers)}

        total_samples += len(npy_files)

    stats = {
        "total_words": len(word_stats),
        "total_samples": total_samples,
        "word_statistics": word_stats,
    }

    stats_file = output_root / "word_statistics.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 60)
    print("Word Statistics")
    print("=" * 60)
    print(f"Total unique words: {stats['total_words']}")
    print(f"Total samples: {stats['total_samples']}")
    print("\nPer-word breakdown:")
    print(f"{'Word':<15} {'Count':<10} {'Avg Duration':<15} "
          f"{'Avg Frames':<15}")
    print("-" * 60)

    for word, stat in sorted(
        word_stats.items(),
        key=lambda x: x[1]["count"],
        reverse=True,
    ):
        print(
            f"{word:<15} {stat['count']:<10} "
            f"{stat['avg_duration']:<15.3f} {stat['avg_frames']:<15.1f}"
        )

    print(f"\nStatistics saved to: {stats_file}")
    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    """Create and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Word-level preprocessing pipeline for GRID dataset",
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
    parser.add_argument(
        "--target-frame-count",
        type=int,
        default=DEFAULT_TARGET_FRAME_COUNT,
        help="Number of frames to save per word clip",
    )
    parser.add_argument(
        "--target-frame-height",
        type=int,
        default=DEFAULT_TARGET_FRAME_HEIGHT,
        help="Saved frame height for word clips",
    )
    parser.add_argument(
        "--target-frame-width",
        type=int,
        default=DEFAULT_TARGET_FRAME_WIDTH,
        help="Saved frame width for word clips",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    output_root = BASE_DIR / args.output
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GRID Word-Level Preprocessing Pipeline")
    print("=" * 60)
    print(f"Output directory: {output_root}")
    print(
        "Target saved clip shape: "
        f"({args.target_frame_count}, "
        f"{args.target_frame_height}, {args.target_frame_width}, 3)"
    )
    if args.limit:
        print(f"Processing limit: {args.limit} videos")
    print("=" * 60 + "\n")
    print("Step 1: Finding GRID videos...")
    video_pairs = find_all_grid_videos()

    if args.limit:
        video_pairs = video_pairs[: args.limit]

    print(f"Found {len(video_pairs)} video-alignment pairs\n")

    print("Step 2: Extracting word segments from videos...")
    results: List[Dict[str, Any]] = []
    total_words = 0
    total_failed = 0

    for video_info in tqdm(video_pairs, desc="Processing videos"):
        result = process_single_video(
            video_info,
            output_root,
            target_frame_count=args.target_frame_count,
            target_frame_height=args.target_frame_height,
            target_frame_width=args.target_frame_width
        )
        results.append(result)

        if result["status"] == "success":
            total_words += int(result["words_extracted"])
            total_failed += int(result["words_failed"])

    summary = {
        "total_videos_processed": len(results),
        "successful_videos": sum(1 for r in results
                                 if r["status"] == "success"),
        "failed_videos": sum(1 for r in results if r["status"] == "failed"),
        "total_words_extracted": total_words,
        "total_words_failed": total_failed,
        "results": results}

    summary_file = output_root / "extraction_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\n" + "=" * 60)
    print("Extraction Summary")
    print("=" * 60)
    print(f"Total videos processed: {summary['total_videos_processed']}")
    print(f"Successful: {summary['successful_videos']}")
    print(f"Failed: {summary['failed_videos']}")
    print(f"Total words extracted: {total_words}")
    print(f"Total words failed: {total_failed}")
    print(f"Summary saved to: {summary_file}")
    collect_word_statistics(output_root)
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"Word segments saved to: {output_root}")
    print("Ready for word-level training!")


if __name__ == "__main__":
    main()
