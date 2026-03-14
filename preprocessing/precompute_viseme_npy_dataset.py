"""
Script to precompute viseme segment frames into .npy files for faster training.
It converts JSON-only viseme segment datasets (MFA timestamps)
into paired .json + .npy samples.
"""

from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple
import cv2
import numpy as np
from tqdm import tqdm


def _safe_load_json(path: Path) -> Optional[Dict]:
    try:
        with path.open("r", encoding="utf-8") as file_handle:
            return json.load(file_handle)
    except Exception:
        return None


def _resolve_video_path(
    sample_json_path: Path,
    metadata: Dict,
    interval_root: Path,
    cache: Dict[str, Optional[Path]],
) -> Optional[Path]:
    embedded_video = metadata.get("video")
    if isinstance(embedded_video, str):
        embedded_path = Path(embedded_video)
        if embedded_path.exists():
            return embedded_path

    try:
        base_video_id = sample_json_path.stem.rsplit("_", 2)[0]
    except Exception:
        return None

    if base_video_id in cache:
        return cache[base_video_id]

    interval_json_path = (
        interval_root / f"{base_video_id}_phoneme_intervals_mfa.json"
    )
    if not interval_json_path.exists():
        cache[base_video_id] = None
        return None

    interval_json = _safe_load_json(interval_json_path)
    if interval_json is None:
        cache[base_video_id] = None
        return None

    video_path_str = interval_json.get("video")
    if not isinstance(video_path_str, str):
        cache[base_video_id] = None
        return None

    video_path = Path(video_path_str)
    if not video_path.exists():
        cache[base_video_id] = None
        return None

    cache[base_video_id] = video_path
    return video_path


def _extract_segment_frames(
    video_path: Path,
    start_time_seconds: float,
    end_time_seconds: float,
) -> Optional[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 25.0

    start_frame = max(0, int(start_time_seconds * fps))
    end_frame = max(start_frame, int(end_time_seconds * fps))

    frames = []
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for _ in range(start_frame, end_frame + 1):
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)

    capture.release()

    if not frames:
        return None

    return np.array(frames)


def _collect_json_files(input_root: Path) -> list[Path]:
    json_files = []
    for viseme_dir in sorted(input_root.iterdir()):
        if not viseme_dir.is_dir():
            continue
        json_files.extend(sorted(viseme_dir.glob("*.json")))
    return json_files


def precompute_dataset(
    input_root: Path,
    interval_root: Path,
    output_root: Path,
    overwrite: bool,
    max_samples: Optional[int],
) -> Tuple[int, int, int, int]:
    json_files = _collect_json_files(input_root)
    if max_samples is not None:
        json_files = json_files[:max_samples]

    output_root.mkdir(parents=True, exist_ok=True)
    video_path_cache: Dict[str, Optional[Path]] = {}

    processed_count = 0
    skipped_existing_count = 0
    skipped_invalid_count = 0
    failed_count = 0

    for json_path in tqdm(json_files, desc="Precomputing viseme .npy files"):
        rel_parent = json_path.parent.relative_to(input_root)
        output_viseme_dir = output_root / rel_parent
        output_viseme_dir.mkdir(parents=True, exist_ok=True)

        npy_out_path = output_viseme_dir / f"{json_path.stem}.npy"
        json_out_path = output_viseme_dir / json_path.name

        if npy_out_path.exists() and not overwrite:
            if not json_out_path.exists():
                shutil.copy2(json_path, json_out_path)
            skipped_existing_count += 1
            continue

        metadata = _safe_load_json(json_path)
        if metadata is None:
            skipped_invalid_count += 1
            continue

        start_time = metadata.get("start_time")
        end_time = metadata.get("end_time")
        if start_time is None or end_time is None:
            skipped_invalid_count += 1
            continue

        video_path = _resolve_video_path(
            sample_json_path=json_path,
            metadata=metadata,
            interval_root=interval_root,
            cache=video_path_cache,
        )
        if video_path is None:
            skipped_invalid_count += 1
            continue

        frames = _extract_segment_frames(
            video_path=video_path,
            start_time_seconds=float(start_time),
            end_time_seconds=float(end_time),
        )
        if frames is None or frames.size == 0:
            failed_count += 1
            continue

        np.save(str(npy_out_path), frames)
        shutil.copy2(json_path, json_out_path)
        processed_count += 1

    return (
        processed_count,
        skipped_existing_count,
        skipped_invalid_count,
        failed_count,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute viseme .npy dataset from JSON segments"
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/processed/visemes_bozkurt_mfa"),
        help="Input viseme root containing class folders with JSON files",
    )
    parser.add_argument(
        "--interval-root",
        type=Path,
        default=Path("data/processed/phonemes_mfa"),
        help="Root folder containing *_phoneme_intervals_mfa.json files",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/visemes_bozkurt_mfa_npy"),
        help="Output root for paired .json + .npy viseme samples",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .npy files",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit for quick test runs",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("PRECOMPUTE VISEME NPY DATASET")
    print("=" * 80)
    print(f"Input root:    {args.input_root}")
    print(f"Interval root: {args.interval_root}")
    print(f"Output root:   {args.output_root}")

    if not args.input_root.exists():
        raise FileNotFoundError(f"Input root not found: {args.input_root}")
    if not args.interval_root.exists():
        raise FileNotFoundError(
            f"Interval root not found: {args.interval_root}"
        )

    (
        processed_count,
        skipped_existing_count,
        skipped_invalid_count,
        failed_count,
    ) = precompute_dataset(
        input_root=args.input_root,
        interval_root=args.interval_root,
        output_root=args.output_root,
        overwrite=args.overwrite,
        max_samples=args.max_samples,
    )

    print("\nDone")
    print(f"  Processed:        {processed_count}")
    print(f"  Skipped existing: {skipped_existing_count}")
    print(f"  Skipped invalid:  {skipped_invalid_count}")
    print(f"  Failed decode:    {failed_count}")


if __name__ == "__main__":
    main()
