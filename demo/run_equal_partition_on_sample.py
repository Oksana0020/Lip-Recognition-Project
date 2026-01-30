"""
Batch-run the equal-partition phoneme extraction pipeline on a sample of
50 GRID videos. The script ensures the file
'sample_50_grid_pairs.json' exists. If the file is missing, it will sample
50 video/alignment pairs using find_grid_videos_and_alignments() and save
them. For each sampled video the script runs
extract_phonemes_equal_partition(), saves the results, and calls
save_phoneme_frames.py as a subprocess to generate labelled frames.
All extraction results are collected into batch_equal_partition_results.json.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from extract_phonemes_equal_partition import (
    extract_phonemes_equal_partition,
    find_grid_videos_and_alignments,
)


HERE = Path(__file__).parent
SAMPLE_FILE = HERE / "sample_50_grid_pairs.json"
OUTPUT_FILE = HERE / "phoneme_extraction_equal_partition.json"
PHONEME_JSON_DIR = HERE / "phoneme_jsons"
PHONEME_FRAMES_DIR = HERE / "phoneme_frames"


def ensure_sample_file():
    """Create sample_50_grid_pairs.json if missing by sampling pairs."""
    if SAMPLE_FILE.exists():
        try:
            with open(SAMPLE_FILE, "r", encoding="utf-8") as f:
                json.load(f)
            return
        except Exception:
            print("Existing sample file is malformed — recreating it...")
            SAMPLE_FILE.unlink()

    print("Sample file not found — creating sample_50_grid_pairs.json now...")
    all_pairs = find_grid_videos_and_alignments()
    import random

    random.seed(42)
    n = min(50, len(all_pairs))
    samples = random.sample(all_pairs, n)

    # convert Path objects to strings for JSON serialization
    serializable = []
    for p in samples:
        serializable.append({
            "base_name": p.get("base_name"),
            "video_path": str(p.get("video_path")),
            "alignment_path": str(p.get("alignment_path")),
        })

    payload = {
        "num_samples": len(serializable),
        "seed": 42,
        "pairs": serializable,
    }
    with open(SAMPLE_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_save_frames_for_json(json_path: Path, frame_at: str = "center"):
    """
    Call demo/save_phoneme_frames.py as a subprocess.

    Save frames for one phoneme JSON file.
    """
    script = HERE / "save_phoneme_frames.py"
    cmd = [
        sys.executable,
        str(script),
        "--json",
        str(json_path),
        "--outdir",
        str(PHONEME_FRAMES_DIR.name),
        "--frame-at",
        frame_at,
    ]
    subprocess.check_call(cmd, cwd=str(HERE))


def main(limit: int = None, frame_at: str = "center"):
    ensure_sample_file()

    with open(SAMPLE_FILE, "r", encoding="utf-8") as f:
        sample_data = json.load(f)

    pairs = sample_data.get("pairs", [])
    if not pairs:
        print("No pairs in sample_50_grid_pairs.json")
        return

    PHONEME_JSON_DIR.mkdir(exist_ok=True)
    PHONEME_FRAMES_DIR.mkdir(exist_ok=True)

    print(f"Running EQUAL PARTITION extraction on {len(pairs)} samples...\n")

    all_results = []
    for i, pair in enumerate(pairs, 1):
        if limit is not None and i > limit:
            break

        video_path = Path(pair["video_path"])
        align_path = Path(pair["alignment_path"])
        base = pair.get("base_name", video_path.stem)

        print(f"[{i}/{len(pairs)}] {video_path.name}")
        try:
            result = extract_phonemes_equal_partition(video_path, align_path)
            result["base_name"] = base

            out_json = PHONEME_JSON_DIR / f"{base}_equal_partition.json"
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

            # call the frame saver script to produce labeled frames
            try:
                run_save_frames_for_json(out_json, frame_at=frame_at)
            except subprocess.CalledProcessError as e:
                print(f"Frame extraction failed for {base}: {e}")

            all_results.append(result)
        except Exception as e:
            print(f"Failed on {video_path.name}: {e}")

    out_payload = {
        "num_samples": len(all_results),
        "method": "equal_partition",
        "results": all_results,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)

    print("\n\N{WHITE HEAVY CHECK MARK} Saved batch results to")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to process (useful for smoke tests)",
    )
    parser.add_argument(
        "--frame-at",
        choices=["center", "start"],
        default="center",
        help="Where to capture frame within phoneme",
    )
    args = parser.parse_args()
    main(limit=args.limit, frame_at=args.frame_at)
