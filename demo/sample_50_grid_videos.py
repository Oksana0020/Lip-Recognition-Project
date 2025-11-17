"""
Script randomly selects 50 GRID video–alignment pairs for evaluation.
This implements the “clinical trial” sampling method:
- Scans: data/grid/GRID dataset full/s1 (videos)
         data/grid/GRID dataset full/alignments/s1 (alignments)
- Only keeps pairs that exist in both folders.
- Randomly picks 50 pairs.
- Saves the sample to sample_50_grid_pairs.json in the demo folder.
"""

import os
import json
import random
from pathlib import Path

def find_grid_videos_and_alignments() -> list:
    """Return all matching video + alignment pairs from speaker s1."""
    root = Path(__file__).parent.parent / "data" / "grid" / "GRID dataset full"
    videos_path = root / "s1"
    alignments_path = root / "alignments" / "s1"

    if not videos_path.exists():
        raise FileNotFoundError(f"Video folder not found: {videos_path}")
    if not alignments_path.exists():
        raise FileNotFoundError(f"Alignment folder not found: {alignments_path}")

    video_files = [f for f in os.listdir(videos_path) if f.endswith(".mpg")]

    pairs = []
    for video_file in video_files:
        base_name = os.path.splitext(video_file)[0]
        align_file = base_name + ".align"

        video_path = videos_path / video_file
        align_path = alignments_path / align_file

        if align_path.exists():
            pairs.append(
                {
                    "base_name": base_name,
                    "video_path": str(video_path),
                    "alignment_path": str(align_path),
                }
            )

    return pairs


def sample_50_pairs(seed: int = 42) -> list:
    """Randomly sample 50 video–alignment pairs (or fewer if dataset too small)."""
    all_pairs = find_grid_videos_and_alignments()
    total = len(all_pairs)

    if total == 0:
        raise RuntimeError("No video–alignment pairs found in GRID dataset.")

    n = min(50, total)

    random.seed(seed)
    return random.sample(all_pairs, n)


def main():
    print("=" * 70)
    print("GRID VIDEO SAMPLING. SELECTING 50 VIDEO–ALIGNMENT PAIRS FOR EVALUATION")
    print("=" * 70)

    samples = sample_50_pairs(seed=42)

    print(f"Total pairs selected: {len(samples)}\n")
    print(f"{'#':<3} | {'Base name':<12} | {'Video file':<20} | {'Alignment file'}")
    print("-" * 70)

    for i, item in enumerate(samples, 1):
        video = Path(item["video_path"]).name
        align = Path(item["alignment_path"]).name
        print(f"{i:<3} | {item['base_name']:<12} | {video:<20} | {align}")

    output_path = Path(__file__).parent / "sample_50_grid_pairs.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_samples": len(samples),
                "seed": 42,
                "pairs": samples,
            },
            f,
            indent=2,
        )

    print(f"\n✅ Sample saved to {output_path}")


if __name__ == "__main__":
    main()
