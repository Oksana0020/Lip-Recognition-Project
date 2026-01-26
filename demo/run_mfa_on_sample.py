"""
Batch runner for MFA-based phoneme extraction on a sample of 50 GRID videos.

The script reads video + .align pairs from `sample_50_grid_pairs.json` and runs
`demo/mfa_phoneme_frame_extraction.py` for each sample. It collects returned
phoneme timing output into one JSON file and also saves per-video frame images
for manual checking.
"""

import argparse
import json
from pathlib import Path

from mfa_phoneme_frame_extraction import (
    main as mfa_phoneme_frame_extraction,
)


SCRIPT_DIR = Path(__file__).parent
SAMPLE_PAIRS_PATH = SCRIPT_DIR / "sample_50_grid_pairs.json"
BATCH_OUTPUT_PATH = SCRIPT_DIR / "batch_mfa_results.json"
DEFAULT_FRAMES_ROOT = SCRIPT_DIR / "phoneme_frames_mfa_50"


def resolve_video_path(video_path: Path) -> Path:
    if video_path.exists():
        return video_path

    data_root = SCRIPT_DIR.parent / "data" / "grid" / "GRID dataset full"
    alt_path = data_root / "s1" / video_path.name
    return alt_path if alt_path.exists() else video_path


def resolve_alignment_path(alignment_path: Path, video_path: Path) -> Path:
    if alignment_path.exists():
        return alignment_path

    data_root = SCRIPT_DIR.parent / "data" / "grid" / "GRID dataset full"
    alt_dir = data_root / "alignments" / "s1"
    alt_path = alt_dir / f"{video_path.stem}.align"
    return alt_path if alt_path.exists() else alignment_path


def main(
    limit: int | None = None,
    outdir: str | None = None,
    batch_output_file: str | None = None,
) -> None:
    if not SAMPLE_PAIRS_PATH.exists():
        msg = (
            f"Sample file not found: {SAMPLE_PAIRS_PATH}. "
            "Run sample_50_grid_videos.py first."
        )
        raise FileNotFoundError(msg)

    with SAMPLE_PAIRS_PATH.open("r", encoding="utf-8") as sample_file:
        sample_pairs_data = json.load(sample_file)

    video_alignment_pairs = sample_pairs_data.get("pairs", [])
    if not video_alignment_pairs:
        print("No pairs in sample_50_grid_pairs.json")
        return

    total = len(video_alignment_pairs)
    print(f"Running MFA extraction on {total} samples\n")

    frames_output_root = Path(outdir) if outdir else DEFAULT_FRAMES_ROOT
    frames_output_root.mkdir(parents=True, exist_ok=True)

    batch_results: list[dict] = []

    for idx, pair in enumerate(video_alignment_pairs, start=1):
        if limit is not None and idx > limit:
            break

        video_path = resolve_video_path(Path(pair["video_path"]))
        alignment_path = resolve_alignment_path(
            Path(pair["alignment_path"]),
            video_path,
        )

        base_name = pair.get("base_name", video_path.stem)
        frames_dir = frames_output_root / base_name
        frames_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{idx}/{total}] {video_path.name}")

        try:
            result = mfa_phoneme_frame_extraction(
                video_path,
                alignment_path,
                None,
                False,
                frames_output_root,
            )
        except Exception as exc:
            print(f"ERROR processing {video_path.name}: {exc}")
            continue

        result["base_name"] = base_name
        result["frames_dir"] = str(frames_dir)
        batch_results.append(result)

    batch_output = {
        "num_samples": len(batch_results),
        "method": "mfa_or_internal_fallback",
        "frames_root": str(frames_output_root),
        "results": batch_results,
    }

    batch_output_path = (
        Path(batch_output_file)
        if batch_output_file
        else BATCH_OUTPUT_PATH
    )
    batch_output_path.parent.mkdir(parents=True, exist_ok=True)

    with batch_output_path.open("w", encoding="utf-8") as output_file:
        json.dump(batch_output, output_file, indent=2)

    print(f"\nSaved batch results to: {batch_output_path}")
    print(f"Frames saved: {frames_output_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to process.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Output root for frames overrides default.",
    )
    parser.add_argument(
        "--batch-output",
        type=str,
        default=None,
        help="Output path for batch results JSON overrides default.",
    )
    args = parser.parse_args()

    main(
        limit=args.limit,
        outdir=args.outdir,
        batch_output_file=args.batch_output,
    )
