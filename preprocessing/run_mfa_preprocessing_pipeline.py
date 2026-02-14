"""
Complete MFA-based preprocessing pipeline for GRID dataset:
1. Extract phoneme intervals from all GRID videos using MFA
   (Montreal Forced Aligner)
2. Reorganize phoneme intervals into phoneme-labelled folders
3. Save structured outputs to data/processed/ with _mfa suffix
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent

CONDA_EXE = "C:/Users/oksan/miniconda3/Scripts/conda.exe"
DEFAULT_MODEL = "english_mfa"


def _load_mfa_extractor():
    """Load MFA extraction helper from demo folder at runtime."""
    demo_path = ROOT_DIR / "demo"
    if str(demo_path) not in sys.path:
        sys.path.insert(0, str(demo_path))

    import importlib

    module = importlib.import_module("mfa_phoneme_frame_extraction")
    return module.main


def _print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def ensure_mfa_model(model_name: str = DEFAULT_MODEL) -> bool:
    """Ensure MFA acoustic model is downloaded."""
    check_cmd = [
        CONDA_EXE,
        "run",
        "-n",
        "base",
        "mfa",
        "model",
        "list",
        "acoustic",
    ]
    result = subprocess.run(
        check_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if model_name in result.stdout:
        print(f"MFA model '{model_name}' is already downloaded")
        return True

    print(f"Downloading MFA acoustic model: {model_name}")
    download_cmd = [
        CONDA_EXE,
        "run",
        "-n",
        "base",
        "mfa",
        "model",
        "download",
        "acoustic",
        model_name,
    ]
    proc = subprocess.run(
        download_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="ignore")
        print(f"ERROR downloading model:\n{err}")
        return False

    print(f"Model {model_name} downloaded successfully")
    return True


def find_grid_videos_and_alignments(
    speakers: Optional[List[str]] = None,
) -> List[Dict[str, object]]:
    """Find all GRID video+alignment pairs for specified speakers."""
    grid_root = ROOT_DIR / "data" / "grid" / "GRID dataset full"
    if not grid_root.exists():
        print(f"ERROR: GRID dataset not found at {grid_root}")
        return []

    pairs: List[Dict[str, object]] = []

    if speakers:
        speaker_dirs = [
            grid_root / s for s in speakers if (grid_root / s).exists()
        ]
    else:
        speaker_dirs = [
            d
            for d in grid_root.iterdir()
            if d.is_dir() and d.name.startswith("s")
        ]

    for speaker_dir in sorted(speaker_dirs):
        speaker_name = speaker_dir.name
        alignments_dir = grid_root / "alignments" / speaker_name

        if not alignments_dir.exists():
            print(f"WARNING: No alignments found for {speaker_name}")
            continue

        for video_file in speaker_dir.glob("*.mpg"):
            align_file = alignments_dir / f"{video_file.stem}.align"
            if not align_file.exists():
                continue

            pairs.append(
                {
                    "video_path": video_file,
                    "alignment_path": align_file,
                    "base_name": video_file.stem,
                    "speaker": speaker_name,
                }
            )

    return pairs


def step_1_extract_phonemes_mfa(
    output_root: Path,
    speakers: Optional[List[str]] = None,
    start_from: Optional[str] = None,
) -> bool:
    """Extract phoneme intervals from all GRID videos using MFA."""
    _print_header("STEP 1: Extracting phoneme intervals using MFA")

    print("\nChecking MFA acoustic model")
    if not ensure_mfa_model(DEFAULT_MODEL):
        print("ERROR: Failed to download MFA acoustic model")
        return False

    output_root.mkdir(parents=True, exist_ok=True)
    pairs = find_grid_videos_and_alignments(speakers)

    if not pairs:
        print("ERROR: No GRID video+alignment pairs found!")
        print("Expected dataset location: data/grid/GRID dataset full/")
        return False

    print(f"Found {len(pairs)} video+alignment pairs")

    speakers_found = {str(p["speaker"]) for p in pairs}
    print(f"Speakers: {', '.join(sorted(speakers_found))}")

    if start_from:
        start_idx = None
        for idx, pair in enumerate(pairs):
            if pair["base_name"] == start_from:
                start_idx = idx
                break

        if start_idx is not None:
            pairs = pairs[start_idx:]
            print(f"\n*** Starting from video: {start_from} ***")
            print(f"Processing {len(pairs)} videos from this point onwards")
        else:
            print(f"\nWARNING: Start video '{start_from}' not found!")
            print("Processing all videos instead")

    success_count = 0
    error_count = 0
    mfa_extract_phonemes = _load_mfa_extractor()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        for i, pair in enumerate(pairs, 1):
            video_path = pair["video_path"]
            alignment_path = pair["alignment_path"]
            base_name = str(pair["base_name"])

            try:
                result = mfa_extract_phonemes(
                    video_path=video_path,
                    align_path=alignment_path,
                    tmpdir=tmp_path / base_name,
                    keep_temp=False,
                    out_root=None,
                    skip_model_check=True,
                )

                out_file = (
                    output_root / f"{base_name}_phoneme_intervals_mfa.json"
                )
                with out_file.open("w", encoding="utf-8") as file_handle:
                    json.dump(result, file_handle, indent=2)

                success_count += 1

                if i % 10 == 0:
                    print(
                        f"  Processed {i}/{len(pairs)} videos... "
                        f"({success_count} success, {error_count} errors)"
                    )

            except Exception as exc:
                print(f"  ERROR processing {base_name}: {exc}")
                error_count += 1
                if error_count <= 5:
                    traceback.print_exc()

    print("\nMFA phoneme extraction complete:")
    print(f"  ✓ Success: {success_count}")
    print(f"  ✗ Errors: {error_count}")
    print(f"  Saved to: {output_root}")

    return success_count > 0


def main() -> None:
    """Run MFA preprocessing pipeline (phoneme extraction only)."""
    parser = argparse.ArgumentParser(
        description="Run MFA preprocessing pipeline on GRID dataset"
    )
    parser.add_argument(
        "--speakers",
        nargs="+",
        default=None,
        help=(
            "List of speakers to process (e.g., s1 s2 s3). "
            "If not specified, all speakers will be processed."
        ),
    )
    parser.add_argument(
        "--start-from",
        type=str,
        default=None,
        help=(
            "Start processing from a specific video ID (e.g., lwwf8a). "
            "Videos before this will be skipped."
        ),
    )
    args = parser.parse_args()

    start_time = time.time()

    _print_header("GRID DATASET MFA PREPROCESSING PIPELINE")
    print("\nThis will process the GRID dataset using MFA and create:")
    print("  1. Phoneme intervals (MFA method)")
    print("\nAll outputs will be saved to: data/processed/ with _mfa suffix")

    if args.speakers:
        print(f"\nProcessing speakers: {', '.join(args.speakers)}")
    else:
        print("\nProcessing all speakers")

    if args.start_from:
        print(f"Starting from video: {args.start_from}")

    processed_root = ROOT_DIR / "data" / "processed"
    phonemes_raw_mfa = processed_root / "phonemes_mfa"

    try:
        ok = step_1_extract_phonemes_mfa(
            phonemes_raw_mfa,
            args.speakers,
            args.start_from,
        )
        if not ok:
            print("\nERROR: MFA phoneme extraction failed.")
            return

        elapsed = time.time() - start_time
        print(f"\nTotal processing time: {elapsed / 60:.1f} minutes")
        print(f"Total processing time: {elapsed / 3600:.2f} hours")

    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
    except Exception as exc:
        print(f"\n\nERROR: Pipeline failed with exception: {exc}")
        traceback.print_exc()


if __name__ == "__main__":
    main()

#commit 1
#Add MFA GRID phoneme interval extraction 
