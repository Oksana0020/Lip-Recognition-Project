"""
Complete MFA-based preprocessing pipeline for GRID dataset:
1. Extract phoneme intervals from all GRID videos using MFA
   (Montreal Forced Aligner)
2. Reorganize phoneme intervals into phoneme-labelled folders
3. Map phonemes to visemes using Bozkurt mapping
4. Save structured outputs to data/processed/ with _mfa suffix
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Set

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
                    output_root
                    / f"{base_name}_phoneme_intervals_mfa.json"
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


def step_2_reorganize_phonemes(input_root: Path, output_root: Path) -> bool:
    """Reorganize interval JSONs into phoneme-labelled folders."""
    _print_header("STEP 2: Reorganizing phoneme segments by phoneme label")

    output_root.mkdir(parents=True, exist_ok=True)
    json_files = list(input_root.glob("*_phoneme_intervals_mfa.json"))
    print(f"Processing {len(json_files)} MFA phoneme interval files")

    phoneme_count: Dict[str, int] = {}

    for json_file in json_files:
        try:
            with json_file.open("r", encoding="utf-8") as file_handle:
                data = json.load(file_handle)

            intervals = data.get("phoneme_intervals", [])
            for entry in intervals:
                phoneme = str(entry["phoneme"]).upper()

                if phoneme in {"SIL", "SP", "SPN"}:
                    continue

                phoneme_count[phoneme] = phoneme_count.get(phoneme, 0) + 1

                phoneme_dir = output_root / phoneme
                phoneme_dir.mkdir(parents=True, exist_ok=True)

                start_frame = entry.get("start_frame", 0)
                base_id = json_file.stem.replace("_phoneme_intervals_mfa", "")
                out_name = f"{base_id}_{start_frame}_{phoneme}.json"
                out_file = phoneme_dir / out_name

                with out_file.open("w", encoding="utf-8") as out_handle:
                    json.dump(entry, out_handle, indent=2)

        except Exception as exc:
            print(f"  ERROR processing {json_file.name}: {exc}")

    print("\nReorganization complete:")
    print(f"  Unique phonemes: {len(phoneme_count)}")
    print(f"  Total segments: {sum(phoneme_count.values())}")
    print(f"  Saved to: {output_root}")

    print("\n  Top 10 phonemes by frequency:")
    sorted_phonemes = sorted(
        phoneme_count.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    for phoneme, count in sorted_phonemes[:10]:
        print(f"    {phoneme}: {count}")

    return len(phoneme_count) > 0


def load_viseme_map(mapping_file: Path) -> Dict[str, str]:
    """Load Bozkurt viseme mapping from CSV (phoneme -> viseme)."""
    viseme_map: Dict[str, str] = {}

    with mapping_file.open("r", encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        for row in reader:
            viseme = (row.get("viseme_class") or "").strip()
            phonemes_str = (row.get("phonemes") or "").strip()
            if not viseme or not phonemes_str:
                continue

            phonemes = [
                p.strip().upper()
                for p in phonemes_str.split(",")
                if p.strip()
            ]
            for phoneme in phonemes:
                viseme_map[phoneme] = viseme

    return viseme_map


def step_3_map_to_visemes(
    phoneme_root: Path,
    mapping_file: Path,
    out_root: Path,
    system_name: str,
) -> bool:
    """Map phoneme-labelled folders into viseme class folders."""
    _print_header(
        f"STEP 3: Mapping phonemes to visemes ({system_name} system)"
    )

    viseme_map = load_viseme_map(mapping_file)
    viseme_classes = len(set(viseme_map.values()))
    print(f"Loaded mapping with {viseme_classes} viseme classes")

    out_root.mkdir(parents=True, exist_ok=True)

    viseme_counts: Dict[str, int] = {}
    unmapped_phonemes: Set[str] = set()

    for phoneme_dir in phoneme_root.iterdir():
        if not phoneme_dir.is_dir():
            continue

        phoneme_label = phoneme_dir.name.upper()
        viseme = viseme_map.get(phoneme_label)

        if not viseme:
            unmapped_phonemes.add(phoneme_label)
            continue

        viseme_dir = out_root / viseme
        viseme_dir.mkdir(parents=True, exist_ok=True)

        segment_count = 0
        for seg_file in phoneme_dir.iterdir():
            if seg_file.is_file():
                shutil.copy(seg_file, viseme_dir / seg_file.name)
                segment_count += 1

        viseme_counts[viseme] = viseme_counts.get(viseme, 0) + segment_count

    print(f"\nViseme mapping complete ({system_name}):")
    print(f"  Viseme classes: {len(viseme_counts)}")
    print(f"  Total segments: {sum(viseme_counts.values())}")
    print(f"  Unmapped phonemes: {len(unmapped_phonemes)}")

    if unmapped_phonemes:
        print(f"  Unmapped: {sorted(unmapped_phonemes)}")

    print(f"  Saved to: {out_root}")

    print("\n  Viseme distribution:")
    sorted_visemes = sorted(
        viseme_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    for viseme, count in sorted_visemes:
        print(f"    {viseme}: {count}")

    return len(viseme_counts) > 0


def verify_outputs(processed_root: Path) -> None:
    """Verify the structure and contents of processed outputs."""
    _print_header("VERIFICATION: Checking output structure")

    checks = {
        "phonemes_mfa (raw intervals)": processed_root / "phonemes_mfa",
        "phonemes_by_label_mfa": processed_root / "phonemes_by_label_mfa",
        "visemes_bozkurt_mfa": processed_root / "visemes_bozkurt_mfa",
    }

    for name, path in checks.items():
        if path.exists() and path.is_dir():
            subdirs = [d for d in path.iterdir() if d.is_dir()]
            files = list(path.glob("*.json"))
            if subdirs:
                total_files = sum(
                    1 for d in subdirs for f in d.iterdir() if f.is_file()
                )
                print(
                    f"✓ {name}: {len(subdirs)} categories, "
                    f"{total_files} files"
                )
            else:
                print(f"✓ {name}: {len(files)} files")
        else:
            print(f"✗ {name}: NOT FOUND")

    _print_header("MFA PREPROCESSING PIPELINE COMPLETE")
    print("Data is ready to train models on viseme-level using:")
    print("  - Bozkurt system (MFA): data/processed/visemes_bozkurt_mfa/")


def main() -> None:
    """Run the complete MFA preprocessing pipeline."""
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
    print("  2. Phoneme-based segments")
    print("  3. Viseme-based segments (Bozkurt system)")
    print("\nAll outputs will be saved to: data/processed/ with _mfa suffix")

    if args.speakers:
        print(f"\nProcessing speakers: {', '.join(args.speakers)}")
    else:
        print("\nProcessing all speakers")

    if args.start_from:
        print(f"Starting from video: {args.start_from}")

    processed_root = ROOT_DIR / "data" / "processed"
    phonemes_raw_mfa = processed_root / "phonemes_mfa"
    phonemes_by_label_mfa = processed_root / "phonemes_by_label_mfa"
    visemes_bozkurt_mfa = processed_root / "visemes_bozkurt_mfa"

    bozkurt_map = ROOT_DIR / "mapping" / "bozkurt_viseme_map.csv"
    if not bozkurt_map.exists():
        print(f"\nERROR: Bozkurt mapping file not found: {bozkurt_map}")
        return

    try:
        success = step_1_extract_phonemes_mfa(
            phonemes_raw_mfa,
            args.speakers,
            args.start_from,
        )
        if not success:
            print("\nERROR: MFA phoneme extraction failed. Aborting pipeline.")
            return

        success = step_2_reorganize_phonemes(
            phonemes_raw_mfa,
            phonemes_by_label_mfa,
        )
        if not success:
            print("\nERROR: Phoneme reorganization failed. Aborting pipeline.")
            return

        success = step_3_map_to_visemes(
            phonemes_by_label_mfa,
            bozkurt_map,
            visemes_bozkurt_mfa,
            "Bozkurt (MFA)",
        )
        if not success:
            print("\nWARNING: Bozkurt viseme mapping produced no outputs.")

        verify_outputs(processed_root)

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
