"""
Organize phoneme-labeled frame images into viseme class folders using the
Bozkurt viseme mapping system.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Dict, Iterator, Optional, Set


def load_viseme_mapping(mapping_file: Path) -> Dict[str, str]:
    phoneme_to_viseme: Dict[str, str] = {}

    with mapping_file.open("r", encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)

        for mapping_row in reader:
            viseme_class = (mapping_row.get("viseme_class") or "").strip()
            phoneme_list_text = (mapping_row.get("phonemes") or "").strip()

            if not viseme_class or not phoneme_list_text:
                continue

            phoneme_labels = [
                phoneme.strip().upper()
                for phoneme in phoneme_list_text.split(",")
                if phoneme.strip()
            ]

            for phoneme_label in phoneme_labels:
                phoneme_to_viseme[phoneme_label] = viseme_class

    return phoneme_to_viseme


# Yield frame image files from single video directory
def iter_frame_files(video_dir: Path) -> Iterator[Path]:
    yield from sorted(video_dir.glob("*.jpg"))
    yield from sorted(video_dir.glob("*.png"))


def parse_phoneme_from_frame_name(frame_path: Path) -> Optional[str]:
    name_tokens = frame_path.stem.split("_")
    if len(name_tokens) < 4:
        return None

    phoneme_label = name_tokens[2].strip().upper()
    return phoneme_label or None


def organize_frames_by_viseme(
    frames_root: Path,
    phoneme_to_viseme: Dict[str, str],
    output_root: Path,
) -> None:
    if not frames_root.exists():
        print(f"ERROR: Frames directory not found: {frames_root}")
        return

    output_root.mkdir(parents=True, exist_ok=True)

    for viseme_class in sorted(set(phoneme_to_viseme.values())):
        (output_root / viseme_class).mkdir(exist_ok=True)

    frames_per_viseme: Dict[str, int] = {}
    unmapped_phonemes: Set[str] = set()
    total_frames_seen = 0

    for video_dir in sorted(frames_root.iterdir()):
        if not video_dir.is_dir():
            continue

        for frame_path in iter_frame_files(video_dir):
            total_frames_seen += 1

            phoneme_label = parse_phoneme_from_frame_name(frame_path)
            if phoneme_label is None:
                continue

            viseme_class = phoneme_to_viseme.get(phoneme_label)
            if viseme_class is None:
                unmapped_phonemes.add(phoneme_label)
                continue

            destination_path = (output_root / viseme_class) / frame_path.name
            shutil.copy2(frame_path, destination_path)

            frames_per_viseme[viseme_class] = (
                frames_per_viseme.get(viseme_class, 0) + 1
            )

    print("\n" + "=" * 80)
    print("PHONEME FRAMES ORGANIZED INTO VISEME CLASSES")
    print("=" * 80)
    print(f"\nSource: {frames_root}")
    print(f"Output: {output_root}")

    print("\nFrames by viseme class:")
    for viseme_class in sorted(frames_per_viseme):
        print(f"  {viseme_class}: {frames_per_viseme[viseme_class]} frames")

    total_frames_copied = sum(frames_per_viseme.values())
    print(f"\nTotal frames organized: {total_frames_copied}")
    print(f"Total frames processed: {total_frames_seen}")

    if unmapped_phonemes:
        print(f"\nUnmapped phonemes found: {sorted(unmapped_phonemes)}")

    print("=" * 80)


def main() -> None:
    """Run viseme-based frame organization using project default paths."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    mapping_file = script_dir / "bozkurt_viseme_map.csv"
    frames_root = project_root / "demo" / "phoneme_frames_mfa_50"
    output_root = script_dir / "processed"

    print("=" * 80)
    print("ORGANIZE PHONEME FRAMES BY VISEME CLASS")
    print("=" * 80)

    if not mapping_file.exists():
        print(f"ERROR: Mapping file not found: {mapping_file}")
        return

    print(f"\nLoading Bozkurt viseme mapping from: {mapping_file}")
    phoneme_to_viseme = load_viseme_mapping(mapping_file)
    print(f"Loaded {len(phoneme_to_viseme)} phoneme-to-viseme mappings")

    if not frames_root.exists():
        print(f"ERROR: Frames directory not found: {frames_root}")
        print(
            "Please ensure phoneme frames exist in demo/phoneme_frames_mfa_50 "
            "before running this script."
        )
        return

    print(f"Processing frames from: {frames_root}")
    organize_frames_by_viseme(frames_root, phoneme_to_viseme, output_root)


if __name__ == "__main__":
    main()
