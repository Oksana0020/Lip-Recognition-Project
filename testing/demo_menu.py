"""
Main entry point for the project demonstration.
Provides an interactive menu to run various demos:
1. Viseme-level inference on GRID
2. Word-level inference on GRID
3. Live webcam inference
4. Inference on user-provided videos (my_words)
5. Inference on user-provided cropped videos (my_words_cropped_videos)
6. Inference on user-provided viseme videos (my_sounds)
7. Inference on user-provided cropped viseme videos (my_sounds_cropped_videos)

"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
PYTHON = sys.executable


def clear() -> None:
    print("\n" + "=" * 60)


def run(cmd: list[str]) -> None:
    """Run a command in a subprocess, streaming output live."""
    try:
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        print("\n  (interrupted)")


def run_my_words_demo() -> None:
    """Recognize words from videos in testing/my_words."""
    clear()
    print("   Demo 4— Recognize My Word Videos")
    print("=" * 60)
    my_words_dir = PROJECT_ROOT / "testing" / "my_words"
    if not my_words_dir.exists():
        print(f"  Folder not found: {my_words_dir}")
        return
    videos = sorted([
        f for f in my_words_dir.iterdir()
        if f.suffix.lower() in {
            '.mp4', '.mov', '.avi', '.mkv', '.mpg', '.mpeg'
        }])
    if not videos:
        print("  No video files found in my_words.")
        return
    print("  Pick a video to recognize:")
    for i, vid in enumerate(videos, 1):
        print(f"  {i}. {vid.name}")
    print("   0. Back to main menu\n")
    while True:
        raw = input("  Enter number: ").strip()
        if raw == "0":
            return
        if raw.isdigit() and 1 <= int(raw) <= len(videos):
            vid_path = videos[int(raw) - 1]
            print(f"\n  Running inference on: {vid_path.name}\n")
            run([
                PYTHON,
                str(PROJECT_ROOT / "training" / "infer.py"),
                "word",
                str(vid_path),
                "--checkpoint",
                str(
                    PROJECT_ROOT
                    / "training/checkpoints_words/best_model_words.pth"
                ),
                "--top-k", "5"
            ])
            print()
            break
        print("  Invalid choice.")


def run_viseme_demo() -> None:
    """Launch the interactive viseme-level inference demo"""
    clear()
    print("   Demo 1 — Viseme-Level Inference")
    print("=" * 60)
    print(
        "  Pick a GRID phoneme clip from all 13 Bozkurt viseme classes.\n"
        "  The model predicts which viseme is shown, with confidence bars.\n"
        "  Try V8 (easiest, 99%) and V13 (hardest, 79%) for comparison.\n"
    )
    run([PYTHON, str(PROJECT_ROOT / "testing" / "viseme_demo.py")])


def run_word_demo() -> None:
    """Launch word-level inference on pre-selected GRID clip samples"""
    clear()
    print("   Demo 2 — Word-Level Inference on GRID Clips")
    print("=" * 60)
    # Preselected clips from high-accuracy word classes
    samples = [
        (
            "please  (99.6% class accuracy)",
            "data/processed/words_by_label/please/bbaczp_word_39750_please.npy"
        ),
        (
            "zero    (99.2% class accuracy)",
            "data/processed/words_by_label/zero/bbaczp_word_32750_zero.npy"
        ),
        (
            "four    (99.4% class accuracy)",
            "data/processed/words_by_label/four/bbae4n_word_33500_four.npy"
        ),
        (
            "blue    (98.6% class accuracy)",
            "data/processed/words_by_label/blue/bbab8n_word_16500_blue.npy"
        ),
        (
            "green   (98.2% class accuracy)",
            "data/processed/words_by_label/green/bgaa1a_word_25250_green.npy"
        ),
    ]
    available = [
        (label, path)
        for label, path in samples
        if (PROJECT_ROOT / path).exists()]
    if not available:
        print("  No sample clips found. Check data/processed/words_by_label/")
        return
    print("  Pick a GRID word clip to run through the word-level model:\n")
    print("  " + "─" * 60)
    for i, (label, _) in enumerate(available, 1):
        print(f"  {i}. {label}")
    print("  " + "─" * 60)
    print("   0. Back to main menu\n")
    while True:
        raw = input("  Enter number: ").strip()
        if raw == "0":
            return
        if raw.isdigit() and 1 <= int(raw) <= len(available):
            label, rel_path = available[int(raw) - 1]
            clip_path = str(PROJECT_ROOT / rel_path)
            expected_word = label.split()[0].strip()
            print(f"\n  Running inference on: {Path(clip_path).name}")
            print(f"  Expected word       : {expected_word}\n")
            run([
                PYTHON,
                str(PROJECT_ROOT / "training" / "infer.py"),
                "word",
                clip_path,
                "--checkpoint",
                str(PROJECT_ROOT
                    / "training/checkpoints_words/best_model_words.pth"),
                "--top-k", "3"])
            print()
            continue
        print("  Invalid choice.")


def run_video_demo() -> None:
    """Run word inference on a pre-recorded video file (mp4/mov/avi)"""
    clear()
    print("   Demo 4 — Pre-Recorded Video Inference")
    print("=" * 60)
    print(
        "  Point to any MP4/MOV/AVI video of someone saying a single word.\n"
        "  Works with phone recordings, screen recordings or clips saved\n"
    )
    while True:
        raw = input(
            "  Enter video file path (or 0 to go back): "
        ).strip().strip('"')
        if raw == "0":
            return
        clip = Path(raw)
        if not clip.exists():
            clip = PROJECT_ROOT / raw
        if not clip.exists():
            print(f"  File not found: {raw}")
            continue
        if clip.suffix.lower() not in {
            '.mp4', '.mov', '.avi', '.mkv', '.mpg', '.mpeg'
        }:
            print(
                "  Unsupported file type. Use mp4, mov, avi, mkv or mpg.")
            continue
        break
    print(f"\n  Running inference on: {clip.name}\n")
    run([
        PYTHON,
        str(PROJECT_ROOT / "training" / "infer.py"),
        "word",
        str(clip),
        "--checkpoint",
        str(PROJECT_ROOT
            / "training/checkpoints_words/best_model_words.pth"),
        "--top-k", "5"])
    print()


def run_webcam_demo() -> None:
    """Launch the live webcam word inference demo."""
    clear()
    print("   Demo 3 — Live Webcam Inference")
    print("=" * 60)
    print(
        "  Opens your webcam and predicts words from lip movements.\n"
        "  The model was trained on GRID corpus (controlled studio)\n"
        "\n"
        "  Press Q in the video window to quit.\n")
    input("  Press Enter to start...")
    run([PYTHON, str(PROJECT_ROOT / "testing" / "webcam_test.py")])


def run_my_visemes_demo() -> None:
    """Recognize visemes from videos in testing/my_sounds."""
    clear()
    print("   Demo 6 — Recognize My Viseme Videos")
    print("=" * 60)
    my_sounds_dir = PROJECT_ROOT / "testing" / "my_sounds"
    if not my_sounds_dir.exists():
        print(f"  Folder not found: {my_sounds_dir}")
        return
    videos = sorted([
        f for f in my_sounds_dir.iterdir()
        if f.suffix.lower() in {
            '.mp4', '.mov', '.avi', '.mkv', '.mpg', '.mpeg'
        }
    ])
    if not videos:
        print("  No video files found in my_sounds.")
        return
    print("  Pick a video to recognize:")
    for i, vid in enumerate(videos, 1):
        print(f"  {i}. {vid.name}")
    print("   0. Back to main menu\n")
    while True:
        raw = input("  Enter number: ").strip()
        if raw == "0":
            return
        if raw.isdigit() and 1 <= int(raw) <= len(videos):
            vid_path = videos[int(raw) - 1]
            print(f"\n  Running inference on: {vid_path.name}\n")
            checkpoint_path = PROJECT_ROOT / (
                "training/checkpoints_bozkurt_viseme/"
                "bozkurt_viseme_best_model.pth"
            )
            run([
                PYTHON,
                str(PROJECT_ROOT / "training" / "infer.py"),
                "viseme",
                str(vid_path),
                "--checkpoint",
                str(checkpoint_path),
                "--top-k", "5"
            ])
            print()
            break
        print("  Invalid choice.")


def run_my_words_cropped_video_demo() -> None:
    """
    Recognize words from cropped videos in
    testing/my_words_cropped_videos.
    """
    clear()
    print("   Demo 5 — Recognize My Cropped Word Videos")
    print("=" * 60)
    crop_dir = PROJECT_ROOT / "testing" / "my_words_cropped_videos"
    if not crop_dir.exists():
        print(f"  Folder not found: {crop_dir}")
        return
    videos = sorted([
        f for f in crop_dir.iterdir()
        if f.suffix.lower() in {
            '.mp4', '.mov', '.avi', '.mkv', '.mpg', '.mpeg'
        }
    ])
    if not videos:
        print("  No cropped video files found in my_words_cropped_videos.")
        return
    print("  Pick a cropped video to recognize:")
    for i, vid in enumerate(videos, 1):
        print(f"  {i}. {vid.name}")
    print("   0. Back to main menu\n")
    while True:
        raw = input("  Enter number: ").strip()
        if raw == "0":
            return
        if raw.isdigit() and 1 <= int(raw) <= len(videos):
            vid_path = videos[int(raw) - 1]
            print(f"\n  Running inference on: {vid_path.name}\n")
            run([
                PYTHON,
                str(PROJECT_ROOT / "training" / "infer.py"),
                "word",
                str(vid_path),
                "--checkpoint",
                str(
                    PROJECT_ROOT
                    / "training/checkpoints_words/best_model_words.pth"
                ),
                "--top-k", "5"
            ])
            print()
            break
        print("  Invalid choice.")


def run_my_visemes_cropped_video_demo() -> None:
    """
    Recognize visemes from cropped videos in
    testing/my_sounds_cropped_videos.
    """
    clear()
    print("   Demo 7— Recognize My Cropped Viseme Videos")
    print("=" * 60)
    crop_dir = PROJECT_ROOT / "testing" / "my_sounds_cropped_videos"
    if not crop_dir.exists():
        print(f"  Folder not found: {crop_dir}")
        return
    videos = sorted([
        f for f in crop_dir.iterdir()
        if f.suffix.lower() in {
            '.mp4', '.mov', '.avi', '.mkv', '.mpg', '.mpeg'
        }
    ])
    if not videos:
        print("  No cropped video files found in my_sounds_cropped_videos.")
        return
    print("  Pick a cropped viseme video to recognize:")
    for i, vid in enumerate(videos, 1):
        print(f"  {i}. {vid.name}")
    print("   0. Back to main menu\n")
    while True:
        raw = input("  Enter number: ").strip()
        if raw == "0":
            return
        if raw.isdigit() and 1 <= int(raw) <= len(videos):
            vid_path = videos[int(raw) - 1]
            print(f"\n  Running inference on: {vid_path.name}\n")
            checkpoint_path = PROJECT_ROOT / (
                "training/checkpoints_bozkurt_viseme/"
                "bozkurt_viseme_best_model.pth"
            )
            run([
                PYTHON,
                str(PROJECT_ROOT / "training" / "infer.py"),
                "viseme",
                str(vid_path),
                "--checkpoint",
                str(checkpoint_path),
                "--top-k", "5"
            ])
            print()
            break
        print("  Invalid choice.")


MENU_ITEMS = [
    (
        "Viseme-level inference — pick a phoneme clip, see predictions",
        run_viseme_demo
    ),
    (
        "Word-level inference — pick a GRID word clip, see predictions",
        run_word_demo
    ),
    (
        "Live webcam — real-time word prediction from camera",
        run_webcam_demo
    ),
    (
        "Recognize my word videos (testing/my_words)",
        run_my_words_demo
    ),
    (
        "Recognize my cropped word videos (testing/my_words_cropped_videos)",
        run_my_words_cropped_video_demo
    ),
    (
        "Recognize my viseme videos (testing/my_sounds)",
        run_my_visemes_demo
    ),
    (
        "Recognize my cropped viseme videos "
        "(testing/my_sounds_cropped_videos)",
        run_my_visemes_cropped_video_demo
    ),
]


def main() -> None:
    while True:
        print("\n" + "=" * 60)
        print("   Lip Reading Demo  —  FYP Project")
        print("=" * 60)
        for i, (label, _) in enumerate(MENU_ITEMS, 1):
            print(f"  {i}. {label}")
        print("  " + "─" * 58)
        print("   0. Exit")
        print()
        raw = input("  Choose an option: ").strip()
        if raw == "0":
            print("\n  Goodbye.\n")
            break
        if raw.isdigit() and 1 <= int(raw) <= len(MENU_ITEMS):
            _, fn = MENU_ITEMS[int(raw) - 1]
            fn()
        else:
            print("  Invalid choice — enter 1, 2, 3 or 0.")


if __name__ == "__main__":
    main()
