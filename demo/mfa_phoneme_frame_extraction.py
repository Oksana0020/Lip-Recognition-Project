"""
MFA demo on GRID:
- Extract 16k mono WAV from .mpg using ffmpeg
- Build transcript from GRID .align
- Run MFA align
- Locate TextGrid output
"""

import argparse
import sys
import shutil
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Optional, Dict, List


def get_ffmpeg_path() -> str:
    project_ffmpeg = Path(__file__).parent.parent / "tools" / "ffmpeg" / "ffmpeg.exe"
    if project_ffmpeg.exists():
        return str(project_ffmpeg)
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    conda_ffmpeg = Path(sys.prefix) / "Library" / "bin" / "ffmpeg.exe"
    if conda_ffmpeg.exists():
        return str(conda_ffmpeg)
    raise FileNotFoundError("ffmpeg not found. Install ffmpeg or ensure it is on PATH.")


def extract_words_from_alignment(alignment_file: Path) -> List[str]:
    words: List[str] = []
    if not alignment_file.exists():
        return words
    content = alignment_file.read_text(encoding="utf-8", errors="ignore").strip()
    for line in content.splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        word = parts[2]
        if word.lower() == "sil":
            continue
        words.append(word)
    return words


def main(
    video_path: Optional[Path] = None,
    align_path: Optional[Path] = None,
    tmpdir: Optional[Path] = None,
    keep_temp: bool = False,
) -> Dict:
    if video_path is None or align_path is None:
        grid_root = Path(__file__).parent.parent
        grid_data_path = grid_root / "data" / "grid" / "GRID dataset full"
        videos_path = grid_data_path / "s1"
        alignments_path = grid_data_path / "alignments" / "s1"

        video_files = sorted([f for f in videos_path.iterdir() if f.suffix.lower() == ".mpg"])
        if not video_files:
            print("No GRID video files found")
            return {}

        video_path = video_files[0]
        align_path = alignments_path / f"{video_path.stem}.align"

    video_path = Path(video_path)
    align_path = Path(align_path)

    use_temp_context = tmpdir is None
    if use_temp_context:
        tmp_ctx = tempfile.TemporaryDirectory()
        tmp = Path(tmp_ctx.name)
        cleanup_tmp = True
    else:
        tmp = Path(tmpdir)
        tmp.mkdir(parents=True, exist_ok=True)
        cleanup_tmp = not keep_temp

    corpus = tmp / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    utter_dir = corpus / video_path.stem
    utter_dir.mkdir(parents=True, exist_ok=True)

    try:
        wav_path = utter_dir / f"{video_path.stem}.wav"
        ffmpeg_bin = get_ffmpeg_path()
        ffmpeg_cmd = [
            ffmpeg_bin, "-y", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(wav_path)
        ]
        print("Running ffmpeg:", " ".join(ffmpeg_cmd))
        proc = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0 or not wav_path.exists():
            raise RuntimeError("ffmpeg failed to create WAV")

        words = extract_words_from_alignment(align_path)
        txt_path = utter_dir / f"{video_path.stem}.txt"
        txt_path.write_text(" ".join(words), encoding="utf-8")

        mfa_out = tmp / "mfa_output"
        mfa_out.mkdir(parents=True, exist_ok=True)

        mfa_bin = shutil.which("mfa") or str(Path(sys.prefix) / "Scripts" / ("mfa.exe" if os.name == "nt" else "mfa"))
        cmd = [mfa_bin, "align", str(utter_dir), "english_us_arpa", "english_mfa", str(mfa_out)]
        print("Running MFA:", " ".join(cmd))
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore"))

        tg_path = None
        for f in mfa_out.rglob("*.TextGrid"):
            tg_path = f
            break
        if tg_path is None:
            for f in mfa_out.rglob("*.textgrid"):
                tg_path = f
                break

        if tg_path is None:
            raise RuntimeError("MFA did not produce a TextGrid file.")

        print("OK TextGrid:", tg_path)
        return {"video": str(video_path), "textgrid": str(tg_path)}
    finally:
        if use_temp_context and cleanup_tmp:
            try:
                shutil.rmtree(tmp)
            except Exception:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tmpdir", type=str, default=None)
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()

    tmp = Path(args.tmpdir) if args.tmpdir else None
    main(tmpdir=tmp, keep_temp=args.keep_temp)