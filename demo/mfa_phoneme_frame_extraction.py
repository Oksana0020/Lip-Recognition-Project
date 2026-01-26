"""
MFA demo on GRID:
- Extract 16k mono WAV from .mpg using ffmpeg
- Build cleaned transcript from GRID .align
- Build per-utterance lexicon from CMUdict (ARPABET->MFA phone symbols)
- Run MFA align with lexicon.txt
- Parse TextGrid
- Match MFA word timings to GRID .align word sequence
- Build MFA phoneme intervals, convert to ARPABET, add 25k-frame fields
- Attach word labels to phonemes + group phones by words
- Export JSON with phoneme intervals
- Save 1 labeled frame per phoneme midpoint
References:
- GRID Audio-Visual Corpus: http://spandh.dcs.shef.ac.uk/gridcorpus/
- Montreal Forced Aligner: https://montreal-forced-aligner.readthedocs.io/
- CMU Pronouncing Dictionary: http://www.speech.cs.cmu.edu/cgi-bin/cmudict
"""

import argparse
import json
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, List
import re


try:
    import cv2
    OPENCV_IS_AVAILABLE = True
except Exception:
    cv2 = None
    OPENCV_IS_AVAILABLE = False
    print("OpenCV not available")


# CMU Pronouncing Dictionary (CMUdict)
# Source: http://www.speech.cs.cmu.edu/cgi-bin/cmudict
# ARPABET phonetic transcription system documentation
# https://en.wikipedia.org/wiki/ARPABET
def ensure_cmudict() -> object:
    try:
        from nltk.corpus import cmudict
        _ = cmudict.dict()
        return cmudict
    except Exception:
        print("NLTK or cmudict not available — installing/downloading now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"])
        import nltk
        try:
            nltk.data.find("corpora/cmudict")
        except Exception:
            nltk.download("cmudict")
        from nltk.corpus import cmudict
        return cmudict


cmudict_module = ensure_cmudict()
cmudict_dictionary = cmudict_module.dict()


# Letter to spoken name mapping for single-character words
# Adapted from standard English alphabet pronunciation
_letter_name_map = {
    "a": "a", "b": "bee", "c": "see", "d": "dee", "e": "ee", "f": "ef",
    "g": "gee", "h": "aitch", "i": "eye", "j": "jay", "k": "kay", "l": "el",
    "m": "em", "n": "en", "o": "oh", "p": "pee", "q": "cue", "r": "ar",
    "s": "ess", "t": "tee", "u": "you", "v": "vee", "w": "doubleyou",
    "x": "ex", "y": "why", "z": "zee",
}

# Common ARPABET -> model phone mapping (english_mfa)
# Taken and adapted from Montreal Forced Aligner documentation
# https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/dictionary.html
_arpabet_to_model = {
    "AA": "ɑ", "AE": "æ", "AH": "ə", "AO": "ɔ", "AW": "aw", "AY": "aj",
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "EH": "ɛ", "ER": "ɜ",
    "EY": "ej", "F": "f", "G": "ɡ", "HH": "h", "IH": "ɪ", "IY": "i",
    "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ",
    "OW": "ow", "OY": "oj", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ",
    "T": "t", "TH": "θ", "UH": "ʊ", "UW": "u",
    "V": "v", "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ",
}

_model_to_arpabet = {v: k for k, v in _arpabet_to_model.items()}
_model_to_arpabet.update({"sil": "SIL", "sp": "SIL", "spn": "SIL"})


def cmu_pron_to_model_pron(cmu_phones: List[str]) -> List[str]:
    model_phones: List[str] = []
    for cmu_phone in cmu_phones:
        arpabet_key = re.sub(r"\d", "", cmu_phone).upper()
        mapped_phone = _arpabet_to_model.get(arpabet_key)
        model_phones.append(
            mapped_phone if mapped_phone else arpabet_key.lower()
        )
    return model_phones


def ensure_mfa_model(model_name: str = "english_mfa") -> bool:
    """Ensure MFA acoustic model is downloaded."""
    check_cmd = [
        "C:/Users/oksan/miniconda3/Scripts/conda.exe",
        "run",
        "-n",
        "base",
        "mfa",
        "model",
        "list",
        "acoustic",
    ]
    
    result = subprocess.run(
        check_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    
    if model_name in result.stdout:
        return True
    
    print(f"Downloading MFA acoustic model: {model_name}")
    download_cmd = [
        "C:/Users/oksan/miniconda3/Scripts/conda.exe",
        "run",
        "-n",
        "base",
        "mfa",
        "model",
        "download",
        "acoustic",
        model_name,
    ]
    
    proc = subprocess.run(download_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        print(f"ERROR downloading model: {proc.stderr.decode('utf-8', errors='ignore')}")
        return False
    
    print(f"✓ Model {model_name} downloaded successfully")
    return True


def get_ffmpeg_path() -> str:
    project_ffmpeg = (
        Path(__file__).parent.parent / "tools" / "ffmpeg" / "ffmpeg.exe"
    )
    if project_ffmpeg.exists():
        return str(project_ffmpeg)

    if shutil.which("ffmpeg"):
        return "ffmpeg"

    conda_ffmpeg = Path(sys.prefix) / "Library" / "bin" / "ffmpeg.exe"
    if conda_ffmpeg.exists():
        return str(conda_ffmpeg)

    raise FileNotFoundError(
        "ffmpeg not found. Install ffmpeg or ensure it is on PATH."
    )


def extract_words_from_alignment(alignment_file: Path) -> List[Dict]:
    words: List[Dict] = []
    if not alignment_file.exists():
        return words

    content = alignment_file.read_text(
        encoding="utf-8", errors="ignore"
    ).strip()

    for line in content.splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue

        start_frame = int(parts[0])
        end_frame = int(parts[1])
        word = parts[2]
        if word.lower() == "sil":
            continue

        words.append(
            {
                "word": word,
                "start_time": start_frame / 25000.0,
                "end_time": end_frame / 25000.0,
            }
        )
    return words


# TextGrid format specification (Praat)
# Reference https://www.fon.hum.uva.nl/praat/manual/TextGrid_file_formats.html
def _parse_textgrid_phones(tg_file: Path) -> List[Dict]:
    phone_intervals: List[Dict] = []
    in_phones_tier = False
    current: Dict = {}

    with open(tg_file, "r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = raw.strip()

            if (
                line.lower().startswith("name =")
                and "phones" in line.lower()
            ):
                in_phones_tier = True
                continue

            if not in_phones_tier:
                continue

            if line.startswith("intervals ["):
                current = {}
                continue

            if line.startswith("xmin ="):
                current["start_time"] = float(
                    line.split("=", 1)[1].strip()
                )
            elif line.startswith("xmax ="):
                current["end_time"] = float(
                    line.split("=", 1)[1].strip()
                )
            elif line.startswith("text ="):
                text = line.split("=", 1)[1].strip().strip('"')
                current["phoneme"] = text if text else "sil"
                phone_intervals.append(current.copy())
                current = {}

    return phone_intervals


def _parse_textgrid_words(tg_file: Path) -> List[Dict]:
    word_intervals: List[Dict] = []
    in_words_tier = False
    current: Dict = {}

    with open(tg_file, "r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = raw.strip()

            if (
                line.lower().startswith("name =")
                and "words" in line.lower()
            ):
                in_words_tier = True
                continue

            if not in_words_tier:
                continue

            if line.startswith("intervals ["):
                current = {}
                continue

            if line.startswith("xmin ="):
                current["start_time"] = float(
                    line.split("=", 1)[1].strip()
                )
            elif line.startswith("xmax ="):
                current["end_time"] = float(
                    line.split("=", 1)[1].strip()
                )
            elif line.startswith("text ="):
                text = line.split("=", 1)[1].strip().strip('"')
                if text:
                    current["word"] = text
                    word_intervals.append(current.copy())
                current = {}

    return word_intervals


def match_mfa_timings_to_align_words(
    mfa_words: List[Dict],
    align_words: List[Dict],
) -> List[Dict]:
    matched_words: List[Dict] = []
    mfa_word_index = 0

    for align_word in align_words:
        align_word_text = align_word.get("word")
        align_start_time = align_word.get("start_time", 0.0)
        align_end_time = align_word.get("end_time", align_start_time)
        timing_matched = False

        while mfa_word_index < len(mfa_words):
            mfa_word = mfa_words[mfa_word_index]
            mfa_word_text = mfa_word.get("word")

            if (mfa_word_text or "").strip().lower() == (
                align_word_text or ""
            ).strip().lower():
                matched_words.append(
                    {
                        "word": align_word_text,
                        "start_time": float(
                            mfa_word.get("start_time", align_start_time)
                        ),
                        "end_time": float(
                            mfa_word.get("end_time", align_end_time)
                        ),
                    }
                )
                mfa_word_index += 1
                timing_matched = True
                break

            mfa_word_index += 1

        if not timing_matched:
            matched_words.append(
                {
                    "word": align_word_text,
                    "start_time": align_start_time,
                    "end_time": align_end_time,
                }
            )

    return matched_words


def build_mfa_phone_intervals(tg_path: Path) -> List[Dict]:
    phones = _parse_textgrid_phones(tg_path)
    out: List[Dict] = []

    for ph in phones:
        s = float(ph.get("start_time", 0.0))
        e = float(ph.get("end_time", s))
        ipa = ph.get("phoneme", "sil")
        arpabet = _model_to_arpabet.get(ipa, str(ipa).upper())
        start_frame = int(s * 25000)
        end_frame = int(e * 25000)

        out.append(
            {
                "phoneme": arpabet,
                "start_time": s,
                "end_time": e,
                "duration": e - s,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "frame_count": max(0, end_frame - start_frame),
                "alignment_method": "mfa",
            }
        )

    return out


def group_phones_by_words(
    word_intervals: List[Dict],
    phone_intervals: List[Dict],
) -> List[Dict]:
    groups: List[Dict] = []

    for w in word_intervals:
        ws = float(w.get("start_time", 0.0))
        we = float(w.get("end_time", ws))
        word_label = w.get("word")

        phones = [
            p
            for p in phone_intervals
            if float(p.get("start_time", 0.0)) >= ws
            and float(p.get("end_time", 0.0)) <= we
        ]

        groups.append(
            {
                "word": word_label,
                "start_time": ws,
                "end_time": we,
                "phonemes": phones,
                "phoneme_count": len(phones),
            }
        )

    return groups


def save_phoneme_frames(
    video_file: Path,
    phoneme_intervals: List[Dict],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if OPENCV_IS_AVAILABLE:
        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            print(f"Could not open video: {video_file}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        saved = 0

        for idx, ph in enumerate(phoneme_intervals, start=1):
            if str(ph.get("phoneme", "")).lower() in {"sil", "spn", "sp"}:
                continue

            mid_t = 0.5 * (ph["start_time"] + ph["end_time"])
            frame_idx = int(mid_t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            word = ph.get("word") or ""
            label = f"{ph['phoneme']} ({word})" if word else f"{ph['phoneme']}"
            cv2.putText(
                frame,
                label,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            out_path = out_dir / (
                f"{video_file.stem}_{idx:03d}_{ph['phoneme']}_{mid_t:.3f}s.jpg"
            )
            if cv2.imwrite(str(out_path), frame):
                saved += 1

        cap.release()
        print(f"Saved {saved} phoneme frames to: {out_dir}")
        return

    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
    saved = 0

    for idx, ph in enumerate(phoneme_intervals, start=1):
        if str(ph.get("phoneme", "")).lower() in {"sil", "spn", "sp"}:
            continue

        mid_t = 0.5 * (
            float(ph.get("start_time", 0.0)) + float(ph.get("end_time", 0.0))
        )
        fname = (
            f"{video_file.stem}_{idx:03d}_"
            f"{ph.get('phoneme', 'unk')}_{mid_t:.3f}s.jpg"
        )
        out_path = out_dir / fname

        cmd = [
            ffmpeg_bin, "-y", "-ss", f"{mid_t:.3f}", "-i", str(video_file),
            "-frames:v", "1", "-q:v", "2", str(out_path),
        ]
        proc = subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        if proc.returncode == 0 and out_path.exists():
            saved += 1

    print(f"Saved {saved} phoneme frames (ffmpeg fallback) to: {out_dir}")


def main(
    video_path: Optional[Path] = None,
    align_path: Optional[Path] = None,
    tmpdir: Optional[Path] = None,
    keep_temp: bool = False,
    out_root: Optional[Path] = None,
    skip_model_check: bool = False,
) -> Dict:
    if video_path is None or align_path is None:
        grid_root = Path(__file__).parent.parent
        grid_data_path = grid_root / "data" / "grid" / "GRID dataset full"
        videos_path = grid_data_path / "s1"
        alignments_path = grid_data_path / "alignments" / "s1"

        video_files = sorted(
            [f for f in videos_path.iterdir() if f.suffix.lower() == ".mpg"]
        )
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
        # 0) Ensure MFA model is downloaded (skip if already checked in batch)
        if not skip_model_check:
            if not ensure_mfa_model("english_mfa"):
                raise RuntimeError("Failed to download MFA acoustic model")
        
        # 1) Extract WAV
        wav_path = utter_dir / f"{video_path.stem}.wav"
        ffmpeg_bin = get_ffmpeg_path()
        ffmpeg_cmd = [
            ffmpeg_bin,
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(wav_path),
        ]
        print("Running ffmpeg:", " ".join(ffmpeg_cmd))
        proc = subprocess.run(
            ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if proc.returncode != 0 or not wav_path.exists():
            raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore"))

        # 2) Transcript from .align
        align_words = extract_words_from_alignment(align_path)
        cleaned: List[str] = []
        for w in align_words:
            tok = str(w.get("word", "")).strip()
            if not tok:
                continue
            if re.match(r"^\[.*\]$", tok):
                continue
            if tok.lower() in {"sil", "spn", "<unk>", "<cutoff>"}:
                continue
            cleaned.append(tok)

        transcript = " ".join(cleaned)
        (utter_dir / f"{video_path.stem}.txt").write_text(
            transcript, encoding="utf-8"
        )

        # Build lexicon
        lex_path = utter_dir / "lexicon.txt"
        words_for_lex: set[str] = set()
        for tok in transcript.split():
            w = re.sub(r"[^A-Za-z0-9]", "", tok).strip().lower()
            if w:
                words_for_lex.add(w)

        with open(lex_path, "w", encoding="utf-8") as lf:
            for w in sorted(words_for_lex):
                entry = None
                if w in cmudict_dictionary:
                    cmu_phones = cmudict_dictionary[w][0]
                    entry = " ".join(
                        cmu_pron_to_model_pron(cmu_phones)
                    )
                elif len(w) == 1 and w in _letter_name_map:
                    spelled = _letter_name_map[w]
                    if spelled in cmudict_dictionary:
                        cmu_phones = cmudict_dictionary[spelled][0]
                        entry = " ".join(
                            cmu_pron_to_model_pron(cmu_phones)
                        )

                if not entry:
                    entry = " ".join(list(w))

                lf.write(f"{w.upper()} {entry}\n")

        # 3) run MFA align
        mfa_out = tmp / "mfa_output"
        mfa_out.mkdir(parents=True, exist_ok=True)

        # Using conda run to execute MFA (required for kalpy on Windows)
        # kalpy is MFA's acoustic modeling backend and requires conda
        # Reference: https://github.com/mmcauliffe/kalpy
        # MFA installation guide
        # https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html
        cmd = [
            "C:/Users/oksan/miniconda3/Scripts/conda.exe",
            "run",
            "-n",
            "base",
            "mfa",
            "align",
            str(utter_dir),
            str(lex_path),
            "english_mfa",
            str(mfa_out),
            "--clean",
        ]
        print("Running MFA:", " ".join(cmd))
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore"))

        # 4) Locate TextGrid
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

        # 5) Words tier + match timings to GRID .align words
        mfa_words = _parse_textgrid_words(tg_path)
        word_intervals = match_mfa_timings_to_align_words(
            mfa_words, align_words
        ) if mfa_words else align_words

        # 6) PURE MFA phones -> ARPABET + frame fields
        phoneme_intervals = build_mfa_phone_intervals(tg_path)

        # 7) Attach word to each phoneme by midpoint
        for ph in phoneme_intervals:
            ph_mid = 0.5 * (ph.get("start_time", 0.0) +
                            ph.get("end_time", 0.0))
            for w in word_intervals:
                if (
                    float(w.get("start_time", 0.0)) <= ph_mid
                    <= float(w.get("end_time", 0.0))
                ):
                    ph["word"] = w.get("word")
                    break

        word_groups = group_phones_by_words(word_intervals, phoneme_intervals)

        if out_root is None:
            out_root = Path(__file__).parent / "phoneme_frames_mfa"
        out_root.mkdir(parents=True, exist_ok=True)
        vid_out = out_root / video_path.stem
        vid_out.mkdir(parents=True, exist_ok=True)

        save_phoneme_frames(video_path, phoneme_intervals, vid_out)

        out_json = vid_out / f"{video_path.stem}_phoneme_intervals.json"
        payload = {
            "video": str(video_path),
            "textgrid": str(tg_path),
            "transcript": transcript,
            "alignment_method": "mfa_pure",
            "phoneme_intervals": phoneme_intervals,
            "word_groups": word_groups,
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print("Done. Frames+JSON saved to:", vid_out)
        return payload
    finally:
        if use_temp_context and cleanup_tmp:
            try:
                shutil.rmtree(tmp)
            except Exception:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tmpdir",
        type=str,
        default=None,
        help="temporary folder for MFA corpus",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="do not remove MFA temp folder",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="output root for frames",
    )
    parser.add_argument(
        "--num-videos",
        type=int,
        default=2,
        help="number of GRID videos to process",
    )
    args = parser.parse_args()

    tmp = Path(args.tmpdir) if args.tmpdir else None
    out = Path(args.outdir) if args.outdir else None

    grid_root = Path(__file__).parent.parent
    grid_data_path = grid_root / "data" / "grid" / "GRID dataset full"
    videos_path = grid_data_path / "s1"
    alignments_path = grid_data_path / "alignments" / "s1"

    video_files = sorted(
        [f for f in videos_path.iterdir() if f.suffix.lower() == ".mpg"]
    )[:args.num_videos]
    if not video_files:
        print("No GRID video files found")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"Processing {len(video_files)} GRID videos with PURE MFA alignment")
    print(f"{'='*70}\n")

    results = []
    for idx, video_file in enumerate(video_files, 1):
        base = video_file.stem
        align_file = alignments_path / f"{base}.align"

        print(f"\n[{idx}/{len(video_files)}] Processing: {video_file.name}")
        print(f"{'='*70}")

        try:
            result = main(
                video_path=video_file,
                align_path=align_file,
                tmpdir=tmp,
                keep_temp=args.keep_temp,
                out_root=out,
            )
            results.append(result)
            num_ph = len(result.get("phoneme_intervals", []))
            print(f"✓ Success: {num_ph} phonemes extracted")
        except Exception as e:
            print(f"✗ Failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"SUMMARY: Processed {len(results)}/{len(video_files)} videos "
          f"successfully")
    print(f"{'='*70}")
