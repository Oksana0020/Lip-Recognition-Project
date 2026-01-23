"""
MFA demo on GRID:
- Extract 16k mono WAV from .mpg using ffmpeg
- Build cleaned transcript from GRID .align
- Build per-utterance lexicon from CMUdict (ARPABET->MFA phone symbols)
- Run MFA align with lexicon.txt
- Parse TextGrid "phones" tier
- Export JSON with phoneme intervals
"""

import argparse
import json
import sys
import shutil
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Optional, Dict, List
import re


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


_letter_name_map = {
    "a": "a", "b": "bee", "c": "see", "d": "dee", "e": "ee", "f": "ef",
    "g": "gee", "h": "aitch", "i": "eye", "j": "jay", "k": "kay", "l": "el",
    "m": "em", "n": "en", "o": "oh", "p": "pee", "q": "cue", "r": "ar",
    "s": "ess", "t": "tee", "u": "you", "v": "vee", "w": "doubleyou",
    "x": "ex", "y": "why", "z": "zee",
}

# Common ARPABET -> model phone mapping (english_mfa)
_arpabet_to_model = {
    "AA": "ɑ", "AE": "æ", "AH": "ə", "AO": "ɔ", "AW": "aw", "AY": "aj",
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "EH": "ɛ", "ER": "ɜ",
    "EY": "ej", "F": "f", "G": "ɡ", "HH": "h", "IH": "ɪ", "IY": "i",
    "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ",
    "OW": "ow", "OY": "oj", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ",
    "T": "t", "TH": "θ", "UH": "ʊ", "UW": "u",
    "V": "v", "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ",
}


def cmu_pron_to_model_pron(cmu_phones: List[str]) -> List[str]:
    model_phones: List[str] = []
    for cmu_phone in cmu_phones:
        arpabet_key = re.sub(r"\d", "", cmu_phone).upper()
        mapped_phone = _arpabet_to_model.get(arpabet_key)
        model_phones.append(
            mapped_phone if mapped_phone else arpabet_key.lower()
        )
    return model_phones


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


def main(
    video_path: Optional[Path] = None,
    align_path: Optional[Path] = None,
    tmpdir: Optional[Path] = None,
    keep_temp: bool = False,
    out_root: Optional[Path] = None,
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

        mfa_bin = shutil.which("mfa")
        if not mfa_bin:
            cand = Path(sys.prefix) / "Scripts" / (
                "mfa.exe" if os.name == "nt" else "mfa"
            )
            if cand.exists():
                mfa_bin = str(cand)

        if not mfa_bin:
            mfa_bin = "mfa"

        cmd = [
            mfa_bin,
            "align",
            str(utter_dir),
            str(lex_path),
            "english_mfa",
            str(mfa_out),
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

        # 5) Parse phones tier and JSON
        phone_intervals = _parse_textgrid_phones(tg_path)

        if out_root is None:
            out_root = Path(__file__).parent / "phoneme_json_mfa"
        out_root.mkdir(parents=True, exist_ok=True)

        out_json = out_root / f"{video_path.stem}_phones.json"
        payload = {
            "video": str(video_path),
            "textgrid": str(tg_path),
            "transcript": transcript,
            "phoneme_intervals": phone_intervals,
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print("Done JSON:", out_json)
        return payload
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
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()

    tmp = Path(args.tmpdir) if args.tmpdir else None
    out = Path(args.outdir) if args.outdir else None
    main(tmpdir=tmp, keep_temp=args.keep_temp, out_root=out)
