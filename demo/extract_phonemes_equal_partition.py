"""Equal-partition phoneme extractor using the CMU Pronouncing Dictionary.
For each word in a GRID .align file this script looks up the CMU pronunciation
and divides the word duration evenly across the CMU phonemes.
"""

from pathlib import Path
from typing import List, Dict
import re
import sys
import subprocess
import json
import os


def ensure_cmudict() -> object:
    """Import or install + download the NLTK cmudict resource and return it."""
    try:
        from nltk.corpus import cmudict
        _ = cmudict.dict()
        return cmudict
    except Exception:
        print("NLTK or cmudict not available — installing/downloading now...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "nltk"],
            stdout=subprocess.DEVNULL,
        )
        import nltk

        try:
            nltk.data.find("corpora/cmudict")
        except Exception:
            nltk.download("cmudict")

        from nltk.corpus import cmudict
        return cmudict


_cmu = ensure_cmudict()
_cmu_dict = _cmu.dict()


_letter_name_map = {
    "a": "a",
    "b": "bee",
    "c": "see",
    "d": "dee",
    "e": "ee",
    "f": "ef",
    "g": "gee",
    "h": "aitch",
    "i": "eye",
    "j": "jay",
    "k": "kay",
    "l": "el",
    "m": "em",
    "n": "en",
    "o": "oh",
    "p": "pee",
    "q": "cue",
    "r": "ar",
    "s": "ess",
    "t": "tee",
    "u": "you",
    "v": "vee",
    "w": "doubleyou",
    "x": "ex",
    "y": "why",
    "z": "zee",
}


def normalize_word(w: str) -> str:
    w = w.strip()
    w = re.sub(r"^[^A-Za-z0-9<>]+|[^A-Za-z0-9<>]+$", "", w)
    return w


def word_to_phonemes(word: str) -> List[str]:
    """Return CMU phonemes (ARPABET) for a word or empty list if not found."""
    if word is None:
        return []
    w = normalize_word(word).lower()
    if not w:
        return []

    # direct lookup in dictionary
    if w in _cmu_dict:
        phones = _cmu_dict[w][0]
        return [re.sub(r"\d", "", p) for p in phones]

    # single-letter spelled out form
    if len(w) == 1 and w in _letter_name_map:
        spelled = _letter_name_map[w]
        if spelled in _cmu_dict:
            phones = _cmu_dict[spelled][0]
            return [re.sub(r"\d", "", p) for p in phones]

    return []


def find_grid_videos_and_alignments() -> List[Dict]:
    """Discover video+alignment pairs in the GRID dataset (s1 speaker)."""
    root = Path(__file__).parent.parent
    grid_root = root / "data" / "grid" / "GRID dataset full"
    videos_dir = grid_root / "s1"
    aligns_dir = grid_root / "alignments" / "s1"

    pairs: List[Dict] = []
    if not videos_dir.exists() or not aligns_dir.exists():
        return pairs

    for fn in sorted(os.listdir(videos_dir)):
        if not fn.endswith(".mpg"):
            continue
        base = Path(fn).stem
        vpath = videos_dir / fn
        apath = aligns_dir / f"{base}.align"
        if apath.exists():
            pairs.append(
                {
                    "video_path": vpath,
                    "alignment_path": apath,
                    "base_name": base,
                }
            )

    return pairs


def extract_words_from_alignment(alignment_file: Path) -> List[Dict]:
    words: List[Dict] = []
    with open(alignment_file, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            start_frame = int(parts[0])
            end_frame = int(parts[1])
            word = parts[2]
            if word.lower() == "sil":
                continue
            start_time = start_frame / 25000.0
            end_time = end_frame / 25000.0
            words.append({
                "word": word,
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
                "start_frame": start_frame,
                "end_frame": end_frame,
            })
    return words


def extract_phonemes_equal_partition(
    video_file: Path, alignment_file: Path
) -> Dict:
    words = extract_words_from_alignment(alignment_file)
    words_out: List[Dict] = []
    phoneme_entries: List[Dict] = []

    for w in words:
        word_text = w["word"]
        s = w["start_time"]
        e = w["end_time"]
        cms = word_to_phonemes(word_text)
        if not cms:
            cms = ["spn"]
        per = (e - s) / len(cms) if cms else 0.0
        phs: List[Dict] = []
        for i, ph in enumerate(cms):
            ps = s + i * per
            pe = ps + per
            entry = {
                "phoneme": ph,
                "word": word_text,
                "start_time": ps,
                "end_time": pe,
                "duration": pe - ps,
                "start_frame": int(ps * 25000),
                "end_frame": int(pe * 25000),
                "alignment_method": "equal_partition",
            }
            phs.append(entry)
            phoneme_entries.append(entry)
        words_out.append({
            "word": word_text,
            "word_timing": {"start_time": s, "end_time": e, "duration": e - s},
            "phonemes": phs,
            "phoneme_count": len(phs),
        })

    return {
        "video_file": str(video_file),
        "sentence": " ".join(w["word"] for w in words),
        "words": words_out,
        "phonemes": [p["phoneme"] for p in phoneme_entries],
        "total_words": len(words_out),
        "total_phonemes": len(phoneme_entries),
        "unique_phonemes": list(set(p["phoneme"] for p in phoneme_entries)),
        "unique_phoneme_count": len(
            set(p["phoneme"] for p in phoneme_entries)
        ),
        "alignment_method": "equal_partition",
        "phoneme_intervals": phoneme_entries,
    }


def display_phoneme_extraction(video_file: Path, phoneme_data: Dict):
    print(f"VIDEO: {video_file.name}")
    print(f"SENTENCE: {phoneme_data.get('sentence', '')}")
    print(f"WORDS: {phoneme_data.get('total_words', 0)}")
    print(f"PHONEMES: {phoneme_data.get('total_phonemes', 0)}")
    print(f"UNIQUE PHONEMES: {phoneme_data.get('unique_phoneme_count', 0)}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--align", type=str, default=None)
    args = parser.parse_args()

    if args.video and args.align:
        video = Path(args.video)
        align = Path(args.align)
    else:
        pairs = find_grid_videos_and_alignments()
        if not pairs:
            print("No GRID video+alignment pairs found")
            sys.exit(1)
        first = pairs[0]
        video = first["video_path"]
        align = first["alignment_path"]

    out = extract_phonemes_equal_partition(video, align)
    payload = {
        "video_processed": str(video),
        "alignment_file": str(align),
        "phoneme_extraction": out,
    }

    out_json_dir = Path(__file__).parent
    out_json = out_json_dir / "phoneme_extraction_equal_partition.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote: {out_json}")
    display_phoneme_extraction(video, out)


if __name__ == "__main__":
    main()
