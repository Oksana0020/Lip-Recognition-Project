"""
Demo script to summarize phoneme statistics from equal-partition extraction.
Script reads demo/phoneme_extraction_equal_partition.json and produce
per-video and overall phoneme statistics, including total phoneme counts,
unique phoneme counts and unique observed phonemes.
"""

import json
from pathlib import Path


def main() -> None:
    script_dir = Path(__file__).parent
    results_path = script_dir / "phoneme_extraction_equal_partition.json"

    if not results_path.exists():
        print(f"ERROR: Results file not found: {results_path}")
        return

    with results_path.open(encoding="utf-8") as f:
        data = json.load(f)

    # expect batch format with results array
    if not isinstance(data, dict) or "results" not in data:
        print("ERROR: Invalid format.")
        return

    video_results = data.get("results", [])

    if not video_results:
        print("No results found in the file.")
        return

    summary_rows = []

    for video_result in video_results:
        base_name = video_result.get("base_name", "unknown")
        phoneme_list = []

        if "phoneme_extraction" in video_result:
            # format with phoneme_extraction object
            phoneme_data = video_result["phoneme_extraction"]
            words = phoneme_data.get("words", [])
            for word in words:
                for phoneme in word.get("phonemes", []):
                    phoneme_list.append(phoneme.get("phoneme"))
        elif "phoneme_intervals" in video_result:
            # format with phoneme_intervals array
            for it in video_result.get("phoneme_intervals", []):
                ph = it.get("phoneme")
                if ph:
                    phoneme_list.append(ph)

        unique_phonemes = sorted(set(phoneme_list))
        unk_count = sum(1 for p in phoneme_list if p == "UNK")

        summary_rows.append(
            (
                base_name,
                len(phoneme_list),
                len(unique_phonemes),
                unk_count,
                unique_phonemes[:8],
            )
        )

    if not summary_rows:
        print("No video results found to summarize.")
        return

    print("=" * 80)
    print("EQUAL PARTITION PHONEME EXTRACTION SUMMARY")
    print("=" * 80)
    print()

    for row in sorted(summary_rows):
        base_name, total_p, unique_n, unk_n, sample = row
        line1 = (
            f"{base_name}: total={total_p}, "
            f"unique={unique_n}, UNK={unk_n},"
        )
        line2 = f" uniq_sample={sample}"
        print(line1 + line2)

    # Aggregate all phonemes
    all_phonemes = []
    for video_result in video_results:
        if "phoneme_extraction" in video_result:
            phoneme_data = video_result["phoneme_extraction"]
            words = phoneme_data.get("words", [])
            for word in words:
                for phoneme in word.get("phonemes", []):
                    ph = phoneme.get("phoneme")
                    if ph:
                        all_phonemes.append(ph)
        elif "phoneme_intervals" in video_result:
            for it in video_result.get("phoneme_intervals", []):
                ph = it.get("phoneme")
                if ph:
                    all_phonemes.append(ph)

    all_unique_phonemes = sorted(set(all_phonemes))

    print()
    print("=" * 80)
    print(
        "Overall: "
        f"videos={len(summary_rows)}, "
        f"total_phonemes={len(all_phonemes)}, "
        f"unique_phonemes={len(all_unique_phonemes)}"
    )
    print("Unique phonemes:")
    print(all_unique_phonemes)
    print("=" * 80)


if __name__ == "__main__":
    main()
