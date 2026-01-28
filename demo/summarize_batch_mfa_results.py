"""
Demo script summarize phoneme statistics from batch MFA extraction results.
The script reads demo/batch_mfa_results.json and aggregates per-video and
overall phoneme statistics, including total phoneme counts, unique phoneme
counts, UNK frequency and sample of observed phoneme labels.
"""

import json
from pathlib import Path


def main() -> None:
    batch_results_path = Path("batch_mfa_results.json")
    with batch_results_path.open(encoding="utf-8") as f:
        batch_data = json.load(f)

    video_results = batch_data.get("results", [])

    summary_rows = []
    for video_result in video_results:
        intervals = video_result.get("phoneme_intervals", [])
        phoneme_list = [
            it.get("phoneme")
            for it in intervals
            if it.get("phoneme")
        ]

        unique_phonemes = sorted(set(phoneme_list))
        unk_count = sum(1 for p in phoneme_list if p == "UNK")

        base_name = video_result.get("base_name")
        if not base_name:
            frames_dir = video_result.get("frames_dir", "")
            base_name = Path(frames_dir).name

        summary_rows.append(
            (
                base_name,
                len(phoneme_list),
                len(unique_phonemes),
                unk_count,
                unique_phonemes[:8],
            )
        )

    for row in sorted(summary_rows):
        base_name, total_p, unique_n, unk_n, sample = row
        line1 = (
            f"{base_name}: total={total_p}, "
            f"unique={unique_n}, UNK={unk_n},"
        )
        line2 = f" uniq_sample={sample}"
        print(line1 + line2)

    all_phonemes = []
    for video_result in video_results:
        for it in video_result.get("phoneme_intervals", []):
            ph = it.get("phoneme")
            if ph:
                all_phonemes.append(ph)

    all_unique_phonemes = sorted(set(all_phonemes))

    print(
        "\nOverall: "
        f"videos={len(summary_rows)}, "
        f"total_phonemes={len(all_phonemes)}, "
        f"unique_phonemes={len(all_unique_phonemes)}"
    )
    print("Unique phonemes (sample 40):")
    print(all_unique_phonemes[:40])


if __name__ == "__main__":
    main()
