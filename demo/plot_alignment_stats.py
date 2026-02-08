"""
Generate visual summaries for alignment_manual_ratings.csv.

This script creates plots for:
- Overall accuracy (with 95% CI) for equal-partition vs MFA
- Per-phoneme accuracy (top-N most frequent phonemes)
- Accuracy gap per phoneme (MFA - Equal), showing best and worst gaps
- Phoneme frequency counts (top-N)
- Phonemes missed by both methods (both wrong)
- Both-wrong rate by phoneme position index extracted from frame_name.
"""

import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

RATINGS_FILE = Path(__file__).parent / "alignment_manual_ratings.csv"
OUTPUT_DIR = Path(__file__).parent / "plots"


def _proportion_ci(
    num_correct: int,
    num_samples: int,
    z_score: float = 1.96,
) -> Tuple[float, float, float]:
    """Return (p, low, high) for normal-approx CI of a proportion."""
    if num_samples == 0:
        return 0.0, 0.0, 0.0

    proportion = num_correct / num_samples
    standard_error = math.sqrt(
        proportion * (1 - proportion) / num_samples
    )
    lower = max(0.0, proportion - z_score * standard_error)
    upper = min(1.0, proportion + z_score * standard_error)
    return proportion, lower, upper


def _safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def load_ratings() -> List[Dict[str, str]]:
    """Load manual ratings CSV into a list of dict rows."""
    if not RATINGS_FILE.exists():
        raise FileNotFoundError(
            f"Manual ratings file not found: {RATINGS_FILE}\n"
            "Create it with columns: video,frame_name,phoneme,"
            "equal_partition_evaluation,mfa_evaluation"
        )

    with RATINGS_FILE.open("r", encoding="utf-8") as file_handle:
        return list(csv.DictReader(file_handle))


def summarize_overall(rows: List[Dict[str, str]]) -> Dict[str, int]:
    """Summarize total samples and correct counts for each method."""
    total_samples = 0
    equal_correct = 0
    mfa_correct = 0

    for row in rows:
        try:
            equal_eval = int(row["equal_partition_evaluation"])
            mfa_eval = int(row["mfa_evaluation"])
        except (KeyError, ValueError):
            continue

        total_samples += 1
        equal_correct += equal_eval
        mfa_correct += mfa_eval

    return {
        "total": total_samples,
        "eq_correct": equal_correct,
        "mfa_correct": mfa_correct,
    }


def summarize_by_phoneme(
    rows: List[Dict[str, str]],
) -> Dict[str, Dict[str, int]]:
    """Aggregate totals and correct counts per phoneme."""
    stats: Dict[str, Dict[str, int]] = {}

    for row in rows:
        phoneme = (row.get("phoneme") or "").strip().upper()
        if not phoneme:
            continue

        try:
            equal_eval = int(row["equal_partition_evaluation"])
            mfa_eval = int(row["mfa_evaluation"])
        except (KeyError, ValueError):
            continue

        if phoneme not in stats:
            stats[phoneme] = {"total": 0, "eq_correct": 0, "mfa_correct": 0}

        stats[phoneme]["total"] += 1
        stats[phoneme]["eq_correct"] += equal_eval
        stats[phoneme]["mfa_correct"] += mfa_eval

    return stats


def plot_overall_accuracy(overall: Dict[str, int]) -> None:
    """Plot overall accuracy bars with 95% confidence intervals."""
    total = overall["total"]
    equal_correct = overall["eq_correct"]
    mfa_correct = overall["mfa_correct"]

    equal_p, equal_lo, equal_hi = _proportion_ci(equal_correct, total)
    mfa_p, mfa_lo, mfa_hi = _proportion_ci(mfa_correct, total)

    labels = ["Equal partition", "MFA"]
    values = [equal_p * 100, mfa_p * 100]
    errors = [
        [values[0] - equal_lo * 100, equal_hi * 100 - values[0]],
        [values[1] - mfa_lo * 100, mfa_hi * 100 - values[1]],
    ]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        labels,
        values,
        yerr=list(zip(*errors)),
        capsize=6,
    )
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Overall Accuracy with 95% CI")

    for idx, value in enumerate(values):
        upper_error = errors[idx][1]
        label_y = value + upper_error + 2.0
        ax.text(
            idx,
            label_y,
            f"{value:.1f}%",
            ha="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"),
        )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "overall_accuracy_ci.png", dpi=200)
    plt.close(fig)


def plot_per_phoneme_accuracy(
    stats: Dict[str, Dict[str, int]],
    top_n: int = 15,
) -> None:
    """Plot grouped bars for per-phoneme accuracy (top-N by frequency)."""
    items = []
    for phoneme, phoneme_stats in stats.items():
        total = phoneme_stats["total"]
        items.append(
            (
                phoneme,
                _safe_div(phoneme_stats["eq_correct"], total),
                _safe_div(phoneme_stats["mfa_correct"], total),
                total,
            )
        )

    items.sort(key=lambda item: item[3], reverse=True)
    items = items[:top_n]

    phonemes = [item[0] for item in items]
    equal_vals = [item[1] * 100 for item in items]
    mfa_vals = [item[2] * 100 for item in items]

    positions = list(range(len(phonemes)))
    bar_width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        [p - bar_width / 2 for p in positions],
        equal_vals,
        bar_width,
        label="Equal partition",
    )
    ax.bar(
        [p + bar_width / 2 for p in positions],
        mfa_vals,
        bar_width,
        label="MFA",
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(phonemes, rotation=45, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Per-Phoneme Accuracy (Top {top_n} by frequency)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "per_phoneme_accuracy_top.png", dpi=200)
    plt.close(fig)


def plot_accuracy_gap_phonemes(
    stats: Dict[str, Dict[str, int]],
    min_count: int = 5,
    top_n_each: int = 12,
    gap_eps: float = 1e-9,
) -> None:
    """
    Plot phonemes with largest accuracy gaps (MFA - Equal), both positive
    and negative. This avoids "empty" bars caused by clipped negative values.
    """
    gap_items = []
    for phoneme, phoneme_stats in stats.items():
        total = phoneme_stats["total"]
        if total < min_count:
            continue

        equal_acc = _safe_div(phoneme_stats["eq_correct"], total)
        mfa_acc = _safe_div(phoneme_stats["mfa_correct"], total)
        gap_pp = (mfa_acc - equal_acc) * 100

        if abs(gap_pp) <= gap_eps:
            continue

        gap_items.append((phoneme, gap_pp, total))

    if not gap_items:
        return

    # Best (largest positive gaps) and worst (most negative gaps)
    best = sorted(gap_items, key=lambda t: t[1], reverse=True)[:top_n_each]
    worst = sorted(gap_items, key=lambda t: t[1])[:top_n_each]

    combined = worst + best
    phonemes = [t[0] for t in combined]
    gaps = [t[1] for t in combined]

    y_min = min(gaps) - 1.0
    y_max = max(gaps) + 1.0

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(range(len(phonemes)), gaps)
    ax.axhline(0, linewidth=0.8)
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("Accuracy gap (pp): MFA – Equal")
    ax.set_title(
        "Phoneme Accuracy Gap (worst vs best, MFA – Equal)"
    )
    ax.set_xticks(range(len(phonemes)))
    ax.set_xticklabels(phonemes, rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "problem_phonemes_gap.png", dpi=200)
    plt.close(fig)


def plot_counts(
    stats: Dict[str, Dict[str, int]],
    top_n: int = 20,
) -> None:
    """Plot top-N most frequent phonemes."""
    top_items = sorted(
        stats.items(),
        key=lambda item: item[1]["total"],
        reverse=True,
    )[:top_n]

    phonemes = [item[0] for item in top_items]
    totals = [item[1]["total"] for item in top_items]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(range(len(phonemes)), totals)
    ax.set_ylabel("Count")
    ax.set_title(f"Most Frequent Phonemes (Top {top_n})")
    ax.set_xticks(range(len(phonemes)))
    ax.set_xticklabels(phonemes, rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "phoneme_counts_top.png", dpi=200)
    plt.close(fig)


def plot_both_wrong_phonemes(
    rows: List[Dict[str, str]],
    stats: Dict[str, Dict[str, int]],
    top_n: int = 12,
    min_count: int = 5,
) -> None:
    """Plot phonemes most often marked wrong by both methods."""
    both_wrong_counts: Dict[str, int] = {}

    for row in rows:
        phoneme = (row.get("phoneme") or "").strip().upper()
        if not phoneme:
            continue

        try:
            equal_eval = int(row["equal_partition_evaluation"])
            mfa_eval = int(row["mfa_evaluation"])
        except (KeyError, ValueError):
            continue

        if equal_eval == 0 and mfa_eval == 0:
            both_wrong_counts[phoneme] = both_wrong_counts.get(phoneme, 0) + 1

    items = []
    for phoneme, wrong_count in both_wrong_counts.items():
        total = stats.get(phoneme, {}).get("total", 0)
        if total < min_count:
            continue

        wrong_rate = _safe_div(wrong_count, total) * 100
        items.append((phoneme, wrong_count, wrong_rate))

    items.sort(key=lambda t: t[2], reverse=True)
    items = items[:top_n]

    if not items:
        return

    phonemes = [t[0] for t in items]
    rates = [t[2] for t in items]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(range(len(phonemes)), rates)
    ax.set_ylabel("Both-wrong rate (%)")
    ax.set_title("Phonemes Missed by Both Methods")
    ax.set_xticks(range(len(phonemes)))
    ax.set_xticklabels(phonemes, rotation=45, ha="right")

    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{rate:.1f}%",
            ha="center",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "both_wrong_phonemes.png", dpi=200)
    plt.close(fig)


def plot_position_trend(rows: List[Dict[str, str]]) -> None:
    """
    Plot "both wrong" rate by position index extracted from frame_name.

    Expected frame_name format example:
      "03_bbal8p_..."  -> position index = 3
    """
    total_by_position: Dict[int, int] = {}
    both_wrong_by_position: Dict[int, int] = {}

    for row in rows:
        frame_name = row.get("frame_name") or ""
        try:
            position_index = int(frame_name.split("_")[0])
        except (ValueError, IndexError):
            continue

        try:
            equal_eval = int(row["equal_partition_evaluation"])
            mfa_eval = int(row["mfa_evaluation"])
        except (KeyError, ValueError):
            continue

        total_by_position[position_index] = total_by_position.get(
            position_index, 0
        ) + 1
        if equal_eval == 0 and mfa_eval == 0:
            both_wrong_by_position[position_index] = (
                both_wrong_by_position.get(position_index, 0) + 1
            )

    positions = sorted(total_by_position.keys())
    if not positions:
        return

    rates = [
        _safe_div(
            both_wrong_by_position.get(pos, 0),
            total_by_position[pos],
        )
        * 100
        for pos in positions
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(positions, rates, marker="o")
    ax.set_xlabel("Phoneme position index (from frame_name)")
    ax.set_ylabel("Both-wrong rate (%)")
    ax.set_title("Position Trend: Both Methods Incorrect")
    ax.set_xticks(positions)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "both_wrong_position_trend.png", dpi=200)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_ratings()
    overall = summarize_overall(rows)
    per_phoneme_stats = summarize_by_phoneme(rows)

    plot_overall_accuracy(overall)
    plot_per_phoneme_accuracy(per_phoneme_stats)
    plot_accuracy_gap_phonemes(per_phoneme_stats)
    plot_counts(per_phoneme_stats)
    plot_both_wrong_phonemes(rows, per_phoneme_stats)
    plot_position_trend(rows)

    print(f"Saved plots to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
