"""
Compute accuracy, 95% confidence intervals and a p-value comparing
equal-partition vs MFA alignment.
Results are based on manual ratings with 0/1 (false/true).
Statistical comparison is performed using a two-proportion z-test.
"""

import csv
import math
from pathlib import Path


RATINGS_FILE = Path(__file__).parent / "alignment_manual_ratings.csv"

# Significance level for hypothesis testing
ALPHA = 0.05


def compute_proportion_confidence_interval(
    num_successes: int,
    num_samples: int,
    z_score: float = 1.96,
):
    if num_samples == 0:
        return 0.0, 0.0, 0.0

    proportion = num_successes / num_samples
    standard_error = math.sqrt(
        proportion * (1 - proportion) / num_samples
    )

    lower_bound = max(0.0, proportion - z_score * standard_error)
    upper_bound = min(1.0, proportion + z_score * standard_error)

    return proportion, lower_bound, upper_bound


def two_proportion_z_test(
    successes_equal: int,
    samples_equal: int,
    successes_mfa: int,
    samples_mfa: int,
):
    if samples_equal == 0 or samples_mfa == 0:
        return 0.0, 1.0

    proportion_equal = successes_equal / samples_equal
    proportion_mfa = successes_mfa / samples_mfa

    pooled_proportion = (
        successes_equal + successes_mfa
    ) / (samples_equal + samples_mfa)

    standard_error = math.sqrt(
        pooled_proportion
        * (1 - pooled_proportion)
        * (1 / samples_equal + 1 / samples_mfa)
    )

    if standard_error == 0:
        return 0.0, 1.0

    z_value = (
        proportion_equal - proportion_mfa
    ) / standard_error

    def normal_cdf(z_val: float) -> float:
        return 0.5 * (1 + math.erf(z_val / math.sqrt(2)))

    p_value = 2 * (1 - normal_cdf(abs(z_value)))

    return z_value, p_value


def main() -> None:

    if not RATINGS_FILE.exists():
        raise FileNotFoundError(
            f"Manual ratings file not found: {RATINGS_FILE}\n"
            "Required columns:\n"
            "video, frame_name, phoneme,\n"
            "equal_partition_evaluation, mfa_evaluation"
        )

    # Accuracy counters
    equal_partition_correct = 0
    mfa_correct = 0
    total_samples = 0

    # Comparison categories
    both_correct = 0
    only_equal_correct = 0
    only_mfa_correct = 0
    neither_correct = 0

    # Read CSV file
    with open(RATINGS_FILE, "r", encoding="utf-8") as file_handle:
        reader = csv.DictReader(file_handle)

        for row in reader:
            try:
                equal_eval = int(row["equal_partition_evaluation"])
                mfa_eval = int(row["mfa_evaluation"])
            except (KeyError, ValueError):
                continue

            total_samples += 1
            equal_partition_correct += equal_eval
            mfa_correct += mfa_eval

            if equal_eval == 1 and mfa_eval == 1:
                both_correct += 1
            elif equal_eval == 1 and mfa_eval == 0:
                only_equal_correct += 1
            elif equal_eval == 0 and mfa_eval == 1:
                only_mfa_correct += 1
            else:
                neither_correct += 1

    print("=" * 60)
    print("Manual Alignment Evaluation Statistics")
    print("=" * 60)

    print(f"Total evaluated samples: {total_samples}\n")

    # Confidence intervals
    (
        equal_accuracy,
        equal_ci_low,
        equal_ci_high,
    ) = compute_proportion_confidence_interval(
        equal_partition_correct,
        total_samples,
    )

    (
        mfa_accuracy,
        mfa_ci_low,
        mfa_ci_high,
    ) = compute_proportion_confidence_interval(
        mfa_correct,
        total_samples,
    )

    print("Accuracy Results (95% Confidence Interval)")
    print("-" * 60)

    print(
        f"Equal Partition : {equal_accuracy*100:5.1f}% "
        f"(CI: {equal_ci_low*100:5.1f}% – {equal_ci_high*100:5.1f}%)"
    )

    print(
        f"MFA Alignment   : {mfa_accuracy*100:5.1f}% "
        f"(CI: {mfa_ci_low*100:5.1f}% – {mfa_ci_high*100:5.1f}%)"
    )

    print("\nComparison Breakdown")
    print("-" * 60)

    print(f"Both correct        : {both_correct}")
    print(f"Only equal correct  : {only_equal_correct}")
    print(f"Only MFA correct    : {only_mfa_correct}")
    print(f"Neither correct     : {neither_correct}")

    print("\nHypothesis Testing (Two-Proportion Z-Test)")
    print("-" * 60)

    print("H0 (Null Hypothesis):")
    print("  Equal-partition and MFA have the same accuracy.")

    print("\nH1 (Alternative Hypothesis):")
    print("  MFA has higher accuracy than equal-partition.")

    print(
        f"\nSignificance Level:"
        f"  α (alpha) = {ALPHA}"
        " (5% probability of false positive)"
    )

    z_value, p_value = two_proportion_z_test(
        equal_partition_correct,
        total_samples,
        mfa_correct,
        total_samples,
    )

    print("\nTest Statistics")
    print("-" * 60)

    print(f"Z-score : {z_value:.3f}")
    print(f"P-value : {p_value:.5f}")

    print("\nDecision")
    print("-" * 60)

    if p_value < ALPHA:
        print(
            f"Reject H0 at α = {ALPHA}. "
            "MFA performs significantly better."
        )
    else:
        print(
            f"Fail to reject H0 at α = {ALPHA}. "
            "No significant difference detected."
        )

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
