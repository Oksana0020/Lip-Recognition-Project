"""
Plot viseme per-class accuracy analysis charts.
Script saves horizontal bar chart, articulation group summary chart,
sample count vs accuracy scatter plot
"""

from __future__ import annotations
from pathlib import Path
import matplotlib.patches as matplotlib_patches
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT_DIRECTORY = Path(__file__).parent.parent
OUTPUT_DIRECTORY = PROJECT_ROOT_DIRECTORY / "visuals" / "viseme_eval"
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
VISEME_CLASS_DATA = [
    ("V2", "ay, ah", "Open mouth", 7725, 7478, 96.80),
    ("V3", "ey, eh, ae", "Mid-open", 7180, 6577, 91.60),
    ("V6", "uw, uh, w", "Rounded lips", 6727, 6413, 95.33),
    ("V7", "ao, aa, oy, ow", "Back rounded", 991, 965, 97.38),
    ("V8", "aw", "Diphthong", 2471, 2447, 99.03),
    ("V9", "g, hh, k, ng", "Velar", 4941, 4400, 89.05),
    ("V10", "r", "R-colored", 8315, 7660, 92.12),
    ("V11", "l, d, n, t", "Alveolar", 10358, 9228, 89.09),
    ("V12", "s, z", "Fricatives", 12164, 11156, 91.71),
    ("V13", "ch, sh, jh, zh", "Post-alveolar", 393, 311, 79.13),
    ("V14", "th, dh", "Dental", 994, 842, 84.71),
    ("V15", "f, v", "Labiodental", 2378, 2187, 91.97),
    ("V16", "m, b, p", "Bilabial", 398, 356, 89.45)]

ARTICULATION_GROUPS = {
    "Lip visible": {
        "ids": {"V2", "V3", "V6", "V7", "V8", "V15", "V16"},
        "color": "#2196F3"},
    "Internal": {
        "ids": {"V9", "V10", "V11", "V12", "V13", "V14"},
        "color": "#FF5722"}}


def get_group_color(viseme_class_id: str) -> str:
    """Return the display color for a viseme class."""
    for group_definition in ARTICULATION_GROUPS.values():
        if viseme_class_id in group_definition["ids"]:
            return group_definition["color"]
    return "#9E9E9E"


def plot_accuracy_bar_chart() -> None:
    """Plot per-class accuracy as an annotated horizontal bar chart."""
    sorted_viseme_rows = sorted(
        VISEME_CLASS_DATA,
        key=lambda viseme_row: viseme_row[5],
        reverse=True)
    y_axis_labels = [
        f"{viseme_class_id} ({description})\n/{phoneme_list}/"
        for viseme_class_id, phoneme_list,
        description, *_ in sorted_viseme_rows]
    accuracy_values = [
        accuracy_percentage for *_, accuracy_percentage in sorted_viseme_rows]
    total_sample_counts = [
        total_samples for _, _, _, total_samples, *_ in sorted_viseme_rows]
    bar_colors = [
        get_group_color(viseme_class_id)
        for viseme_class_id, *_ in sorted_viseme_rows]
    mean_accuracy = float(np.mean(accuracy_values))
    figure, axes = plt.subplots(figsize=(12, 8))
    bar_containers = axes.barh(
        y_axis_labels,
        accuracy_values,
        color=bar_colors)

    for bar, accuracy_percentage, total_samples in zip(
            bar_containers,
            accuracy_values,
            total_sample_counts):
        bar_center_y = bar.get_y() + bar.get_height() / 2
        axes.text(
            accuracy_percentage + 0.2,
            bar_center_y,
            f"{accuracy_percentage:.1f}%",
            va="center")
        axes.text(
            1.5,
            bar_center_y,
            f"n={total_samples}",
            va="center",
            color="white")

    axes.axvline(mean_accuracy, linestyle="--", color="#4CAF50")
    axes.set_xlim(0, 103)
    axes.set_xlabel("Accuracy (%)")
    axes.set_title("Viseme Per-Class Accuracy")
    axes.invert_yaxis()
    axes.grid(True, axis="x", linestyle=":")
    legend_handles = [
        matplotlib_patches.Patch(
            color=group_definition["color"],
            label=group_name)
        for group_name, group_definition in ARTICULATION_GROUPS.items()]
    axes.legend(handles=legend_handles)
    output_file_path = OUTPUT_DIRECTORY / "viseme_accuracy.png"
    plt.tight_layout()
    plt.savefig(output_file_path, dpi=150)
    plt.close(figure)
    print(f"Saved: {output_file_path}")


def plot_sample_count_accuracy_scatter() -> None:
    """Plot sample count against accuracy for each viseme class."""
    viseme_class_ids = [
        viseme_class_id for viseme_class_id, *_ in VISEME_CLASS_DATA]
    total_sample_counts = [
        total_samples for _, _, _, total_samples, *_ in VISEME_CLASS_DATA]
    accuracy_values = [
        accuracy_percentage for *_, accuracy_percentage in VISEME_CLASS_DATA]
    point_colors = [
        get_group_color(viseme_class_id)
        for viseme_class_id in viseme_class_ids]
    mean_accuracy = float(np.mean(accuracy_values))
    figure, axes = plt.subplots(figsize=(10, 6))
    axes.scatter(
        total_sample_counts,
        accuracy_values,
        c=point_colors,
        s=100)

    for viseme_class_id, total_samples, accuracy_percentage in zip(
            viseme_class_ids,
            total_sample_counts,
            accuracy_values):
        axes.text(
            total_samples + 100,
            accuracy_percentage,
            viseme_class_id,
            fontsize=8)

    axes.axhline(mean_accuracy, linestyle="--", color="#4CAF50")
    axes.set_xlabel("Sample Count")
    axes.set_ylabel("Accuracy (%)")
    axes.set_title("Sample Count vs Accuracy")
    axes.grid(True, linestyle=":")
    output_file_path = OUTPUT_DIRECTORY / "viseme_scatter.png"
    plt.tight_layout()
    plt.savefig(output_file_path, dpi=150)
    plt.close(figure)
    print(f"Saved: {output_file_path}")


def plot_articulation_group_summary() -> None:
    """Plot mean accuracy for each articulation group."""
    articulation_group_accuracies = {
        group_name: []
        for group_name in ARTICULATION_GROUPS}

    for (
        viseme_class_id,
        phoneme_list,
        description,
        total_samples,
        correct_samples,
        accuracy_percentage,
    ) in VISEME_CLASS_DATA:
        for group_name, group_definition in ARTICULATION_GROUPS.items():
            if viseme_class_id in group_definition["ids"]:
                articulation_group_accuracies[group_name].append(
                    accuracy_percentage)

    group_names = list(articulation_group_accuracies.keys())
    mean_group_accuracies = [
        float(np.mean(group_accuracy_values))
        for group_accuracy_values in articulation_group_accuracies.values()]
    bar_colors = [
        ARTICULATION_GROUPS[group_name]["color"]
        for group_name in group_names]
    x_axis_positions = np.arange(len(group_names))
    figure, axes = plt.subplots(figsize=(8, 5))
    axes.bar(
        x_axis_positions,
        mean_group_accuracies,
        color=bar_colors)

    for bar_index, mean_accuracy in enumerate(mean_group_accuracies):
        axes.text(
            bar_index,
            mean_accuracy + 0.5,
            f"{mean_accuracy:.1f}%",
            ha="center")

    axes.set_xticks(x_axis_positions)
    axes.set_xticklabels(group_names)
    axes.set_ylabel("Accuracy (%)")
    axes.set_title("Accuracy by Articulation Group")
    axes.grid(True, axis="y", linestyle=":")
    output_file_path = OUTPUT_DIRECTORY / "viseme_groups.png"
    plt.tight_layout()
    plt.savefig(output_file_path, dpi=150)
    plt.close(figure)
    print(f"Saved: {output_file_path}")


def main() -> None:
    """Generate all viseme class analysis charts."""
    print("Generating viseme charts...")
    plot_accuracy_bar_chart()
    plot_sample_count_accuracy_scatter()
    plot_articulation_group_summary()
    print(f"Done. Saved to {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
