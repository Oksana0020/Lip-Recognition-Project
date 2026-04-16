"""
Generate visual comparisons of the viseme-level and
word-level 3D-CNN models.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT_DIRECTORY = Path(__file__).parent.parent
OUTPUT_DIRECTORY = PROJECT_ROOT_DIRECTORY / "validation" / "results_comparison"
VISEME_FINAL_METRICS_PATH = (
    PROJECT_ROOT_DIRECTORY
    / "training"
    / "results_bozkurt_viseme"
    / "final_test_metrics.json")
WORD_FINAL_METRICS_PATH = (
    PROJECT_ROOT_DIRECTORY
    / "training"
    / "results_words"
    / "final_test_metrics.json")
VISEME_TRAINING_HISTORY_PATH = (
    PROJECT_ROOT_DIRECTORY
    / "training"
    / "results_bozkurt_viseme"
    / "training_history.json")
WORD_TRAINING_HISTORY_PATH = (
    PROJECT_ROOT_DIRECTORY
    / "training"
    / "results_words"
    / "training_history.json")
VISEME_COLOR = "#2980b9"
WORD_COLOR = "#e74c3c"


def load_json_file(json_file_path: Path) -> Any:
    """Load and return JSON content from a file"""
    with json_file_path.open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


def save_figure(figure: plt.Figure, output_file_path: Path) -> None:
    """Save a matplotlib figure and close it"""
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_file_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {output_file_path}")


def style_plot_axes(plot_axes: plt.Axes) -> None:
    """Apply shared styling to axes."""
    plot_axes.grid(axis="y", linestyle="--", alpha=0.4)
    plot_axes.spines["top"].set_visible(False)
    plot_axes.spines["right"].set_visible(False)


def extract_test_metrics(metrics_file_path: Path) -> dict[str, float]:
    """Extract the test_metrics dictionary from a metrics JSON file."""
    metrics_json = load_json_file(metrics_file_path)
    return metrics_json["test_metrics"]


def extract_training_epochs(
    training_history_file_path: Path,
) -> list[dict[str, Any]]:
    """Extract epoch history from a training history JSON file"""
    training_history_json = load_json_file(training_history_file_path)
    if isinstance(training_history_json, dict):
        return training_history_json.get("epochs", [])
    return training_history_json


def add_value_labels_to_bars(
    plot_axes: plt.Axes,
    bar_container,
    label_color: str,
) -> None:
    """Add numeric labels above bars in a bar chart"""
    for bar in bar_container:
        bar_height = bar.get_height()
        plot_axes.text(
            bar.get_x() + bar.get_width() / 2,
            bar_height + 0.8,
            f"{bar_height:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=label_color,
            fontweight="bold")


def plot_metrics_comparison(output_file_path: Path) -> None:
    """Plot grouped bar chart comparing key metrics of both models."""
    viseme_test_metrics = extract_test_metrics(VISEME_FINAL_METRICS_PATH)
    word_test_metrics = extract_test_metrics(WORD_FINAL_METRICS_PATH)
    metric_names = [
        "Test Accuracy (%)",
        "Best Val Accuracy (%)",
        "Test Loss"]
    viseme_metric_values = [
        viseme_test_metrics["accuracy"],
        viseme_test_metrics["best_validation_accuracy"],
        viseme_test_metrics["loss"]]
    word_metric_values = [
        word_test_metrics["accuracy"],
        word_test_metrics["best_validation_accuracy"],
        word_test_metrics["loss"]]
    metric_positions = np.arange(len(metric_names))
    bar_width = 0.32
    figure, plot_axes = plt.subplots(figsize=(10, 6))
    figure.patch.set_facecolor("white")
    viseme_bars = plot_axes.bar(
        metric_positions - bar_width / 2,
        viseme_metric_values,
        bar_width,
        label="Viseme model (13 classes)",
        color=VISEME_COLOR,
        edgecolor="white",
        linewidth=0.8)
    word_bars = plot_axes.bar(
        metric_positions + bar_width / 2,
        word_metric_values,
        bar_width,
        label="Word model (51 classes)",
        color=WORD_COLOR,
        edgecolor="white",
        linewidth=0.8)

    add_value_labels_to_bars(plot_axes, viseme_bars, VISEME_COLOR)
    add_value_labels_to_bars(plot_axes, word_bars, WORD_COLOR)
    plot_axes.set_xticks(metric_positions)
    plot_axes.set_xticklabels(metric_names, fontsize=11)
    plot_axes.set_ylabel("Value", fontsize=11)
    plot_axes.set_title(
        "Viseme vs Word 3D-CNN — Key Metrics Comparison",
        fontsize=13,
        fontweight="bold",
        pad=12)
    plot_axes.legend(fontsize=10)
    plot_axes.set_ylim(
        0,
        max(max(viseme_metric_values), max(word_metric_values)) * 1.18)
    style_plot_axes(plot_axes)
    save_figure(figure, output_file_path)


def plot_training_curves(output_file_path: Path) -> None:
    """Plot validation accuracy curves for viseme and word models."""
    viseme_training_epochs = extract_training_epochs(
        VISEME_TRAINING_HISTORY_PATH)
    word_training_epochs = extract_training_epochs(WORD_TRAINING_HISTORY_PATH)
    viseme_test_accuracy = extract_test_metrics(
        VISEME_FINAL_METRICS_PATH
    )["accuracy"]
    word_test_accuracy = extract_test_metrics(
        WORD_FINAL_METRICS_PATH)["accuracy"]
    viseme_epoch_numbers = [epoch["epoch"] for epoch in viseme_training_epochs]
    viseme_validation_accuracies = [
        epoch["val_accuracy"] for epoch in viseme_training_epochs]
    word_epoch_numbers = [epoch["epoch"] for epoch in word_training_epochs]
    word_validation_accuracies = [
        epoch["val_accuracy"] for epoch in word_training_epochs]
    figure, plot_axes = plt.subplots(figsize=(12, 6))
    figure.patch.set_facecolor("white")
    plot_axes.plot(
        viseme_epoch_numbers,
        viseme_validation_accuracies,
        color=VISEME_COLOR,
        linewidth=2.2,
        label="Viseme — Validation Accuracy")
    plot_axes.plot(
        word_epoch_numbers,
        word_validation_accuracies,
        color=WORD_COLOR,
        linewidth=2.2,
        label="Word — Validation Accuracy")
    plot_axes.axhline(
        viseme_test_accuracy,
        color=VISEME_COLOR,
        linewidth=1.4,
        linestyle="--",
        label=f"Viseme test accuracy: {viseme_test_accuracy:.2f}%")
    plot_axes.axhline(
        word_test_accuracy,
        color=WORD_COLOR,
        linewidth=1.4,
        linestyle="--",
        label=f"Word test accuracy: {word_test_accuracy:.2f}%")
    plot_axes.set_title(
        "Viseme vs Word 3D-CNN — Validation Accuracy over Training",
        fontsize=13,
        fontweight="bold",
        pad=12)
    plot_axes.set_xlabel("Epoch", fontsize=11)
    plot_axes.set_ylabel("Validation Accuracy (%)", fontsize=11)
    plot_axes.set_ylim(0, 100)
    plot_axes.set_xlim(
        1,
        max(max(viseme_epoch_numbers), max(word_epoch_numbers)))
    plot_axes.legend(fontsize=10, loc="lower right")
    plot_axes.grid(True, linestyle="--", alpha=0.4)
    plot_axes.spines["top"].set_visible(False)
    plot_axes.spines["right"].set_visible(False)
    save_figure(figure, output_file_path)


def main() -> None:
    """Generate all model comparison plots."""
    plot_metrics_comparison(
        OUTPUT_DIRECTORY / "compare_metrics_bar.png")
    plot_training_curves(
        OUTPUT_DIRECTORY / "compare_training_curves.png")


if __name__ == "__main__":
    main()
