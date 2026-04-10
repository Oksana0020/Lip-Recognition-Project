"""
Generate summary images for word-level 3D-CNN training:
accuracy curve and training summary dashboard
"""

from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).parent.parent
HISTORY_JSON = PROJECT_ROOT / "training/results_words/training_history.json"
METRICS_JSON = PROJECT_ROOT / "training/results_words/final_test_metrics.json"
OUTPUT_DIR = PROJECT_ROOT / "visuals/word"


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_accuracy(history: dict, test_acc: float, out: Path) -> None:
    rows = history["epochs"]
    epochs = [r["epoch"] for r in rows]
    train = [r["train_accuracy"] for r in rows]
    val = [r["val_accuracy"] for r in rows]
    best_val = max(val)
    best_epoch = epochs[val.index(best_val)]
    final_train = train[-1]
    final_epoch = epochs[-1]
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("white")
    ax.plot(epochs, train, color="#e74c3c", lw=2.2, label="Train")
    ax.plot(epochs, val, color="#2980b9", lw=2.2, label="Val")
    ax.axhline(
        test_acc,
        color="#27ae60",
        lw=1.8,
        ls="--",
        label=f"Test: {test_acc:.2f}%")

    ax.annotate(
        f"{final_train:.2f}% (ep {final_epoch})",
        (final_epoch, final_train),
        (-70, -28),
        textcoords="offset points",
        fontsize=9,
        color="#e74c3c",
        arrowprops=dict(arrowstyle="->", color="#e74c3c"))

    ax.annotate(
        f"{best_val:.2f}% (ep {best_epoch})",
        (best_epoch, best_val),
        (10, -30),
        textcoords="offset points",
        fontsize=9,
        color="#2980b9",
        arrowprops=dict(arrowstyle="->", color="#2980b9"))

    ax.set(title="Training vs Validation Accuracy", xlabel="Epoch",
           ylabel="Accuracy (%)", ylim=(0, 100))
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_figure(fig, out)


def plot_dashboard(history: dict, metrics: dict, out: Path) -> None:
    train_final = history["epochs"][-1]["train_accuracy"]
    m = metrics["test_metrics"]
    cards = [
        (f"{train_final:.2f}%", "Train", "#fdecea", "#c0392b"),
        (f"{m['best_validation_accuracy']:.2f}%", "Best Val",
         "#eaf4fb", "#2471a3"),
        (f"{m['accuracy']:.2f}%", "Test", "#eafaf1", "#1e8449"),
        (f"{m['loss']:.4f}", "Loss", "#fef5e7", "#d68910"),
        (f"{len(metrics['word_classes'])}", "Classes", "#f5eef8", "#7d3c98"),
        (f"{59370:,}", "Samples", "#e8f8f5", "#1a9e80")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    fig.suptitle("Training Summary", fontsize=14, weight="bold")
    for ax, (val, label, bg, fg) in zip(axes.flat, cards):
        ax.set_facecolor(bg)
        for s in ax.spines.values():
            s.set_edgecolor(fg)
            s.set_linewidth(2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.6, val, ha="center", va="center",
                transform=ax.transAxes, fontsize=20, weight="bold", color=fg)
        ax.text(0.5, 0.25, label, ha="center", va="center",
                transform=ax.transAxes, fontsize=10)
    fig.tight_layout()
    save_figure(fig, out)


def main() -> None:
    with open(HISTORY_JSON, encoding="utf-8") as f:
        history = json.load(f)
    with open(METRICS_JSON, encoding="utf-8") as f:
        metrics = json.load(f)
    plot_accuracy(
        history,
        metrics["test_metrics"]["accuracy"],
        OUTPUT_DIR / "word_training_accuracy_curve.png")
    plot_dashboard(
        history,
        metrics,
        OUTPUT_DIR / "word_training_summary.png")


if __name__ == "__main__":
    main()
