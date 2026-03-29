import json
import matplotlib.pyplot as plt

with open("training/results_bozkurt_viseme/training_history.json") as f:
    history = json.load(f)

epochs_data = history["epochs"]
epochs = [e["epoch"] for e in epochs_data]
train_acc = [e["train_accuracy"] for e in epochs_data]
val_acc = [e["val_accuracy"] for e in epochs_data]
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(epochs, train_acc, color="#e74c3c", linewidth=2.0,
        label="Training Accuracy")
ax.plot(epochs, val_acc, color="#3498db", linewidth=2.0,
        label="Validation Accuracy")

TEST_ACCURACY = 83.77
ax.axhline(
    y=TEST_ACCURACY,
    color="#2ecc71",
    linewidth=1.5,
    linestyle="--",
    label=f"Test Accuracy (held-out 15%): {TEST_ACCURACY}%",
)

# Annotate the best validation point
best_epoch = max(epochs_data, key=lambda epoch: epoch["val_accuracy"])
best_label = (
    f"Best val: {best_epoch['val_accuracy']:.2f}%\n"
    f"(epoch {best_epoch['epoch']})"
)

ax.annotate(
    best_label,
    xy=(best_epoch["epoch"], best_epoch["val_accuracy"]),
    xytext=(
        best_epoch["epoch"] - 12,
        best_epoch["val_accuracy"] - 12,
    ),
    arrowprops=dict(arrowstyle="->", color="#555"),
    fontsize=9,
    color="#1a1a2e",
)

# Annotate train accuracy at epoch 60
final_train = epochs_data[-1]
ax.annotate(
    f"Train: {final_train['train_accuracy']:.2f}%\n(epoch 60)",
    xy=(60, final_train["train_accuracy"]),
    xytext=(48, final_train["train_accuracy"] - 11),
    arrowprops=dict(arrowstyle="->", color="#555"),
    fontsize=9,
    color="#1a1a2e",
)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title(
    "Viseme 3D-CNN — Training & Validation Accuracy over 60 Epochs",
    fontsize=13,
    fontweight="bold",
    color="#1a1a2e",
)
ax.set_xlim(1, 60)
ax.set_ylim(0, 105)
ax.legend(fontsize=10, loc="upper left")
ax.grid(True, linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
out_path = "visuals/viseme_eval/learning_curve.png"
plt.savefig(out_path, dpi=180, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")
