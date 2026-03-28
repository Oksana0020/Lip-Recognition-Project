"""
Generate a summary card figure
showing training, validation and test accuracy for tviseme 3D-CNN model.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

with open("training/results_bozkurt_viseme/training_history.json") as f:
    history = json.load(f)

with open("training/results_bozkurt_viseme/final_test_metrics.json") as f:
    test_metrics = json.load(f)

last_epoch = history["epochs"][-1]
train_acc = last_epoch["train_accuracy"]           
val_acc = test_metrics["test_metrics"]["best_validation_accuracy"]   
test_acc = test_metrics["test_metrics"]["accuracy"]                  
test_loss = test_metrics["test_metrics"]["loss"]                     
num_classes = len(test_metrics["viseme_classes"])                   
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

# title
ax.text(
    5, 9.4,
    "Viseme 3D-CNN — Final Training Summary",
    ha="center", va="center",
    fontsize=15, fontweight="bold", color="#1a1a2e",
)

# divider
ax.plot([0.5, 9.5], [8.85, 8.85], color="#4a90d9", linewidth=2)

metrics = [
    ("Train Accuracy (Ep 60)",   f"{train_acc:.2f} %",  "#e74c3c"),
    ("Best Val Accuracy",        f"{val_acc:.2f} %",    "#3498db"),
    ("Test Accuracy",            f"{test_acc:.2f} %",   "#2ecc71"),
    ("Test Loss",                f"{test_loss:.4f}",    "#e67e22"),
    ("Viseme Classes",           str(num_classes),      "#9b59b6"),
    ("Total Dataset Samples",    "65,035",              "#1abc9c"),
]

CARD_W = 2.95
GAP = 0.25
MARGIN = 0.325
cx = [MARGIN + CARD_W / 2 + i * (CARD_W + GAP) for i in range(3)]

positions = [
    (cx[0], 7.4), (cx[1], 7.4), (cx[2], 7.4),
    (cx[0], 4.8), (cx[1], 4.8), (cx[2], 4.8),
]

card_widths = [CARD_W] * 6
for (label, value, color), (x, y), cw in zip(metrics, positions, card_widths):
    rect = mpatches.FancyBboxPatch(
        (x - cw / 2, y - 0.85), cw, 1.7,
        boxstyle="round,pad=0.1", linewidth=1.5,
        edgecolor=color, facecolor=color + "22",
    )
    ax.add_patch(rect)
    ax.text(
        x, y + 0.32, value,
        ha="center", va="center",
        fontsize=17, fontweight="bold", color=color,
    )
    ax.text(
        x, y - 0.38, label,
        ha="center", va="center",
        fontsize=9, color="#444444",
    )

# footer
ax.plot([0.5, 9.5], [3.3, 3.3], color="#cccccc", linewidth=0.8)
ax.text(
    5, 2.85,
    "Model: 3D-CNN  |  Checkpoint: bozkurt_viseme_best_model.pth  |  Epoch 60",
    ha="center", va="center", fontsize=7.5, color="#888888",
)
ax.text(
    5, 2.35,
    "Dataset: GRID Corpus  |  Split: 70/15/15 train/val/test  |  Frames: 8  |  Resolution: 64x64",
    ha="center", va="center", fontsize=7.5, color="#888888",
)

plt.tight_layout()
out_path = "visuals/viseme_eval/final_test_metrics_visual.png"
plt.savefig(out_path, dpi=180, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")
