import json

history_path = (
    "training/results_bozkurt_viseme/"
    "training_history.json"
)

with open(history_path, "r", encoding="utf-8") as file_handle:
    history = json.load(file_handle)
target_epochs = [1, 2, 3, 5, 6, 8, 10, 13, 18, 26, 33, 50, 60]
print(f"Total epochs: {len(history)}")

for epoch_data in history:
    if epoch_data["epoch"] in target_epochs:
        print(
            f"Ep{epoch_data['epoch']}: "
            f"train={round(epoch_data['train_accuracy'], 2)}% "
            f"val={round(epoch_data['val_accuracy'], 2)}%"
        )

print("ALL EPOCHS")

for epoch_data in history:
    print(
        f"Ep{epoch_data['epoch']}: "
        f"train={round(epoch_data['train_accuracy'], 2)}% "
        f"val={round(epoch_data['val_accuracy'], 2)}%"
    )
