import json
from pathlib import Path

import matplotlib.pyplot as plt


# =========================
# Configuration
# =========================

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

HISTORY_FILE = OUTPUT_DIR / "cnn_2_history.json"


# =========================
# Load History
# =========================

with open(
    HISTORY_FILE,
    "r",
    encoding="utf-8"
) as file:

    history = json.load(file)


epochs = range(1, len(history["accuracy"]) + 1)


# =========================
# Accuracy Plot
# =========================

plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    history["accuracy"],
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    epochs,
    history["val_accuracy"],
    marker="o",
    label="Validation Accuracy"
)

plt.title("CNN-2 Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.xticks(list(epochs))
plt.ylim(0, 1.05)

plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()

accuracy_path = OUTPUT_DIR / "cnn_2_accuracy.png"

plt.savefig(
    accuracy_path,
    dpi=200
)

plt.show()


# =========================
# Loss Plot
# =========================

plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    history["loss"],
    marker="o",
    label="Training Loss"
)

plt.plot(
    epochs,
    history["val_loss"],
    marker="o",
    label="Validation Loss"
)

plt.title("CNN-2 Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.xticks(list(epochs))

plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()

loss_path = OUTPUT_DIR / "cnn_2_loss.png"

plt.savefig(
    loss_path,
    dpi=200
)

plt.show()


# =========================
# Best Validation Accuracy
# =========================

best_val_accuracy = max(history["val_accuracy"])

best_epoch = (
    history["val_accuracy"].index(best_val_accuracy) + 1
)

print("=" * 60)
print("TRAINING HISTORY ANALYSIS")
print("=" * 60)

print(
    f"Best Validation Accuracy : "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Best Epoch               : "
    f"{best_epoch}"
)

print("\nSaved plots:")

print(accuracy_path)
print(loss_path)

print("=" * 60)