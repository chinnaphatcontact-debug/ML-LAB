import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

RESULT_PATH = OUTPUT_DIR / "epoch_comparison.json"


# ============================================================
# LOAD RESULTS
# ============================================================

with open(RESULT_PATH, "r", encoding="utf-8") as file:
    results = json.load(file)


epochs = []
best_val_accuracy = []
test_accuracy = []
best_epoch = []

for key, result in results.items():

    epochs.append(result["epochs"])
    best_val_accuracy.append(
        result["best_val_accuracy"] * 100
    )
    test_accuracy.append(
        result["test_accuracy"] * 100
    )
    best_epoch.append(
        result["best_epoch"]
    )


# ============================================================
# FIGURE 1
# VALIDATION vs TEST ACCURACY
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    best_val_accuracy,
    marker="o",
    linewidth=2,
    label="Best Validation Accuracy"
)

plt.plot(
    epochs,
    test_accuracy,
    marker="o",
    linewidth=2,
    label="Test Accuracy"
)

for x, y in zip(epochs, best_val_accuracy):
    plt.annotate(
        f"{y:.2f}%",
        (x, y),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center"
    )

for x, y in zip(epochs, test_accuracy):
    plt.annotate(
        f"{y:.2f}%",
        (x, y),
        xytext=(0, -18),
        textcoords="offset points",
        ha="center"
    )

plt.title(
    "CNN-2 Epoch Comparison - Validation and Test Accuracy",
    fontsize=15
)

plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy (%)")

plt.xticks(epochs)
plt.ylim(0, 100)

plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()

accuracy_path = OUTPUT_DIR / "epoch_comparison_accuracy.png"

plt.savefig(
    accuracy_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 2
# BEST EPOCH
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    epochs,
    best_val_accuracy,
    width=1.5
)

for x, y, best in zip(
    epochs,
    best_val_accuracy,
    best_epoch
):

    plt.text(
        x,
        y + 1,
        f"{y:.2f}%\nBest: Epoch {best}",
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.title(
    "CNN-2 Best Validation Accuracy by Training Epochs",
    fontsize=15
)

plt.xlabel("Maximum Number of Epochs")
plt.ylabel("Best Validation Accuracy (%)")

plt.xticks(epochs)
plt.ylim(0, 100)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

val_path = OUTPUT_DIR / "epoch_comparison_validation.png"

plt.savefig(
    val_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SUMMARY
# ============================================================

best_index = int(
    np.argmax(test_accuracy)
)

print("=" * 70)
print("EPOCH COMPARISON")
print("=" * 70)

print(
    f"Best Epoch Configuration : "
    f"{epochs[best_index]} Epochs"
)

print(
    f"Best Test Accuracy       : "
    f"{test_accuracy[best_index]:.2f}%"
)

print(
    f"Best Validation Accuracy : "
    f"{best_val_accuracy[best_index]:.2f}%"
)

print()
print("Saved plots:")
print(accuracy_path)
print(val_path)

print("=" * 70)