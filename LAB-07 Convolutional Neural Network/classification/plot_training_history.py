
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).parent / "outputs"
HISTORY_FILE = OUTPUT_DIR / "cnn_2_history.json"

with open(HISTORY_FILE, "r", encoding="utf-8") as f:
    history = json.load(f)

epochs = np.arange(1, len(history["accuracy"]) + 1)

# Consistent visual style
plt.rcParams.update({
    "figure.figsize": (9, 5.5),
    "figure.dpi": 140,
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "lines.linewidth": 2.5,
    "lines.markersize": 6,
})

# Accuracy
fig, ax = plt.subplots()
ax.plot(
    epochs, history["accuracy"],
    marker="o", label="Training Accuracy",
    color="#2878B5"
)
ax.plot(
    epochs, history["val_accuracy"],
    marker="s", label="Validation Accuracy",
    color="#E87522"
)

ax.set(
    title="CNN-2 Training and Validation Accuracy",
    xlabel="Epoch",
    ylabel="Accuracy",
    ylim=(0, 1.05),
    xticks=epochs,
)
ax.grid(True, linestyle="--", alpha=0.3)
ax.legend(frameon=True)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "cnn_2_accuracy.png", bbox_inches="tight")
plt.close(fig)

# Loss
fig, ax = plt.subplots()
ax.plot(
    epochs, history["loss"],
    marker="o", label="Training Loss",
    color="#2878B5"
)
ax.plot(
    epochs, history["val_loss"],
    marker="s", label="Validation Loss",
    color="#E87522"
)

max_loss = max(max(history["loss"]), max(history["val_loss"]))
ax.set(
    title="CNN-2 Training and Validation Loss",
    xlabel="Epoch",
    ylabel="Loss",
    ylim=(0, max_loss * 1.12),
    xticks=epochs,
)
ax.grid(True, linestyle="--", alpha=0.3)
ax.legend(frameon=True)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "cnn_2_loss.png", bbox_inches="tight")
plt.close(fig)

print("Saved: cnn_2_accuracy.png")
print("Saved: cnn_2_loss.png")
