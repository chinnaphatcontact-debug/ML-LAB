import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_CANDIDATES = [
    OUTPUT_DIR / "final_cnn_model.keras",
    OUTPUT_DIR / "cnn_2_model.keras",
]

X_TEST_PATH = OUTPUT_DIR / "X_test.npy"
Y_TEST_PATH = OUTPUT_DIR / "y_test.npy"
CLASSES_PATH = OUTPUT_DIR / "classes.json"

CM_PATH = OUTPUT_DIR / "confusion_matrix.png"
CM_NPY_PATH = OUTPUT_DIR / "confusion_matrix.npy"


# ============================================================
# FIND MODEL
# ============================================================
model_path = next(
    (path for path in MODEL_CANDIDATES if path.exists()),
    None
)

if model_path is None:
    print("=" * 70)
    print("ERROR")
    print("=" * 70)
    print("Model not found.")
    raise SystemExit(1)


# ============================================================
# LOAD DATA
# ============================================================
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

X_test = np.load(X_TEST_PATH)
y_test = np.load(Y_TEST_PATH)

with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    classes = json.load(f)

print(f"Model       : {model_path}")
print(f"Test data   : {X_test.shape}")
print(f"Classes     : {classes}")


# ============================================================
# LOAD MODEL
# ============================================================
print("\n" + "=" * 70)
print("LOADING MODEL")
print("=" * 70)

model = tf.keras.models.load_model(model_path)

print("Model loaded successfully.")


# ============================================================
# PREDICTION
# ============================================================
print("\n" + "=" * 70)
print("PREDICTING TEST SET")
print("=" * 70)

probabilities = model.predict(X_test, verbose=1)

y_pred = np.argmax(probabilities, axis=1)

test_accuracy = np.mean(y_pred == y_test)

print(f"\nTest Accuracy : {test_accuracy:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")


# ============================================================
# BUILD CONFUSION MATRIX WITHOUT SKLEARN
# ============================================================
num_classes = len(classes)

cm = np.zeros(
    (num_classes, num_classes),
    dtype=int
)

for actual, predicted in zip(y_test, y_pred):
    cm[int(actual), int(predicted)] += 1


# ============================================================
# SAVE MATRIX
# ============================================================
np.save(CM_NPY_PATH, cm)


# ============================================================
# PRINT MATRIX
# ============================================================
print("\n" + "=" * 70)
print("CONFUSION MATRIX VALUES")
print("=" * 70)

print("Rows    = Actual")
print("Columns = Predicted\n")

print("{'Actual \\ Pred':<15}", end="")

for class_name in classes:
    print(f"{class_name:>12}", end="")

print()

print("-" * (15 + 12 * num_classes))

for i, class_name in enumerate(classes):

    print(f"{class_name:<15}", end="")

    for j in range(num_classes):
        print(f"{cm[i, j]:>12}", end="")

    print()


# ============================================================
# PER-CLASS ACCURACY
# ============================================================
print("\n" + "=" * 70)
print("PER-CLASS RESULTS")
print("=" * 70)

for i, class_name in enumerate(classes):

    total = np.sum(cm[i])
    correct = cm[i, i]

    accuracy = (
        correct / total
        if total > 0
        else 0
    )

    print(
        f"{class_name:<12} "
        f"Correct={correct:>3} / {total:<3} "
        f"Accuracy={accuracy * 100:>6.2f}%"
    )


# ============================================================
# PLOT CONFUSION MATRIX
# ============================================================
fig, ax = plt.subplots(figsize=(9, 7))

image = ax.imshow(cm)

ax.set_title(
    "CNN-2 Medium - Confusion Matrix",
    fontsize=16,
    pad=15
)

ax.set_xlabel(
    "Predicted Label",
    fontsize=12
)

ax.set_ylabel(
    "True Label",
    fontsize=12
)

ax.set_xticks(np.arange(num_classes))
ax.set_yticks(np.arange(num_classes))

ax.set_xticklabels(classes)
ax.set_yticklabels(classes)

plt.setp(
    ax.get_xticklabels(),
    rotation=30,
    ha="right",
    rotation_mode="anchor"
)


# Add numbers inside cells
for i in range(num_classes):

    for j in range(num_classes):

        value = cm[i, j]

        ax.text(
            j,
            i,
            str(value),
            ha="center",
            va="center",
            fontsize=12
        )


# Grid
ax.set_xticks(
    np.arange(-0.5, num_classes, 1),
    minor=True
)

ax.set_yticks(
    np.arange(-0.5, num_classes, 1),
    minor=True
)

ax.grid(
    which="minor",
    linewidth=1
)

ax.tick_params(
    which="minor",
    bottom=False,
    left=False
)

plt.tight_layout()

plt.savefig(
    CM_PATH,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# COMPLETE
# ============================================================
print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(CM_PATH)
print(CM_NPY_PATH)

print("\n" + "=" * 70)
print("CONFUSION MATRIX COMPLETE")
print("=" * 70)