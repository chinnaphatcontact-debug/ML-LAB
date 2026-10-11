import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from cnn_model import (
    build_cnn_1,
    build_cnn_2,
    build_cnn_3
)


# =========================
# Configuration
# =========================

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

BATCH_SIZE = 32
EPOCHS = 15

# =========================
# Load Dataset
# =========================

print("=" * 70)
print("LOADING SPLIT DATA")
print("=" * 70)

X_train = np.load(OUTPUT_DIR / "X_train.npy")
X_val = np.load(OUTPUT_DIR / "X_val.npy")
X_test = np.load(OUTPUT_DIR / "X_test.npy")

y_train = np.load(OUTPUT_DIR / "y_train.npy")
y_val = np.load(OUTPUT_DIR / "y_val.npy")
y_test = np.load(OUTPUT_DIR / "y_test.npy")

with open(
    OUTPUT_DIR / "classes.json",
    "r",
    encoding="utf-8"
) as file:
    class_names = json.load(file)


print(f"Training   : {X_train.shape}")
print(f"Validation : {X_val.shape}")
print(f"Testing    : {X_test.shape}")
print(f"Classes    : {class_names}")


# =========================
# Select Model
# =========================

model = build_cnn_2()


# =========================
# Compile
# =========================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# =========================
# Display Model
# =========================

print("\n" + "=" * 70)
print("MODEL")
print("=" * 70)

model.summary()


# =========================
# Train
# =========================

print("\n" + "=" * 70)
print("TRAINING")
print("=" * 70)

history = model.fit(
    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=EPOCHS,
    batch_size=BATCH_SIZE,

    verbose=1
)


# =========================
# Evaluate
# =========================

print("\n" + "=" * 70)
print("TEST EVALUATION")
print("=" * 70)

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print(f"\nTest Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")


# =========================
# Save Model
# =========================

model_path = OUTPUT_DIR / "cnn_2_model.keras"

model.save(model_path)

print("\nModel saved:")
print(model_path)


# =========================
# Save History
# =========================

history_path = OUTPUT_DIR / "cnn_2_history.json"

with open(
    history_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        history.history,
        file,
        indent=4
    )

print("History saved:")
print(history_path)

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)