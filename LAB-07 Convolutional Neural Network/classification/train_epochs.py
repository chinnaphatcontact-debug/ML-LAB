import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from cnn_model import build_cnn_2


# ============================================================
# CONFIGURATION
# ============================================================

EPOCHS_LIST = [5, 10, 15, 20]

BATCH_SIZE = 32

SEED = 42


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# RANDOM SEED
# ============================================================

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING SPLIT DATA")
print("=" * 70)

X_train = np.load(OUTPUT_DIR / "X_train.npy")
X_val = np.load(OUTPUT_DIR / "X_val.npy")
X_test = np.load(OUTPUT_DIR / "X_test.npy")

y_train = np.load(OUTPUT_DIR / "y_train.npy")
y_val = np.load(OUTPUT_DIR / "y_val.npy")
y_test = np.load(OUTPUT_DIR / "y_test.npy")

print(f"Training   : {X_train.shape}")
print(f"Validation : {X_val.shape}")
print(f"Testing    : {X_test.shape}")

print()


# ============================================================
# TRAINING
# ============================================================

results = {}


for epochs in EPOCHS_LIST:

    print("=" * 70)
    print(f"TRAINING CNN-2 MEDIUM - {epochs} EPOCHS")
    print("=" * 70)

    # --------------------------------------------------------
    # Build a fresh model
    # --------------------------------------------------------

    model = build_cnn_2()

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    # --------------------------------------------------------
    # Best validation accuracy
    # --------------------------------------------------------

    best_val_accuracy = max(
        history.history["val_accuracy"]
    )

    best_epoch = (
        history.history["val_accuracy"].index(
            best_val_accuracy
        ) + 1
    )

    # --------------------------------------------------------
    # Test evaluation
    # --------------------------------------------------------

    test_loss, test_accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results[str(epochs)] = {
        "epochs": epochs,
        "best_val_accuracy": float(best_val_accuracy),
        "best_epoch": best_epoch,
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss)
    }

    print()
    print(f"Epochs              : {epochs}")
    print(f"Best Val Accuracy   : {best_val_accuracy * 100:.2f}%")
    print(f"Best Epoch          : {best_epoch}")
    print(f"Test Accuracy       : {test_accuracy * 100:.2f}%")
    print(f"Test Loss           : {test_loss:.4f}")
    print()


# ============================================================
# SAVE RESULTS
# ============================================================

result_path = OUTPUT_DIR / "epoch_comparison.json"

with open(
    result_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        results,
        file,
        indent=4
    )


# ============================================================
# SUMMARY
# ============================================================

print("=" * 70)
print("EPOCH COMPARISON SUMMARY")
print("=" * 70)

print(
    f"{'Epochs':<10}"
    f"{'Best Val Acc':<20}"
    f"{'Best Epoch':<15}"
    f"{'Test Accuracy':<20}"
)

print("-" * 70)

for epochs in EPOCHS_LIST:

    result = results[str(epochs)]

    print(
        f"{epochs:<10}"
        f"{result['best_val_accuracy'] * 100:.2f}%"
        f"{'':<14}"
        f"{result['best_epoch']:<15}"
        f"{result['test_accuracy'] * 100:.2f}%"
    )

print()

print(f"Results saved:")
print(result_path)

print("=" * 70)
print("EPOCH COMPARISON COMPLETE")
print("=" * 70)