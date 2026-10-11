import json
import csv
from pathlib import Path

import numpy as np
import tensorflow as tf

from cnn_model import (
    build_cnn_1,
    build_cnn_2,
    build_cnn_3
)


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

EPOCHS = 10
BATCH_SIZE = 32

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("LOADING DATASET")
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


# ============================================================
# MODEL CONFIGURATIONS
# ============================================================

model_builders = {
    "CNN_1_Simple": build_cnn_1,
    "CNN_2_Medium": build_cnn_2,
    "CNN_3_Deep": build_cnn_3,
}


# ============================================================
# RESULT STORAGE
# ============================================================

results = {}


# ============================================================
# TRAIN EACH MODEL
# ============================================================

for model_name, build_model in model_builders.items():

    print("\n")
    print("=" * 70)
    print(f"TRAINING: {model_name}")
    print("=" * 70)

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_model()

    # --------------------------------------------------------
    # Compile
    # --------------------------------------------------------

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # --------------------------------------------------------
    # Model summary
    # --------------------------------------------------------

    model.summary()

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print(f"START TRAINING {model_name}")
    print("-" * 70)

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

    # --------------------------------------------------------
    # Test evaluation
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print(f"TEST EVALUATION: {model_name}")
    print("-" * 70)

    test_loss, test_accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=1
    )

    # --------------------------------------------------------
    # Best validation accuracy
    # --------------------------------------------------------

    best_val_accuracy = max(
        history.history["val_accuracy"]
    )

    best_val_epoch = (
        history.history["val_accuracy"].index(
            best_val_accuracy
        ) + 1
    )

    # --------------------------------------------------------
    # Best validation loss
    # --------------------------------------------------------

    best_val_loss = min(
        history.history["val_loss"]
    )

    best_loss_epoch = (
        history.history["val_loss"].index(
            best_val_loss
        ) + 1
    )

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    results[model_name] = {
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),

        "best_val_accuracy": float(
            best_val_accuracy
        ),

        "best_val_epoch": int(
            best_val_epoch
        ),

        "best_val_loss": float(
            best_val_loss
        ),

        "best_loss_epoch": int(
            best_loss_epoch
        ),

        "total_params": int(
            model.count_params()
        )
    }

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = (
        OUTPUT_DIR /
        f"{model_name.lower()}.keras"
    )

    model.save(model_path)

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    history_path = (
        OUTPUT_DIR /
        f"{model_name.lower()}_history.json"
    )

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

    print("\nModel saved:")
    print(model_path)

    print("History saved:")
    print(history_path)

    print(
        f"\nTest Accuracy : "
        f"{test_accuracy * 100:.2f}%"
    )

    print(
        f"Best Val Acc  : "
        f"{best_val_accuracy * 100:.2f}% "
        f"(Epoch {best_val_epoch})"
    )

    print(
        f"Best Val Loss : "
        f"{best_val_loss:.4f} "
        f"(Epoch {best_loss_epoch})"
    )


# ============================================================
# SAVE COMPARISON JSON
# ============================================================

comparison_json = (
    OUTPUT_DIR / "architecture_comparison.json"
)

with open(
    comparison_json,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        results,
        file,
        indent=4
    )


# ============================================================
# SAVE COMPARISON CSV
# ============================================================

comparison_csv = (
    OUTPUT_DIR / "architecture_comparison.csv"
)

with open(
    comparison_csv,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Model",
        "Parameters",
        "Test Accuracy",
        "Test Loss",
        "Best Validation Accuracy",
        "Best Validation Epoch",
        "Best Validation Loss",
        "Best Loss Epoch"
    ])

    for model_name, result in results.items():

        writer.writerow([
            model_name,
            result["total_params"],
            result["test_accuracy"],
            result["test_loss"],
            result["best_val_accuracy"],
            result["best_val_epoch"],
            result["best_val_loss"],
            result["best_loss_epoch"]
        ])


# ============================================================
# PRINT FINAL COMPARISON
# ============================================================

print("\n\n")
print("=" * 90)
print("CNN ARCHITECTURE COMPARISON")
print("=" * 90)

print(
    f"{'Model':<20}"
    f"{'Parameters':>15}"
    f"{'Test Acc.':>15}"
    f"{'Best Val Acc.':>18}"
    f"{'Best Epoch':>15}"
)

print("-" * 90)

for model_name, result in results.items():

    print(
        f"{model_name:<20}"
        f"{result['total_params']:>15,}"
        f"{result['test_accuracy'] * 100:>14.2f}%"
        f"{result['best_val_accuracy'] * 100:>17.2f}%"
        f"{result['best_val_epoch']:>15}"
    )


print("=" * 90)

print("\nComparison files saved:")
print(comparison_json)
print(comparison_csv)

print("\n" + "=" * 70)
print("ALL CNN EXPERIMENTS COMPLETE")
print("=" * 70)