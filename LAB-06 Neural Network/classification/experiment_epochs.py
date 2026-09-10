import os
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import load_dataset
from preprocessing import preprocess_data
from nn_model import build_model


# ============================================================
# Configuration
# ============================================================

EPOCHS_LIST = [10, 30, 50, 100]

BATCH_SIZE = 16

HIDDEN_LAYERS = (16,)

OUTPUT_DIR = "outputs"
RESULT_DIR = os.path.join(OUTPUT_DIR, "results")
GRAPH_DIR = os.path.join(OUTPUT_DIR, "graphs")


# ============================================================
# Create output directories
# ============================================================

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)


# ============================================================
# Load Dataset
# ============================================================

print("=" * 60)
print("             EPOCH COMPARISON EXPERIMENT")
print("=" * 60)

X, y, feature_names, target_names = load_dataset()


# ============================================================
# Preprocessing
# ============================================================

(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    scaler
) = preprocess_data(X, y)


# ============================================================
# Store experiment results
# ============================================================

results = []


# ============================================================
# Train models with different epochs
# ============================================================

for epochs in EPOCHS_LIST:

    print("\n" + "=" * 60)
    print(f"TRAINING MODEL - {epochs} EPOCHS")
    print("=" * 60)

    # Build a new model for each experiment
    model = build_model(
        input_dim=X_train.shape[1],
        num_classes=len(target_names),
        hidden_layers=HIDDEN_LAYERS
    )

    # Train model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    # Evaluate model
    train_loss, train_accuracy = model.evaluate(
        X_train,
        y_train,
        verbose=0
    )

    val_loss, val_accuracy = model.evaluate(
        X_val,
        y_val,
        verbose=0
    )

    test_loss, test_accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    # Store results
    results.append({
        "Epochs": epochs,
        "Train Accuracy": train_accuracy,
        "Validation Accuracy": val_accuracy,
        "Test Accuracy": test_accuracy,
        "Train Loss": train_loss,
        "Validation Loss": val_loss,
        "Test Loss": test_loss
    })

    print(f"Train Accuracy      : {train_accuracy * 100:.2f}%")
    print(f"Validation Accuracy : {val_accuracy * 100:.2f}%")
    print(f"Test Accuracy       : {test_accuracy * 100:.2f}%")

    print(f"Test Loss           : {test_loss:.4f}")


# ============================================================
# Create DataFrame
# ============================================================

results_df = pd.DataFrame(results)


# ============================================================
# Display comparison table
# ============================================================

print("\n" + "=" * 60)
print("                 EPOCH COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Train Accuracy": "{:.4f}".format,
            "Validation Accuracy": "{:.4f}".format,
            "Test Accuracy": "{:.4f}".format,
            "Train Loss": "{:.4f}".format,
            "Validation Loss": "{:.4f}".format,
            "Test Loss": "{:.4f}".format
        }
    )
)


# ============================================================
# Save results
# ============================================================

csv_path = os.path.join(
    RESULT_DIR,
    "epoch_comparison.csv"
)

results_df.to_csv(
    csv_path,
    index=False
)

print(f"\nResults saved to: {csv_path}")


# ============================================================
# Plot Accuracy Comparison
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    results_df["Epochs"],
    results_df["Train Accuracy"],
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    results_df["Epochs"],
    results_df["Validation Accuracy"],
    marker="o",
    label="Validation Accuracy"
)

plt.plot(
    results_df["Epochs"],
    results_df["Test Accuracy"],
    marker="o",
    label="Test Accuracy"
)

plt.title("Accuracy Comparison by Number of Epochs")
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy")
plt.xticks(EPOCHS_LIST)
plt.legend()
plt.grid(True)

accuracy_path = os.path.join(
    GRAPH_DIR,
    "epoch_accuracy_comparison.png"
)

plt.savefig(
    accuracy_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Accuracy comparison graph saved to: {accuracy_path}")


# ============================================================
# Find best model
# ============================================================

best_index = results_df["Test Accuracy"].idxmax()

best_result = results_df.loc[best_index]

print("\n" + "=" * 60)
print("                  BEST EPOCH")
print("=" * 60)

print(f"Best Epochs       : {int(best_result['Epochs'])}")
print(
    f"Best Test Accuracy: "
    f"{best_result['Test Accuracy'] * 100:.2f}%"
)

print("=" * 60)