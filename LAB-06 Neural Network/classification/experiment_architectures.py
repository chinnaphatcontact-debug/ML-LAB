import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from data_loader import load_dataset
from preprocessing import preprocess_data
from nn_model import build_model


# ============================================================
# Configuration
# ============================================================

EPOCHS = 50
BATCH_SIZE = 16

CONFIGURATIONS = {
    "NN_1_Hidden_16": (16,),
    "NN_2_Hidden_32_16": (32, 16),
    "NN_2_Hidden_64_32": (64, 32)
}

OUTPUT_DIR = "outputs"
RESULT_DIR = os.path.join(OUTPUT_DIR, "results")
GRAPH_DIR = os.path.join(OUTPUT_DIR, "graphs")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)


# ============================================================
# Load Dataset
# ============================================================

print("=" * 60)
print("       NEURAL NETWORK ARCHITECTURE COMPARISON")
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
# Store Results
# ============================================================

results = []


# ============================================================
# Train Each Configuration
# ============================================================

for name, hidden_layers in CONFIGURATIONS.items():

    print("\n" + "=" * 60)
    print(f"TRAINING: {name}")
    print("=" * 60)

    print(f"Hidden layers : {hidden_layers}")
    print(f"Epochs        : {EPOCHS}")
    print(f"Batch size    : {BATCH_SIZE}")

    # --------------------------------------------------------
    # Set random seed for reproducibility
    # --------------------------------------------------------

    tf.keras.utils.set_random_seed(42)

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_model(
        input_dim=X_train.shape[1],
        num_classes=len(target_names),
        hidden_layers=hidden_layers
    )

    # --------------------------------------------------------
    # Count parameters
    # --------------------------------------------------------

    total_params = model.count_params()

    print(f"Parameters    : {total_params}")

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

    results.append({
        "Model": name,
        "Hidden Layers": str(hidden_layers),
        "Parameters": total_params,
        "Epochs": EPOCHS,
        "Train Accuracy": train_accuracy,
        "Validation Accuracy": val_accuracy,
        "Test Accuracy": test_accuracy,
        "Train Loss": train_loss,
        "Validation Loss": val_loss,
        "Test Loss": test_loss
    })

    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print(f"Train Accuracy      : {train_accuracy * 100:.2f}%")
    print(f"Validation Accuracy : {val_accuracy * 100:.2f}%")
    print(f"Test Accuracy       : {test_accuracy * 100:.2f}%")

    print(f"Test Loss           : {test_loss:.4f}")


# ============================================================
# Create DataFrame
# ============================================================

results_df = pd.DataFrame(results)


# ============================================================
# Display Results
# ============================================================

print("\n" + "=" * 60)
print("          ARCHITECTURE COMPARISON RESULTS")
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
# Save Results
# ============================================================

csv_path = os.path.join(
    RESULT_DIR,
    "architecture_comparison.csv"
)

results_df.to_csv(
    csv_path,
    index=False
)

print(f"\nResults saved to: {csv_path}")


# ============================================================
# Plot Accuracy Comparison
# ============================================================

plt.figure(figsize=(9, 5))

plt.bar(
    results_df["Model"],
    results_df["Test Accuracy"]
)

plt.title("Test Accuracy Comparison of Neural Network Architectures")
plt.xlabel("Neural Network Configuration")
plt.ylabel("Test Accuracy")
plt.ylim(0, 1.05)
plt.xticks(rotation=15)
plt.grid(axis="y")

accuracy_path = os.path.join(
    GRAPH_DIR,
    "architecture_test_accuracy.png"
)

plt.savefig(
    accuracy_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Accuracy graph saved to: {accuracy_path}")


# ============================================================
# Find Best Architecture
# ============================================================

# Select best architecture using validation accuracy
# Test set is reserved for final evaluation.

best_index = results_df["Validation Accuracy"].idxmax()

best_result = results_df.loc[best_index]

print("\n" + "=" * 60)
print("             BEST ARCHITECTURE")
print("=" * 60)

print(f"Model             : {best_result['Model']}")
print(f"Hidden Layers     : {best_result['Hidden Layers']}")
print(f"Parameters        : {int(best_result['Parameters'])}")
print(f"Epochs            : {int(best_result['Epochs'])}")
print(
    f"Test Accuracy     : "
    f"{best_result['Test Accuracy'] * 100:.2f}%"
)
print(
    f"Validation Accuracy: "
    f"{best_result['Validation Accuracy'] * 100:.2f}%"
)
print(f"Test Loss         : {best_result['Test Loss']:.4f}")

print("=" * 60)