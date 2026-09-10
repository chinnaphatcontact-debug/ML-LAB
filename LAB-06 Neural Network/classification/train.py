import os

import matplotlib.pyplot as plt

from data_loader import load_dataset
from preprocessing import preprocess_data
from nn_model import build_model


# ============================================================
# Configuration
# ============================================================

EPOCHS = 50
BATCH_SIZE = 16

HIDDEN_LAYERS = (16,)

OUTPUT_DIR = "outputs"
GRAPH_DIR = os.path.join(OUTPUT_DIR, "graphs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")


# ============================================================
# Create output directories
# ============================================================

os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# Load Dataset
# ============================================================

print("=" * 60)
print("                  LOAD DATASET")
print("=" * 60)

X, y, feature_names, target_names = load_dataset()

print(f"Samples  : {X.shape[0]}")
print(f"Features : {X.shape[1]}")
print(f"Classes  : {len(target_names)}")


# ============================================================
# Preprocessing
# ============================================================

print("\n" + "=" * 60)
print("                  PREPROCESSING")
print("=" * 60)

(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    scaler
) = preprocess_data(X, y)

print(f"Training   : {X_train.shape}")
print(f"Validation : {X_val.shape}")
print(f"Test       : {X_test.shape}")


# ============================================================
# Build Neural Network
# ============================================================

print("\n" + "=" * 60)
print("              BUILD NEURAL NETWORK")
print("=" * 60)

model = build_model(
    input_dim=X_train.shape[1],
    num_classes=len(target_names),
    hidden_layers=HIDDEN_LAYERS
)

model.summary()


# ============================================================
# Train Neural Network
# ============================================================

print("\n" + "=" * 60)
print("                  TRAINING")
print("=" * 60)

print(f"Epochs       : {EPOCHS}")
print(f"Batch size   : {BATCH_SIZE}")
print(f"Hidden layer : {HIDDEN_LAYERS}")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)


# ============================================================
# Evaluate on Test Set
# ============================================================

print("\n" + "=" * 60)
print("                 TEST RESULTS")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")


# ============================================================
# Save Model
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "nn_model.keras"
)

model.save(model_path)

print(f"\nModel saved to: {model_path}")


# ============================================================
# Plot Accuracy
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

accuracy_path = os.path.join(
    GRAPH_DIR,
    "training_validation_accuracy.png"
)

plt.savefig(
    accuracy_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Accuracy graph saved to: {accuracy_path}")


# ============================================================
# Plot Loss
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

loss_path = os.path.join(
    GRAPH_DIR,
    "training_validation_loss.png"
)

plt.savefig(
    loss_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Loss graph saved to: {loss_path}")


print("\n" + "=" * 60)
print("                 TRAINING COMPLETE")
print("=" * 60)