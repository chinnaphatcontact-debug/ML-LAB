import os
import tensorflow as tf

from data_loader import load_dataset
from preprocessing import preprocess_data
from nn_model import build_model


# ============================================================
# Configuration
# ============================================================

EPOCHS = 50
BATCH_SIZE = 16
HIDDEN_LAYERS = (32, 16)

MODEL_DIR = "outputs/models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "final_nn_model.keras"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# Header
# ============================================================

print("=" * 60)
print("              FINAL NEURAL NETWORK MODEL")
print("=" * 60)


# ============================================================
# Load Dataset
# ============================================================

print("\n" + "=" * 60)
print("LOAD DATASET")
print("=" * 60)

X, y, feature_names, target_names = load_dataset()

print(f"Samples  : {len(X)}")
print(f"Features : {X.shape[1]}")
print(f"Classes  : {len(target_names)}")


# ============================================================
# Preprocessing
# ============================================================

print("\n" + "=" * 60)
print("PREPROCESSING")
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
# Build Final Model
# ============================================================

print("\n" + "=" * 60)
print("BUILD FINAL MODEL")
print("=" * 60)

print("Architecture : 13 → 32 → 16 → 3")
print(f"Hidden layers: {HIDDEN_LAYERS}")
print(f"Epochs       : {EPOCHS}")
print(f"Batch size   : {BATCH_SIZE}")

tf.keras.utils.set_random_seed(42)

model = build_model(
    input_dim=X_train.shape[1],
    num_classes=len(target_names),
    hidden_layers=HIDDEN_LAYERS
)

model.summary()


# ============================================================
# Training
# ============================================================

print("\n" + "=" * 60)
print("TRAIN FINAL MODEL")
print("=" * 60)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)


# ============================================================
# Final Evaluation
# ============================================================

print("\n" + "=" * 60)
print("FINAL MODEL RESULTS")
print("=" * 60)

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

print(f"Train Accuracy      : {train_accuracy * 100:.2f}%")
print(f"Validation Accuracy : {val_accuracy * 100:.2f}%")
print(f"Test Accuracy       : {test_accuracy * 100:.2f}%")

print(f"Train Loss          : {train_loss:.4f}")
print(f"Validation Loss     : {val_loss:.4f}")
print(f"Test Loss           : {test_loss:.4f}")


# ============================================================
# Save Model
# ============================================================

model.save(MODEL_PATH)

print("\n" + "=" * 60)
print("MODEL SAVED")
print("=" * 60)

print(f"Final model saved to: {MODEL_PATH}")

print("\n" + "=" * 60)
print("FINAL MODEL TRAINING COMPLETE")
print("=" * 60)