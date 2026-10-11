import os
import json
import numpy as np
import tensorflow as tf

# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "outputs"
)

X_TRAIN_PATH = os.path.join(OUTPUT_DIR, "X_train.npy")
X_VAL_PATH   = os.path.join(OUTPUT_DIR, "X_val.npy")
X_TEST_PATH  = os.path.join(OUTPUT_DIR, "X_test.npy")

Y_TRAIN_PATH = os.path.join(OUTPUT_DIR, "y_train.npy")
Y_VAL_PATH   = os.path.join(OUTPUT_DIR, "y_val.npy")
Y_TEST_PATH  = os.path.join(OUTPUT_DIR, "y_test.npy")

CLASSES_PATH = os.path.join(OUTPUT_DIR, "classes.json")

FINAL_MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "final_cnn_model.keras"
)

FINAL_HISTORY_PATH = os.path.join(
    OUTPUT_DIR,
    "final_history.json"
)

# CNN configuration
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 3

NUM_CLASSES = 5

BATCH_SIZE = 32
EPOCHS = 15


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING SPLIT DATA")
print("=" * 70)

X_train = np.load(X_TRAIN_PATH)
X_val   = np.load(X_VAL_PATH)
X_test  = np.load(X_TEST_PATH)

y_train = np.load(Y_TRAIN_PATH)
y_val   = np.load(Y_VAL_PATH)
y_test  = np.load(Y_TEST_PATH)

with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    classes = json.load(f)

print(f"Training   : {X_train.shape}")
print(f"Validation : {X_val.shape}")
print(f"Testing    : {X_test.shape}")
print(f"Classes    : {classes}")


# ============================================================
# BUILD CNN-2 MEDIUM
# ============================================================

def build_cnn_2_medium():

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
            ),

            # ------------------------------------------------
            # Convolutional Layer 1
            # ------------------------------------------------
            tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=(3, 3),
                activation="relu",
                padding="same"
            ),

            tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2)
            ),

            # ------------------------------------------------
            # Convolutional Layer 2
            # ------------------------------------------------
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                activation="relu",
                padding="same"
            ),

            tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2)
            ),

            # ------------------------------------------------
            # Fully Connected Layer
            # ------------------------------------------------
            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(
                64,
                activation="relu"
            ),

            # ------------------------------------------------
            # Output Layer
            # ------------------------------------------------
            tf.keras.layers.Dense(
                NUM_CLASSES,
                activation="softmax"
            )
        ],
        name="CNN_2_Medium"
    )

    return model


# ============================================================
# CREATE MODEL
# ============================================================

model = build_cnn_2_medium()

print()
print("=" * 70)
print("FINAL MODEL : CNN-2 MEDIUM")
print("=" * 70)

model.summary()


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# CALLBACKS
# ============================================================

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=FINAL_MODEL_PATH,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)

# ลด Learning Rate เมื่อ Validation Loss ไม่ดีขึ้น
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)


# ============================================================
# TRAINING
# ============================================================

print()
print("=" * 70)
print("FINAL TRAINING")
print("=" * 70)

print(f"Architecture : CNN-2 Medium")
print(f"Epochs       : {EPOCHS}")
print(f"Batch Size   : {BATCH_SIZE}")
print(f"Checkpoint   : Best Validation Accuracy")
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

    callbacks=[
        checkpoint,
        reduce_lr
    ],

    verbose=1
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_data = {
    key: [float(value) for value in values]
    for key, values in history.history.items()
}

with open(
    FINAL_HISTORY_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        history_data,
        f,
        indent=4
    )


# ============================================================
# LOAD BEST MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)

best_model = tf.keras.models.load_model(
    FINAL_MODEL_PATH
)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print()
print("=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

test_loss, test_accuracy = best_model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print()
print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")


# ============================================================
# BEST VALIDATION ACCURACY
# ============================================================

val_accuracy = history.history["val_accuracy"]

best_epoch_index = int(
    np.argmax(val_accuracy)
)

best_epoch = best_epoch_index + 1
best_val_accuracy = val_accuracy[best_epoch_index]


print()
print("=" * 70)
print("BEST VALIDATION RESULT")
print("=" * 70)

print(f"Best Epoch            : {best_epoch}")
print(f"Best Validation Acc.  : {best_val_accuracy * 100:.2f}%")
print(f"Final Test Accuracy   : {test_accuracy * 100:.2f}%")


# ============================================================
# FILE INFORMATION
# ============================================================

print()
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print(f"Model   : {FINAL_MODEL_PATH}")
print(f"History : {FINAL_HISTORY_PATH}")

print()
print("=" * 70)
print("FINAL TRAINING COMPLETE")
print("=" * 70)