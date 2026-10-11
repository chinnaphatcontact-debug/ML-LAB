import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from data_loader import load_dataset


# =========================
# Configuration
# =========================

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

RANDOM_STATE = 42

TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15


# =========================
# Create output directory
# =========================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Load Dataset
# =========================

X, y, class_names = load_dataset()


# =========================
# Train/Test Split
# =========================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=VALIDATION_SIZE + TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)


# =========================
# Validation/Test Split
# =========================

relative_test_size = TEST_SIZE / (VALIDATION_SIZE + TEST_SIZE)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=relative_test_size,
    random_state=RANDOM_STATE,
    stratify=y_temp
)


# =========================
# Save arrays
# =========================

np.save(OUTPUT_DIR / "X_train.npy", X_train)
np.save(OUTPUT_DIR / "X_val.npy", X_val)
np.save(OUTPUT_DIR / "X_test.npy", X_test)

np.save(OUTPUT_DIR / "y_train.npy", y_train)
np.save(OUTPUT_DIR / "y_val.npy", y_val)
np.save(OUTPUT_DIR / "y_test.npy", y_test)


# =========================
# Save class names
# =========================

with open(
    OUTPUT_DIR / "classes.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        class_names,
        file,
        indent=4,
        ensure_ascii=False
    )


# =========================
# Display Results
# =========================

print("\n" + "=" * 60)
print("DATA SPLIT")
print("=" * 60)

print(f"Training   : {X_train.shape}")
print(f"Validation : {X_val.shape}")
print(f"Testing    : {X_test.shape}")

print("\nLabel distribution:")

for index, class_name in enumerate(class_names):

    train_count = np.sum(y_train == index)
    val_count = np.sum(y_val == index)
    test_count = np.sum(y_test == index)

    print(
        f"{class_name:<12} "
        f"Train={train_count:<4} "
        f"Val={val_count:<4} "
        f"Test={test_count:<4}"
    )

print("\n" + "=" * 60)
print("FILES SAVED")
print("=" * 60)

print(OUTPUT_DIR / "X_train.npy")
print(OUTPUT_DIR / "X_val.npy")
print(OUTPUT_DIR / "X_test.npy")
print(OUTPUT_DIR / "y_train.npy")
print(OUTPUT_DIR / "y_val.npy")
print(OUTPUT_DIR / "y_test.npy")
print(OUTPUT_DIR / "classes.json")

print("=" * 60)