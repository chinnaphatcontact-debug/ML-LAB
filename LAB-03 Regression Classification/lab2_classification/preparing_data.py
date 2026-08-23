import os
import cv2
import numpy as np
import pandas as pd

from collections import Counter


# ==========================================
# Configuration
# ==========================================

DATASET_PATH = "datasets/regression/UTKFace"

IMAGE_SIZE = (64, 64)


# ==========================================
# 1. Get Image Files
# ==========================================

image_files = [
    file
    for file in os.listdir(DATASET_PATH)
    if file.lower().endswith((".jpg", ".jpeg", ".png"))
]


print("========== CLASSIFICATION DATASET ==========")

print("Total images :", len(image_files))


# ==========================================
# 2. Extract Gender from Filename
# ==========================================

X = []
y = []

valid_files = []

for file in image_files:

    parts = file.split("_")

    # Filename format:
    # age_gender_race_timestamp.jpg.chip.jpg

    if len(parts) < 2:
        continue

    try:

        age = int(parts[0])
        gender = int(parts[1])

    except ValueError:

        continue

    # Gender must be 0 or 1

    if gender not in [0, 1]:
        continue

    image_path = os.path.join(
        DATASET_PATH,
        file
    )

    image = cv2.imread(image_path)

    if image is None:
        continue

    # --------------------------------------
    # Resize
    # --------------------------------------

    image = cv2.resize(
        image,
        IMAGE_SIZE
    )

    # --------------------------------------
    # Grayscale
    # --------------------------------------

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------
    # Normalize
    # --------------------------------------

    image = image.astype(
        np.float32
    ) / 255.0

    # --------------------------------------
    # Flatten
    # --------------------------------------

    image = image.flatten()

    X.append(image)
    y.append(gender)

    valid_files.append(file)


X = np.array(X)
y = np.array(y)


# ==========================================
# 3. Display Dataset Information
# ==========================================

print("\n========== DATA INFORMATION ==========")

print("Valid images     :", len(X))
print("Features/image   :", X.shape[1])

print(
    "Gender distribution:",
    Counter(y)
)


# ==========================================
# 4. Create DataFrame
# ==========================================

gender_labels = np.where(
    y == 0,
    "Male",
    "Female"
)

summary = pd.DataFrame({
    "Gender": gender_labels
})


print("\n========== GENDER DISTRIBUTION ==========")

print(
    summary["Gender"].value_counts()
)


print("\n========== SAMPLE FILES ==========")

for file, gender in zip(
    valid_files[:10],
    y[:10]
):

    label = "Male" if gender == 0 else "Female"

    print(
        f"{file} -> {label}"
    )


print("\nClassification data preparation complete!")