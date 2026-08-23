import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)


# ==========================================
# Configuration
# ==========================================

DATASET_PATH = "datasets/regression/UTKFace"

IMAGE_SIZE = (64, 64)

N_COMPONENTS = 50


# ==========================================
# 1. Load Dataset
# ==========================================

print("========== LOADING DATASET ==========")

image_files = [
    file
    for file in os.listdir(DATASET_PATH)
    if file.lower().endswith((".jpg", ".jpeg", ".png"))
]


X = []
y = []


for file in image_files:

    parts = file.split("_")

    if len(parts) < 2:
        continue

    try:
        gender = int(parts[1])
    except ValueError:
        continue

    if gender not in [0, 1]:
        continue

    image_path = os.path.join(
        DATASET_PATH,
        file
    )

    image = cv2.imread(image_path)

    if image is None:
        continue

    # Resize
    image = cv2.resize(
        image,
        IMAGE_SIZE
    )

    # Grayscale
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Normalize
    image = image.astype(
        np.float32
    ) / 255.0

    # Flatten
    image = image.flatten()

    X.append(image)
    y.append(gender)


X = np.array(X)
y = np.array(y)


print("Images   :", len(X))
print("Features :", X.shape[1])


# ==========================================
# 2. Train / Test Split
# ==========================================

print("\n========== TRAIN / TEST SPLIT ==========")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("Training samples :", len(X_train))
print("Testing samples  :", len(X_test))


# ==========================================
# 3. Standardization
# ==========================================

print("\n========== STANDARDIZATION ==========")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

print("Standardization complete!")


# ==========================================
# 4. PCA
# ==========================================

print("\n========== PCA ==========")

pca = PCA(
    n_components=N_COMPONENTS,
    random_state=42
)

X_train_pca = pca.fit_transform(
    X_train_scaled
)

X_test_pca = pca.transform(
    X_test_scaled
)


explained_variance = (
    pca.explained_variance_ratio_.sum()
)


print(
    "Original features :",
    X_train.shape[1]
)

print(
    "PCA components    :",
    X_train_pca.shape[1]
)

print(
    "Explained variance:",
    explained_variance
)


# ==========================================
# 5. Logistic Regression
# ==========================================

print("\n========== LOGISTIC REGRESSION ==========")

model = LogisticRegression(
    max_iter=2000,
    random_state=42
)


# ==========================================
# 6. Training
# ==========================================

model.fit(
    X_train_pca,
    y_train
)

print("Training complete!")


# ==========================================
# 7. Prediction
# ==========================================

y_pred = model.predict(
    X_test_pca
)

y_prob = model.predict_proba(
    X_test_pca
)[:, 1]


# ==========================================
# 8. Classification Metrics
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

auc = roc_auc_score(
    y_test,
    y_prob
)


print("\n========== MODEL PERFORMANCE ==========")

print("Accuracy :", accuracy)

print("Precision:", precision)

print("Recall   :", recall)

print("F1-score :", f1)

print("AUC      :", auc)


# ==========================================
# 9. Confusion Matrix
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred
)


print("\n========== CONFUSION MATRIX ==========")

print(cm)


# ==========================================
# 10. Classification Report
# ==========================================

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Male",
            "Female"
        ]
    )
)


# ==========================================
# 11. ROC Curve
# ==========================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)


plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC = {auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve - Gender Classification"
)

plt.legend()

plt.grid()

plt.show()