import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# Configuration
# ==========================================

DATASET_PATH = "datasets/regression/UTKFace"

IMAGE_SIZE = (64, 64)
N_COMPONENTS = 50


# ==========================================
# 1. Load Images and Age Labels
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

    # --------------------------------------
    # Extract age from filename
    # Example:
    # 25_1_2_20170116174525125.jpg.chip.jpg
    # age = 25
    # --------------------------------------

    try:
        age = int(file.split("_")[0])
    except ValueError:
        continue

    image_path = os.path.join(DATASET_PATH, file)

    image = cv2.imread(image_path)

    if image is None:
        continue

    # --------------------------------------
    # Resize
    # --------------------------------------

    image = cv2.resize(image, IMAGE_SIZE)

    # --------------------------------------
    # Convert to Grayscale
    # --------------------------------------

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --------------------------------------
    # Normalize pixel values
    # 0-255 → 0-1
    # --------------------------------------

    image = image.astype(np.float32) / 255.0

    # --------------------------------------
    # Flatten image
    # 64 × 64 → 4096 features
    # --------------------------------------

    image = image.flatten()

    X.append(image)
    y.append(age)


X = np.array(X)
y = np.array(y)


print("Number of images :", len(X))
print("Number of features :", X.shape[1])
print("Age range :", y.min(), "-", y.max())


# ==========================================
# 2. Train / Test Split
# ==========================================

print("\n========== TRAIN / TEST SPLIT ==========")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples :", len(X_train))
print("Testing samples  :", len(X_test))


# ==========================================
# 3. Standardization
# ==========================================

print("\n========== STANDARDIZATION ==========")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("Standardization complete!")


# ==========================================
# 4. PCA
# ==========================================

print("\n========== PCA ==========")

pca = PCA(
    n_components=N_COMPONENTS,
    random_state=42
)

X_train_pca = pca.fit_transform(X_train_scaled)

X_test_pca = pca.transform(X_test_scaled)

print("Original features :", X_train.shape[1])
print("PCA components    :", X_train_pca.shape[1])

print(
    "Explained variance:",
    pca.explained_variance_ratio_.sum()
)


# ==========================================
# 5. Create Linear Regression Model
# ==========================================

print("\n========== LINEAR REGRESSION ==========")

model = LinearRegression()


# ==========================================
# 6. Train Model
# ==========================================

model.fit(
    X_train_pca,
    y_train
)

print("Training complete!")


# ==========================================
# 7. Prediction
# ==========================================

y_pred = model.predict(X_test_pca)


# ==========================================
# 8. Evaluation
# ==========================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n========== MODEL EVALUATION ==========")

print("MAE  :", mae)
print("MSE  :", mse)
print("RMSE :", rmse)
print("R²   :", r2)


# ==========================================
# 9. Display Predictions
# ==========================================

result = pd.DataFrame({
    "Actual Age": y_test,
    "Predicted Age": y_pred
})

print("\n========== SAMPLE PREDICTIONS ==========")

print(
    result.head(10).to_string(index=False)
)


# ==========================================
# 10. Visualization
# ==========================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred
)

plt.xlabel("Actual Age")

plt.ylabel("Predicted Age")

plt.title(
    "Age Prediction using PCA + Linear Regression"
)

plt.grid()

plt.show()