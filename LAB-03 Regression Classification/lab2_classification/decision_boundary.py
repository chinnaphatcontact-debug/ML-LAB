import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


# ==========================================
# Configuration
# ==========================================

DATASET_PATH = "datasets/regression/UTKFace"

IMAGE_SIZE = (64, 64)


# ==========================================
# 1. Load Dataset
# ==========================================

X = []
y = []

image_files = [
    file
    for file in os.listdir(DATASET_PATH)
    if file.lower().endswith((".jpg", ".jpeg", ".png"))
]


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

    image = cv2.resize(
        image,
        IMAGE_SIZE
    )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    image = image.astype(
        np.float32
    ) / 255.0

    X.append(
        image.flatten()
    )

    y.append(gender)


X = np.array(X)
y = np.array(y)


# ==========================================
# 2. Train / Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# 3. Standardization
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ==========================================
# 4. PCA → 2 Components
# ==========================================

pca = PCA(
    n_components=2,
    random_state=42
)

X_train_pca = pca.fit_transform(
    X_train_scaled
)

X_test_pca = pca.transform(
    X_test_scaled
)


# ==========================================
# 5. Logistic Regression
# ==========================================

model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

model.fit(
    X_train_pca,
    y_train
)


# ==========================================
# 6. Create Mesh Grid
# ==========================================

x_min = X_train_pca[:, 0].min() - 1
x_max = X_train_pca[:, 0].max() + 1

y_min = X_train_pca[:, 1].min() - 1
y_max = X_train_pca[:, 1].max() + 1


xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)


grid = np.c_[
    xx.ravel(),
    yy.ravel()
]


# ==========================================
# 7. Predict Decision Boundary
# ==========================================

Z = model.predict(
    grid
)

Z = Z.reshape(
    xx.shape
)


# ==========================================
# 8. Plot
# ==========================================

plt.figure(figsize=(9, 7))

plt.contourf(
    xx,
    yy,
    Z,
    alpha=0.25
)

plt.scatter(
    X_train_pca[y_train == 0, 0],
    X_train_pca[y_train == 0, 1],
    label="Male",
    alpha=0.6
)

plt.scatter(
    X_train_pca[y_train == 1, 0],
    X_train_pca[y_train == 1, 1],
    label="Female",
    alpha=0.6
)

plt.xlabel(
    "Principal Component 1"
)

plt.ylabel(
    "Principal Component 2"
)

plt.title(
    "Decision Boundary - Logistic Regression"
)

plt.legend()

plt.grid()

plt.show()