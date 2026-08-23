import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np


# ==========================================
# Dataset
# ==========================================

data = pd.DataFrame({
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Score": [35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
})


X = data[["Hours"]]
y = data["Score"]


# ==========================================
# Train / Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# Train Model
# ==========================================

model = LinearRegression()

model.fit(
    X_train,
    y_train
)


# ==========================================
# Predictions
# ==========================================

y_train_pred = model.predict(X_train)

y_test_pred = model.predict(X_test)


# ==========================================
# Metrics
# ==========================================

train_r2 = r2_score(
    y_train,
    y_train_pred
)

test_r2 = r2_score(
    y_test,
    y_test_pred
)


train_rmse = np.sqrt(
    mean_squared_error(
        y_train,
        y_train_pred
    )
)

test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_test_pred
    )
)


# ==========================================
# Display
# ==========================================

print("========== TRAINING VS TESTING ==========")

print(
    "Training R² :",
    train_r2
)

print(
    "Testing R²  :",
    test_r2
)

print(
    "Training RMSE :",
    train_rmse
)

print(
    "Testing RMSE  :",
    test_rmse
)


# ==========================================
# Comparison Table
# ==========================================

results = pd.DataFrame({
    "Dataset": [
        "Training",
        "Testing"
    ],

    "R2": [
        train_r2,
        test_r2
    ],

    "RMSE": [
        train_rmse,
        test_rmse
    ]
})


print("\n========== COMPARISON TABLE ==========")

print(
    results.to_string(index=False)
)


# ==========================================
# R² Plot
# ==========================================

plt.figure(figsize=(7, 5))

plt.bar(
    results["Dataset"],
    results["R2"]
)

plt.ylabel("R²")

plt.title(
    "Training vs Testing R²"
)

plt.ylim(0, 1.1)

plt.grid(
    axis="y"
)

plt.tight_layout()

plt.show()