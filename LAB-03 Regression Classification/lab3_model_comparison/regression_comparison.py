import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# Regression Model Comparison
# ==========================================

results = pd.DataFrame({
    "Model": [
        "Simple Linear Regression",
        "Multiple Linear Regression"
    ],

    "Features": [
        "Hours",
        "Hours + Attendance + Assignment"
    ],

    "MAE": [
        0.0,
        0.2
    ],

    "MSE": [
        0.0,
        0.08
    ],

    "RMSE": [
        0.0,
        0.2828427125
    ],

    "R2": [
        1.0,
        0.9997231834
    ]
})


# ==========================================
# Display Results
# ==========================================

print("========== REGRESSION MODEL COMPARISON ==========")

print(
    results.to_string(index=False)
)


# ==========================================
# Best Model
# ==========================================

best_r2 = results.loc[
    results["R2"].idxmax()
]

best_rmse = results.loc[
    results["RMSE"].idxmin()
]


print("\n========== BEST MODEL ==========")

print(
    "Best R² model   :",
    best_r2["Model"]
)

print(
    "Best RMSE model :",
    best_rmse["Model"]
)


# ==========================================
# Visualization: R²
# ==========================================

plt.figure(figsize=(8, 5))

plt.bar(
    results["Model"],
    results["R2"]
)

plt.ylabel("R²")

plt.title(
    "Simple vs Multiple Linear Regression"
)

plt.ylim(0, 1.1)

plt.xticks(
    rotation=15
)

plt.grid(
    axis="y"
)

plt.tight_layout()

plt.show()


# ==========================================
# Visualization: RMSE
# ==========================================

plt.figure(figsize=(8, 5))

plt.bar(
    results["Model"],
    results["RMSE"]
)

plt.ylabel("RMSE")

plt.title(
    "RMSE Comparison"
)

plt.xticks(
    rotation=15
)

plt.grid(
    axis="y"
)

plt.tight_layout()

plt.show()