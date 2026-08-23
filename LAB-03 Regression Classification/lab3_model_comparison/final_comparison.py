import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# Final Model Comparison
# ==========================================

results = pd.DataFrame({
    "Task": [
        "Regression",
        "Regression",
        "Regression",
        "Classification"
    ],

    "Model": [
        "Simple Linear Regression",
        "Multiple Linear Regression",
        "PCA + Linear Regression",
        "PCA + Logistic Regression"
    ],

    "Target": [
        "Score",
        "Score",
        "Age",
        "Gender"
    ],

    "MAE": [
        0.000,
        0.200,
        14.951,
        None
    ],

    "RMSE": [
        0.000,
        0.283,
        18.974,
        None
    ],

    "R2": [
        1.000,
        0.9997,
        0.4216,
        None
    ],

    "Accuracy": [
        None,
        None,
        None,
        0.7670
    ],

    "Precision": [
        None,
        None,
        None,
        0.7730
    ],

    "Recall": [
        None,
        None,
        None,
        0.7826
    ],

    "F1": [
        None,
        None,
        None,
        0.7778
    ],

    "AUC": [
        None,
        None,
        None,
        0.8467
    ]
})


# ==========================================
# Display Table
# ==========================================

print("========== FINAL MODEL COMPARISON ==========")

print(
    results.to_string(index=False)
)


# ==========================================
# Regression Comparison
# ==========================================

regression = results[
    results["Task"] == "Regression"
]


print("\n========== REGRESSION MODELS ==========")

print(
    regression[
        [
            "Model",
            "Target",
            "MAE",
            "RMSE",
            "R2"
        ]
    ].to_string(index=False)
)


# ==========================================
# Classification Comparison
# ==========================================

classification = results[
    results["Task"] == "Classification"
]


print("\n========== CLASSIFICATION MODEL ==========")

print(
    classification[
        [
            "Model",
            "Target",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "AUC"
        ]
    ].to_string(index=False)
)


# ==========================================
# Classification Metrics Plot
# ==========================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "AUC"
]

values = classification.iloc[0][metrics].astype(float)


plt.figure(figsize=(8, 5))

plt.bar(
    metrics,
    values
)

plt.ylabel("Score")

plt.title(
    "Classification Model Performance"
)

plt.ylim(0, 1)

plt.grid(
    axis="y"
)

plt.tight_layout()

plt.show()