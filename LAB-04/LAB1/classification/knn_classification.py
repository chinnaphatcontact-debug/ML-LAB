import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================================================
# 1. Load Dataset
# =========================================================

data = pd.read_csv("dataset/iris.csv")

print("========== DATASET ==========")
print(data.head())
print()

print("ขนาด Dataset:", data.shape)
print()

print("ชื่อคอลัมน์:")
print(data.columns.tolist())
print()

# =========================================================
# 2. Select Features and Target
# =========================================================

X = data[
    [
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm"
    ]
]

y = data["Species"]

print("========== FEATURES ==========")
print(X.head())
print()

print("========== TARGET ==========")
print(y.head())
print()

print("จำนวนข้อมูลแต่ละ Class:")
print(y.value_counts())
print()

# =========================================================
# 3. Split Data into Training and Testing
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("========== DATA SPLIT ==========")
print("Training data:", X_train.shape)
print("Testing data :", X_test.shape)
print()

# =========================================================
# 4. Standardization
# =========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("========== STANDARDIZATION ==========")
print("Standardization completed.")
print()

# =========================================================
# 5. Train KNN with Different K Values
# =========================================================

k_values = [3, 5, 7]

results = {}

print("========== KNN RESULTS ==========")

for k in k_values:

    knn = KNeighborsClassifier(
        n_neighbors=k
    )

    # Train
    knn.fit(
        X_train_scaled,
        y_train
    )

    # Predict
    y_pred = knn.predict(
        X_test_scaled
    )

    # Accuracy
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    results[k] = accuracy

    print(
        f"k = {k} | "
        f"Accuracy = {accuracy:.4f} | "
        f"{accuracy * 100:.2f}%"
    )

print()

# =========================================================
# 6. Find Best K
# =========================================================

best_k = max(
    results,
    key=results.get
)

best_accuracy = results[best_k]

print("========== BEST MODEL ==========")

print(
    f"Best k = {best_k}"
)

print(
    f"Best Accuracy = "
    f"{best_accuracy:.4f}"
)

print(
    f"Best Accuracy (%) = "
    f"{best_accuracy * 100:.2f}%"
)

print()

# =========================================================
# 7. Train Best Model
# =========================================================

best_knn = KNeighborsClassifier(
    n_neighbors=best_k
)

best_knn.fit(
    X_train_scaled,
    y_train
)

# Prediction
best_pred = best_knn.predict(
    X_test_scaled
)

# =========================================================
# 8. Classification Report
# =========================================================

print("========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        best_pred,
        target_names=[
            "Iris-setosa",
            "Iris-versicolor",
            "Iris-virginica"
        ]
    )
)

# =========================================================
# 9. Confusion Matrix
# =========================================================

cm = confusion_matrix(
    y_test,
    best_pred
)

print("========== CONFUSION MATRIX ==========")
print(cm)
print()

# =========================================================
# 10. Plot Accuracy Comparison
# =========================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    list(results.keys()),
    list(results.values()),
    marker="o"
)

plt.title(
    "KNN Accuracy Comparison"
)

plt.xlabel(
    "Number of Neighbors (k)"
)

plt.ylabel(
    "Accuracy"
)

plt.xticks(
    k_values
)

plt.ylim(
    0,
    1.05
)

plt.grid(True)

plt.show()

# =========================================================
# 11. Save Results
# =========================================================

with open(
    "classification/result.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "KNN Classification Results\n"
    )

    file.write(
        "==========================\n\n"
    )

    for k, accuracy in results.items():

        file.write(
            f"k = {k} : "
            f"Accuracy = {accuracy:.4f} "
            f"({accuracy * 100:.2f}%)\n"
        )

    file.write("\n")

    file.write(
        f"Best k = {best_k}\n"
    )

    file.write(
        f"Best Accuracy = "
        f"{best_accuracy:.4f} "
        f"({best_accuracy * 100:.2f}%)\n"
    )

print(
    "บันทึกผลลัพธ์ลง "
    "classification/result.txt แล้ว"
)