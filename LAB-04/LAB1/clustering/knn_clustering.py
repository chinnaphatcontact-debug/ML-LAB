import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# =========================================================
# 1. Load Dataset
# =========================================================

data = pd.read_csv("dataset/iris.csv")

print("========== DATASET ==========")
print(data.head())
print()

print("ขนาด Dataset:", data.shape)
print()

# =========================================================
# 2. Select Features
# =========================================================

X = data[
    [
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm"
    ]
]

print("========== FEATURES ==========")
print(X.head())
print()

# =========================================================
# 3. Standardization
# =========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("========== STANDARDIZATION ==========")
print("Standardization completed.")
print()

# =========================================================
# 4. K-Means Clustering
# =========================================================

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

# เพิ่มผล Cluster ลงใน Dataset
data["Cluster"] = clusters

print("========== CLUSTER RESULT ==========")
print(data[["Species", "Cluster"]].head(20))
print()

# =========================================================
# 5. Compare Cluster with Actual Species
# =========================================================

print("========== CLUSTER COUNT ==========")
print(pd.crosstab(
    data["Species"],
    data["Cluster"]
))
print()

# =========================================================
# 6. Evaluate Clustering
# =========================================================

# แปลงชื่อ Species เป็นตัวเลข
species_mapping = {
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2
}

y_true = data["Species"].map(species_mapping)

ari = adjusted_rand_score(
    y_true,
    clusters
)

print("========== CLUSTERING EVALUATION ==========")
print(f"Adjusted Rand Index (ARI) = {ari:.4f}")
print()

# =========================================================
# 7. Visualization
# =========================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    X_scaled[:, 2],
    X_scaled[:, 3],
    c=clusters,
    marker="o"
)

plt.xlabel("Petal Length (Standardized)")
plt.ylabel("Petal Width (Standardized)")

plt.title("K-Means Clustering of Iris Dataset")

plt.grid(True)

plt.show()

# =========================================================
# 8. Save Clustering Result
# =========================================================

data.to_csv(
    "dataset/iris_clustered.csv",
    index=False
)

print("บันทึกผลลัพธ์ไว้ที่ dataset/iris_clustered.csv")