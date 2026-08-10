import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

# โหลด Dataset
df = pd.read_csv("data/titanic.csv")

# ============================================
# LAB 1 : DATASET EXPLORATION

# แสดงข้อมูล 5 แถวแรก
print(df.head())

# Display Shape
print("\n==========================")
print("Dataset Shape")
print("==========================")
print(df.shape)

# Display Data Types
print("\n==========================")
print("Data Types")
print("==========================")
print(df.dtypes)

# Display Summary Statistics
print("\n==========================")
print("Summary Statistics")
print("==========================")
print(df.describe())

# Display Missing Values
print("\n==========================")
print("Missing Values")
print("==========================")
print(df.isnull().sum())

# Display Duplicate Records
print("\n==========================")
print("Duplicate Records")
print("==========================")
print(df.duplicated().sum())

# Display Class Distribution
print("\n==========================")
print("Class Distribution")
print("==========================")
print(df["Survived"].value_counts())

# ============================================

#LAB 2 : Data Visualization
#ขั้นตอนที่ 2 Histogram
print("\n==========================")
print("Histogram")
print("==========================")

df.hist(figsize=(12, 8))

plt.tight_layout()
plt.show()

#ขั้นตอนที่ 3: Correlation Heatmap
print("\n==========================")
print("Correlation Heatmap")
print("==========================")

plt.figure(figsize=(10, 8))

correlation = df.corr(numeric_only=True)

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.show()

# ============================================

#LAB 3 : Data Cleaning
#Step 1 : เปรียบเทียบ Mean และ Median ก่อนทำความสะอาด
print("\n==========================")
print("Mean Before Cleaning")
print("==========================")

print(df.mean(numeric_only=True))

print("\n==========================")
print("Median Before Cleaning")
print("==========================")

print(df.median(numeric_only=True))

#Step 2 : Missing Value Handling
print("\n==========================")
print("Missing Value Handling")
print("==========================")

# Age เติมด้วย Median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Embarked เติมด้วย Mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Cabin เติมข้อความ Unknown
df["Cabin"] = df["Cabin"].fillna("Unknown")

print(df.isnull().sum())

#Step 3 : Duplicate Removal
print("\n==========================")
print("Duplicate Removal")
print("==========================")

before = len(df)

df = df.drop_duplicates()

after = len(df)

print("Before :", before)
print("After  :", after)
print("Removed :", before - after)

#Step 4 : Incorrect Data Correction
print("\n==========================")
print("Incorrect Data Correction")
print("==========================")

# Age ติดลบ
df.loc[df["Age"] < 0, "Age"] = df["Age"].median()

# Fare ติดลบ
df.loc[df["Fare"] < 0, "Fare"] = df["Fare"].median()

print("Incorrect data checked.")

#Step 5 : Data Type Conversion
print("\n==========================")
print("Data Type Conversion")
print("==========================")

df["Survived"] = df["Survived"].astype(int)

print(df.dtypes)

#Step 6 : เปรียบเทียบ Mean และ Median หลัง Cleaning
print("\n==========================")
print("Mean After Cleaning")
print("==========================")

print(df.mean(numeric_only=True))

print("\n==========================")
print("Median After Cleaning")
print("==========================")

print(df.median(numeric_only=True))

# ============================================

#PART 4 : Feature Engineering
#Step 1 : Label Encoding
print("\n==========================")
print("Label Encoding")
print("==========================")

label_encoder = LabelEncoder()

df["Sex_Label"] = label_encoder.fit_transform(df["Sex"])

print(df[["Sex", "Sex_Label"]].head())

#Step 2 : One-Hot Encoding
print("\n==========================")
print("One-Hot Encoding")
print("==========================")

df = pd.get_dummies(
    df,
    columns=["Embarked"],
    dtype=int
)

print(df.head())

#Step 3 : Save Clean Dataset
print("\n==========================")
print("Save Clean Dataset")
print("==========================")

df.to_csv("data/titanic_clean.csv", index=False)

print("Saved : data/titanic_clean.csv")

# ============================================