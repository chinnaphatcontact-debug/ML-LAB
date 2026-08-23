import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. Load Dataset
# ==========================================

data = pd.read_csv("datasets/regression/student_scores.csv")

print("========== DATASET ==========")
print(data)


# ==========================================
# 2. Select Features and Target
# ==========================================

X = data[["Hours", "Attendance", "Assignment"]]
y = data["Score"]


print("\n========== FEATURES ==========")
print(X)

print("\n========== TARGET ==========")
print(y)


# ==========================================
# 3. Split Data
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("\n========== TRAINING DATA ==========")
print(X_train)

print("\n========== TESTING DATA ==========")
print(X_test)


# ==========================================
# 4. Create Model
# ==========================================

model = LinearRegression()


# ==========================================
# 5. Train Model
# ==========================================

model.fit(X_train, y_train)


# ==========================================
# 6. Display Model Coefficients
# ==========================================

print("\n========== MODEL ==========")

print("Intercept:", model.intercept_)

print("Coefficient Hours      :", model.coef_[0])
print("Coefficient Attendance :", model.coef_[1])
print("Coefficient Assignment :", model.coef_[2])


# ==========================================
# 7. Display Equation
# ==========================================

print("\n========== EQUATION ==========")

print(
    f"Score = "
    f"{model.coef_[0]:.2f}(Hours) + "
    f"{model.coef_[1]:.2f}(Attendance) + "
    f"{model.coef_[2]:.2f}(Assignment) + "
    f"{model.intercept_:.2f}"
)


# ==========================================
# 8. Prediction
# ==========================================

y_pred = model.predict(X_test)

print("\n========== PREDICTION ==========")

result = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print(result)


# ==========================================
# 9. Model Evaluation
# ==========================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")

print("MAE  :", mae)
print("MSE  :", mse)
print("RMSE :", rmse)
print("R²   :", r2)


# ==========================================
# 10. Visualization
# ==========================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred
)

plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")

plt.title("Multiple Linear Regression")

plt.grid()

plt.show()