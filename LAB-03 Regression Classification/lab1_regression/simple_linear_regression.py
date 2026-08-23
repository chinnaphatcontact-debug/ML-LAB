import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. Load Dataset
# ==========================================

data = pd.read_csv("datasets/regression/study_hours.csv")

print("========== DATASET ==========")
print(data)


# ==========================================
# 2. Select Features and Target
# ==========================================

X = data[["Hours"]]
y = data["Score"]


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
# 4. Create Linear Regression Model
# ==========================================

model = LinearRegression()


# ==========================================
# 5. Train Model
# ==========================================

model.fit(X_train, y_train)


# ==========================================
# 6. Display Model Equation
# ==========================================

print("\n========== MODEL ==========")

print("Slope (m):", model.coef_[0])
print("Intercept (b):", model.intercept_)

print(
    f"Equation: Score = {model.coef_[0]:.2f} * Hours + "
    f"{model.intercept_:.2f}"
)


# ==========================================
# 7. Prediction
# ==========================================

y_pred = model.predict(X_test)

print("\n========== PREDICTION ==========")

result = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print(result)


# ==========================================
# 8. Model Evaluation
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
# 9. Visualization
# ==========================================

plt.scatter(X, y, label="Actual Data")

plt.plot(
    X,
    model.predict(X),
    label="Regression Line"
)

plt.xlabel("Study Hours")
plt.ylabel("Exam Score")

plt.title("Simple Linear Regression")

plt.legend()

plt.grid()

plt.show()