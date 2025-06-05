import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

boston_df = pd.read_csv("Housing.csv", na_values="?").dropna()

X_boston, y_boston = boston_df[["RM"]], boston_df["MEDV"]

X_train, X_test, y_train, y_test = train_test_split(
    X_boston, y_boston, test_size=0.2, random_state=42
)

linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
y_pred = linear_reg.predict(X_test)

plt.figure(figsize=(10, 5))
plt.scatter(X_test, y_test, color="blue", label="Actual")
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Predicted")
plt.xlabel("Average Number of Rooms (RM)")
plt.ylabel("Housing Price")
plt.title("Linear Regression on Boston Housing Dataset")
plt.legend()
plt.show()

print(f"Mean Squared Error (Linear Regression): {mean_squared_error(y_test, y_pred)}")

auto_df = pd.read_csv("auto_mpg.csv", na_values="?").dropna()

X_auto = auto_df[["horsepower"]]
y_auto = auto_df["mpg"]

X_train, X_test, y_train, y_test = train_test_split(X_auto, y_auto, test_size=0.2)

poly_model = make_pipeline(PolynomialFeatures(3), StandardScaler(), LinearRegression())
poly_model.fit(X_train, y_train)
y_poly_pred = poly_model.predict(X_test)

X_test_sorted, y_poly_pred_sorted = zip(
    *sorted(zip(X_test.values.flatten(), y_poly_pred))
)

plt.figure(figsize=(10, 5))
plt.scatter(X_test, y_test, color="blue", label="Actual")
plt.plot(X_test_sorted, y_poly_pred_sorted, color="red", linewidth=2, label="Predicted")
plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.title("Polynomial Regression")
plt.legend()
plt.show()

print(f"MSE Polynomial: {mean_squared_error(y_test, y_poly_pred):.3f}")
