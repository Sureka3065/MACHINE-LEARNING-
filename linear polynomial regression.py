import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score


np.random.seed(0)
X = 2 - 3 * np.random.normal(0, 1, 100)
y = X - 2 * (X ** 2) + np.random.normal(-3, 3, 100)


X = X[:, np.newaxis]

linear_regressor = LinearRegression()
linear_regressor.fit(X, y)
y_pred_linear = linear_regressor.predict(X)

polynomial_features = PolynomialFeatures(degree=2)
X_poly = polynomial_features.fit_transform(X)

polynomial_regressor = LinearRegression()
polynomial_regressor.fit(X_poly, y)
y_pred_poly = polynomial_regressor.predict(X_poly)

plt.scatter(X, y, s=10, label='Data')
plt.plot(X, y_pred_linear, color='r', label='Linear Regression')
plt.plot(X, y_pred_poly, color='g', label='Polynomial Regression (degree=2)')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Linear vs Polynomial Regression')
plt.show()

print("Linear Regression:")
print("Mean Squared Error:", mean_squared_error(y, y_pred_linear))
print("R2 Score:", r2_score(y, y_pred_linear))

print("\nPolynomial Regression (degree 2):")
print("Mean Squared Error:", mean_squared_error(y, y_pred_poly))
print("R2 Score:", r2_score(y, y_pred_poly))
