import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load dataset (using California housing dataset for demonstration)
from sklearn.datasets import fetch_california_housing
california = fetch_california_housing()
X = california.data
y = california.target

# Use only one feature for simplicity (e.g., MedInc - median income in block group)
X = X[:, np.newaxis, 0]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a Linear Regression model
lr = LinearRegression()

# Train the model
lr.fit(X_train, y_train)

# Make predictions
y_pred = lr.predict(X_test)

# Compute Mean Squared Error and R-squared score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display the MSE and R-squared score
print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

# Plot regression line and data points
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression line')
plt.xlabel('Median Income in Block Group (MedInc)')
plt.ylabel('House Price')
plt.title('Linear Regression')
plt.legend()
plt.show()

