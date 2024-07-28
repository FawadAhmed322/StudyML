from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from core.models.linear_model import LinearRegression
import numpy as np

# Load the dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the scalers
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# Fit the scaler on the training data and transform both training and testing data
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

# Reshape y for scaling, fit the scaler on the training data, and transform both training and testing data
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

# Initialize the model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train, epochs=10000, lr=0.03, verbose=1)

# Predict using the model on the test data
# predictions = model.forward(X_test)

# Print the first 5 predictions and corresponding true values
# print("First 5 predictions:\n", predictions[:5])
# print("First 5 true values:\n", y_test[:5])
