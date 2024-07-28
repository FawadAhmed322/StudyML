from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from core.models.linear_model import LogisticRegression  # Assuming you have a LogisticRegression implementation in your core.models.linear_model module
import numpy as np

# Load the dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and testing data
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = y_train.reshape(-1, 1)  # Reshape to (n_samples, 1) if necessary
y_test = y_test.reshape(-1, 1)

# Initialize the model
model = LogisticRegression()

# Fit the model to the training data
model.fit(X_train, y_train, epochs=10000, lr=0.03, verbose=1)

# Predict using the model on the test data
predictions = model.predict(X_test)

# Convert probabilities to binary predictions with a threshold of 0.5
binary_predictions = (predictions > 0.5).astype(int)

# Print the first 5 predictions and corresponding true values
print("First 5 predictions:\n", binary_predictions[:5])
print("First 5 true values:\n", y_test[:5])

# Calculate and print the accuracy of the model
accuracy = np.mean(binary_predictions == y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
