from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from nn.models.model import Model
from nn.layers.linear_layer import LinearLayer
from core.activations.relu import ReLU
from core.activations.softmax import Softmax
from core.optimizers.sgd import SGD
from core.losses.cce import CCE
from core.data.data_loader import DataLoader

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and testing data
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# One-hot encode the labels
encoder = OneHotEncoder(sparse_output=False)  # Correct parameter name
y_train = encoder.fit_transform(y_train.reshape(-1, 1))
y_test = encoder.transform(y_test.reshape(-1, 1))

# Create DataLoader instances
train_loader = DataLoader(X_train, y_train)
val_loader = DataLoader(X_test, y_test, shuffle=False)

# Initialize the model
model = Model()
model.add_layer(LinearLayer(in_features=X_train.shape[1], out_features=64))
model.add_layer(ReLU())
model.add_layer(LinearLayer(in_features=64, out_features=y_train.shape[1]))  # Number of output features should match number of classes
model.add_layer(Softmax())
model.add_loss(CCE())
model.add_optimizer(SGD(lr=0.0003))

# Train the model
model.train(train_loader=train_loader, val_loader=val_loader, epochs=10000)
