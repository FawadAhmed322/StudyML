from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from nn.models.model import Model
from nn.layers.linear_layer import LinearLayer
from core.activations.relu import ReLU
from core.activations.sigmoid import Sigmoid
from core.optimizers.sgd import SGD
from core.losses.bce import BCE
from core.data.data_loader import DataLoader

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Convert the problem to binary classification (class 0 vs class 1 & 2)
y = (y == 0).astype(int)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and testing data
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape y for training
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Create DataLoader instances
train_loader = DataLoader(X_train, y_train)
val_loader = DataLoader(X_test, y_test, shuffle=False)

# Initialize the model
model = Model()
model.add_layer(LinearLayer(in_features=X_train.shape[1], out_features=64))
model.add_layer(ReLU())
model.add_layer(LinearLayer(in_features=64, out_features=1))
model.add_layer(Sigmoid())
model.add_loss(BCE())
model.add_optimizer(SGD(lr=0.0003))

# Train the model
model.train(train_loader=train_loader, val_loader=val_loader, epochs=10000)
