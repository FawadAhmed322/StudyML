from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from nn.models.model import Model
from nn.layers.linear_layer import LinearLayer
from core.activations.relu import ReLU
from core.optimizers.sgd import SGD
from core.losses.mse import MSE
from core.data.data_loader import DataLoader

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

train_loader = DataLoader(X_train, y_train)
val_loader = DataLoader(X_test, y_test, shuffle=False)

model = Model()
model.add_layer(LinearLayer(in_features=X_train.shape[1], out_features=128))
model.add_layer(ReLU())
model.add_layer(LinearLayer(in_features=128, out_features=y_train.shape[1]))
model.add_loss(MSE())
model.add_optimizer(SGD(lr=0.0003))

model.train(train_loader=train_loader, val_loader=val_loader, epochs=1000)
