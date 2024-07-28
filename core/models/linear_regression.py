from nn.layers.linear_layer import LinearLayer
from core.losses.mse import MSE
from core.optimizers.sgd import SGD
import numpy as np

class LinearRegression:
    def __init__(self):
        self.layer = None

    def fit(self, X, y, epochs=100, bias=True, lr=0.001, verbose=0):
        self.in_features = X.shape[1]
        self.out_features = y.shape[1]
        self.layer = LinearLayer(self.in_features, self.out_features, bias)
        self.losses = []
        optimizer = SGD(lr=lr)
        loss_fn = MSE()

        for e in range(epochs):
            predictions = self.layer.forward(X)
            loss = loss_fn.forward(predictions, y)
            d_loss = loss_fn.backward()
            grad_input = self.layer.backward(d_loss)
            weights = optimizer.step(self.layer.weights, self.layer.grad)
            self.layer.update_weights(weights)
            self.losses.append(loss)
            if verbose == 1:
                print(f'Epoch: {e+1}/{epochs} Training Loss: {loss}')

    def predict(self, X):
        pred = self.layer.forward(X)
        return pred