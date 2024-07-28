from nn.layers.linear_layer import LinearLayer
from core.activations.sigmoid import Sigmoid
from core.losses.bce import BCE
from core.optimizers.sgd import SGD
import numpy as np

class LogisticRegression:
    def __init__(self):
        self.layer = None

    def fit(self, X, y, epochs=100, bias=True, lr=0.001, verbose=0):
        self.in_features = X.shape[1]
        self.out_features = 1
        self.layer = LinearLayer(self.in_features, self.out_features, bias)
        self.activation = Sigmoid()
        self.losses = []
        optimizer = SGD(lr=lr)
        loss_fn = BCE()

        for e in range(epochs):
            z = self.layer.forward(X)
            preds = self.activation.forward(z)
            loss = loss_fn.forward(preds, y)
            d_loss = loss_fn.backward()
            d_act = self.activation.backward(d_loss)
            grad_input = self.layer.backward(d_act)
            weights = optimizer.step(self.layer.weights, self.layer.grad)
            self.layer.update_weights(weights)
            self.losses.append(loss)
            if verbose == 1:
                print(f'Epoch: {e+1}/{epochs} Training Loss: {loss}')

    def predict(self, X):
        pred = self.layer.forward(X)
        pred = self.activation.forward(pred)
        return pred