
import numpy as np

class LinearLayer:
    def __init__(self, in_features, out_features, bias=True, trainable=True):
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.grad = None
        self.trainable = trainable
        if bias:
            self.in_features += 1
        self.weights = np.random.rand(self.in_features, out_features)

    def forward(self, X):
        if self.bias:
            X = np.hstack([np.ones(shape=(X.shape[0], 1)), X])
        self.X = X
        out = X @ self.weights
        return out

    def backward(self, d_output):
        if self.trainable:
            self.grad = self.X.T @ d_output
        d_input = d_output @ self.weights.T
        if self.bias:
            d_input = d_input[:, 1:]
        return d_input
    
    def update_weights(self, weights):
        self.weights = weights