import numpy as np

class ReLU:
    def __init__(self, trainable=False):
        self.trainable=trainable

    def forward(self, X):
        self.X = X
        out = np.where(X > 0, X, 0)
        return out

    def backward(self, d_output):
        d_input = np.where(self.X > 0, 1, 0) * d_output
        return d_input