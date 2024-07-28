import numpy as np

class MSE:
    def __init__(self):
        pass

    def forward(self, predictions, targets):
        self.difference = predictions - targets
        loss = (self.difference.T @ self.difference) / predictions.shape[0]
        return loss

    def backward(self):
        d_input = 2 * self.difference / self.difference.shape[0]
        return d_input