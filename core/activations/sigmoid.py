import numpy as np

class Sigmoid:
    def __init__(self, trainable=False):
        self.trainable = trainable
        self.out = None
        
    def forward(self, X):
        self.out = 1 / (1 + np.exp(-X))
        return self.out

    def backward(self, d_output):
        d_input = d_output * (self.out * (1 - self.out))
        return d_input
