import numpy as np

class BCE:
    def __init__(self, epsilon=1e-12):
        self.epsilon = epsilon

    def forward(self, predictions, targets):
        self.predictions = np.clip(predictions, self.epsilon, 1 - self.epsilon)
        self.targets = targets
        loss = -np.mean(targets * np.log(self.predictions) + (1 - targets) * np.log(1 - self.predictions))
        return loss

    def backward(self):
        d_input = (1 - self.targets) / (1 - self.predictions) - self.targets / self.predictions
        d_input /= self.targets.shape[0]  # Averaging the gradient over the batch size
        return d_input
