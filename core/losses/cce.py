import numpy as np

class CCE:
    def __init__(self):
        pass

    def forward(self, preds, labels):
        # Compute the cross-entropy loss
        self.preds = preds
        self.labels = labels
        loss = -np.sum(labels * np.log(preds + 1e-15)) / preds.shape[0]  # Adding epsilon for numerical stability
        return loss

    def backward(self):
        # Gradient of the loss with respect to the logits
        dL_dx = (self.preds - self.labels) / self.labels.shape[0]
        return dL_dx