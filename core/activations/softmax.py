import numpy as np

class Softmax:
    def __init__(self, trainable=False):
        self.trainable = trainable

    def forward(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Subtract max for numerical stability
        self.preds = exp_x / exp_x.sum(axis=1, keepdims=True)  # Sum along the correct axis
        return self.preds

    def backward(self, dL_dy):
        # dL_dy is the gradient of the loss with respect to the output of the softmax
        dx = np.empty_like(self.preds)
        for i, (softmax_output, dL_dy_row) in enumerate(zip(self.preds, dL_dy)):
            # Compute Jacobian matrix for each sample in the batch
            J = np.diagflat(softmax_output) - np.outer(softmax_output, softmax_output)
            # Compute the gradient with respect to the input
            dx[i] = np.dot(J, dL_dy_row)
        return dx
