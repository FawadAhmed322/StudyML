import numpy as np

class DataLoader:
    def __init__(self, X, y, batch_size=32, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.indices = np.arange(len(X))
        if shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= len(self.X):
            raise StopIteration  # Signal that iteration is complete
        
        end_idx = min(self.current_idx + self.batch_size, len(self.X))
        batch_indices = self.indices[self.current_idx:end_idx]
        batch_X = self.X[batch_indices]
        batch_y = self.y[batch_indices]
        self.current_idx = end_idx
        return batch_X, batch_y

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))