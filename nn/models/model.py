class Model:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.optimizer = None

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, d_loss):
        d_out = d_loss
        for layer in reversed(self.layers):
            d_out = layer.backward(d_out)
            if layer.trainable:
                layer.weights = self.optimizer.step(layer.weights, layer.grad)
        return d_out

    def add_layer(self, layer):
        self.layers.append(layer)

    def add_loss(self, loss):
        self.loss = loss

    def add_optimizer(self, optimizer):
        self.optimizer = optimizer

    def train(self, train_loader, val_loader=None, epochs=None):
        for e in range(epochs):
            epoch_loss = 0
            for batch_x, batch_y in train_loader:
                batch_preds = self.forward(batch_x)
                batch_loss = self.loss.forward(batch_preds, batch_y)
                epoch_loss += batch_loss
                d_loss = self.loss.backward()
                self.backward(d_loss)

            # Optionally print the loss for the current epoch
            print(f"\nEpoch {e+1}/{epochs}, Loss: {epoch_loss/len(train_loader)}")

            # Optionally validate on validation set
            if val_loader is not None:
                val_loss = 0
                for val_x, val_y in val_loader:
                    val_preds = self.forward(val_x)
                    val_loss += self.loss.forward(val_preds, val_y)
                print(f"Validation Loss: {val_loss/len(val_loader)}")

        # Return the final loss for the last epoch
        return epoch_loss / len(train_loader)