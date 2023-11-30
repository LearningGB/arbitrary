import numpy as np

# [Problem 1] Classifying fully connected layers
class FC:
    def __init__(self, n_nodes1, n_nodes2, initializer, optimizer):
        self.optimizer = optimizer
        self.W = initializer.W(n_nodes1, n_nodes2)
        self.B = initializer.B(n_nodes2)
        self.X = None

    def forward(self, X):
        self.X = X
        A = np.dot(X, self.W) + self.B
        return A

    def backward(self, dA):
        dZ = np.dot(dA, self.W.T)
        dW = np.dot(self.X.T, dA)
        dB = np.sum(dA, axis=0)
        self = self.optimizer.update(self, dW, dB)
        return dZ

# [Problem 2] Classifying the initialization method
class SimpleInitializer:
    def __init__(self, sigma):
        self.sigma = sigma

    def W(self, n_nodes1, n_nodes2):
        return self.sigma * np.random.randn(n_nodes1, n_nodes2)

    def B(self, n_nodes2):
        return self.sigma * np.random.randn(n_nodes2)

# [Problem 3] Classifying optimization methods
class SGD:
    def __init__(self, lr):
        self.lr = lr

    def update(self, layer, dW, dB):
        layer.W -= self.lr * dW
        layer.B -= self.lr * dB
        return layer

# [Problem 4] Classifying activation functions
class Tanh:
    def forward(self, A):
        self.Z = np.tanh(A)
        return self.Z

    def backward(self, dZ):
        return dZ * (1 - self.Z**2)

class ReLU:
    def forward(self, A):
        self.Z = np.maximum(0, A)
        return self.Z

    def backward(self, dZ):
        return dZ * np.where(self.Z > 0, 1, 0)

class Softmax:
    def forward(self, A):
        exp_A = np.exp(A - np.max(A, axis=1, keepdims=True))
        self.Z = exp_A / np.sum(exp_A, axis=1, keepdims=True)
        return self.Z

    def backward(self, Z, Y):
        return Z - Y

# [Problem 5] ReLU class creation (Already implemented above)

# [Problem 6] Initial value of weight
class XavierInitializer:
    def W(self, n_nodes1, n_nodes2):
        return np.random.randn(n_nodes1, n_nodes2) / np.sqrt(n_nodes1)

    def B(self, n_nodes2):
        return np.random.randn(n_nodes2) / np.sqrt(n_nodes2)

class HeInitializer:
    def W(self, n_nodes1, n_nodes2):
        return np.random.randn(n_nodes1, n_nodes2) * np.sqrt(2 / n_nodes1)

    def B(self, n_nodes2):
        return np.random.randn(n_nodes2) * np.sqrt(2 / n_nodes2)

# [Problem 7] Optimization method (Already implemented above)

# [Problem 8] Class completion
class ScratchDeepNeuralNetworkClassifier:
    def __init__(self, layers, epochs, batch_size, alpha=0.01):
        # Initialize network parameters
        self.layers = layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.alpha = alpha
        self.losses = []

        # Initialize layers
        self.network = []
        for layer in self.layers:
            self.network.append(layer)

    def fit(self, X, y):
        for epoch in range(self.epochs):
            # Mini-batch processing
            for i in range(0, len(X), self.batch_size):
                X_batch = X[i:i + self.batch_size]
                y_batch = y[i:i + self.batch_size]

                # Forward pass
                for layer in self.network:
                    X_batch = layer.forward(X_batch)

                # Calculate loss and backward pass
                loss = self.cross_entropy_loss(X_batch, y_batch)
                self.losses.append(loss)

                dZ = self.softmax_backward(X_batch, y_batch)
                for layer in reversed(self.network):
                    dZ = layer.backward(dZ)

    def predict(self, X):
        for layer in self.network:
            X = layer.forward(X)
        return np.argmax(X, axis=1)

    def cross_entropy_loss(self, y_pred, y_true):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        N = len(y_pred)
        loss = -np.sum(y_true * np.log(y_pred + 1e-9)) / N
        return loss

    def softmax_backward(self, y_pred, y_true):
        return y_pred - y_true

# [Problem 9] Learning and estimation (Training and testing with MNIST)
# ... (Code to load MNIST data and preprocess)

# Example usage:
# model = ScratchDeepNeuralNetworkClassifier(layers=[FC(784, 100, XavierInitializer(), AdaGrad(0.01)),
#                                                    ReLU(),
#
