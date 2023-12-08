import numpy as np

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
class SimpleInitializer:
    def __init__(self, sigma):
        self.sigma = sigma

    def W(self, n_nodes1, n_nodes2):
        return self.sigma * np.random.randn(n_nodes1, n_nodes2)

    def B(self, n_nodes2):
        return self.sigma * np.random.randn(n_nodes2)
class SGD:
    def __init__(self, lr):
        self.lr = lr

    def update(self, layer, dW, dB):
        layer.W -= self.lr * dW
        layer.B -= self.lr * dB
        return layer
class Tanh:
    def forward(self, A):
        self.Z = np.tanh(A)
        return self.Z

    def backward(self, dZ):
        return dZ * (1 - self.Z**2)

class Softmax:
    def forward(self, A):
        exp_A = np.exp(A - np.max(A, axis=1, keepdims=True))
        self.Z = exp_A / np.sum(exp_A, axis=1, keepdims=True)
        return self.Z

    def backward(self, Z, Y):
        return Z - Y
class ReLU:
    def forward(self, A):
        self.Z = np.maximum(0, A)
        return self.Z

    def backward(self, dZ):
        return dZ * np.where(self.Z > 0, 1, 0)
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
class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.H_W = 0
        self.H_B = 0

    def update(self, layer, dW, dB):
        self.H_W += dW**2
        self.H_B += dB**2

        layer.W -= self.lr / np.sqrt(self.H_W + 1e-7) * dW
        layer.B -= self.lr / np.sqrt(self.H_B + 1e-7) * dB

        return layer
class ScratchDeepNeuralNetworkClassifier:
    def __init__(self, layers, epochs, batch_size, alpha=0.01):
        self.layers = layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.alpha = alpha
        self.losses = []

    def fit(self, X, y):
        for epoch in range(self.epochs):
            for i in range(0, len(X), self.batch_size):
                X_batch = X[i:i + self.batch_size]
                y_batch = y[i:i + self.batch_size]

                # Forward pass
                for layer in self.layers:
                    X_batch = layer.forward(X_batch)

                # Calculate loss and backward pass
                loss = self.cross_entropy_loss(X_batch, y_batch)
                self.losses.append(loss)

                dZ = self.softmax_backward(X_batch, y_batch)
                for layer in reversed(self.layers):
                    dZ = layer.backward(dZ)

    def predict(self, X):
        for layer in self.layers:
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
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load MNIST data
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist['data'].values.astype(np.float32), mnist['target'].values.astype(np.int)
X /= 255.0  # Normalize pixel values to the range [0, 1]

# One-hot encode the labels
encoder = OneHotEncoder(sparse=False, categories='auto')
y_onehot = encoder.fit_transform(y.reshape(-1, 1))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# Example usage:
model = ScratchDeepNeuralNetworkClassifier(layers=[
    FC(784, 100, XavierInitializer(), AdaGrad(0.01)),
    ReLU(),
    FC(100, 10, XavierInitializer(), AdaGrad(0.01)),
    Softmax()
], epochs=10, batch_size=64)

# Train the model
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = np.sum(y_pred == np.argmax(y_test, axis=1)) / len(y_test)
print(f"Accuracy on the test set: {accuracy * 100:.2f}%")
