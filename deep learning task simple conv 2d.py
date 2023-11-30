import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.datasets import mnist
from keras.utils import to_categorical


# Problem 1: 2D Convolutional Layer
class Conv2d:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        # Implementation of initialization
        pass

    def forward(self, x):
        # Implementation of forward propagation
        pass

    def backward(self, delta):
        # Implementation of backward propagation
        pass


# Problem 2: Experiments with 2D Convolutional Layers
# Creating input data and weight
x = np.array([[[[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16]]]])
w = np.array([[[0., 0., 0.],
               [0., 1., 0.],
               [0., -1., 0.]],
              [[0., 0., 0.],
               [0., -1., 1.],
               [0., 0., 0.]]])
# Creating Conv2d instance and testing forward and backward
conv2d_layer = Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0)
output = conv2d_layer.forward(x)
delta = np.array([[[[-4, -4],
                    [10, 11]],
                   [[1, -7],
                    [1, -11]]]])
gradient = conv2d_layer.backward(delta)


# Problem 3: Output size after 2D convolution
def calculate_output_size(input_size, kernel_size, stride, padding):
    # Implementation of output size calculation
    pass


# Problem 4: Creation of Maximum Pooling Layer
class MaxPool2D:
    def __init__(self, pool_size, stride):
        # Implementation of initialization
        pass

    def forward(self, x):
        # Implementation of forward propagation
        pass

    def backward(self, delta):
        # Implementation of backward propagation
        pass


# Problem 5: Creating Average Pooling Layer
class AveragePool2D:
    def __init__(self, pool_size, stride):
        # Implementation of initialization
        pass

    def forward(self, x):
        # Implementation of forward propagation
        pass

    def backward(self, delta):
        # Implementation of backward propagation
        pass


# Problem 6: Smoothing
class Flatten:
    def forward(self, x):
        # Implementation of forward propagation
        pass

    def backward(self, delta):
        # Implementation of backward propagation
        pass


# Problem 7: Learning and Estimation
class Scratch2dCNNClassifier:
    def __init__(self, conv_params, pool_params, hidden_size, output_size, epochs=10, learning_rate=0.01):
        # Implementation of initialization
        pass

    def fit(self, X, y, X_val=None, y_val=None):
        # Implementation of training
        pass

    def predict(self, X):
        # Implementation of prediction
        pass


# Problem 8: LeNet
class LeNet(Scratch2dCNNClassifier):
    def __init__(self, input_size, output_size, epochs=10, learning_rate=0.01):
        # Implementation of LeNet architecture
        pass


# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess the data
X_train = X_train.reshape(-1, 1, 28, 28) / 255.0
X_test = X_test.reshape(-1, 1, 28, 28) / 255.0
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Create and train LeNet
lenet = LeNet(input_size=(1, 28, 28), output_size=10, epochs=5, learning_rate=0.01)
lenet.fit(X_train, y_train, X_val, y_val)

# Make predictions on the test set
y_pred_lenet = lenet.predict(X_test)

# Calculate accuracy
accuracy_lenet = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred_lenet, axis=1))
print(f"LeNet Accuracy: {accuracy_lenet * 100:.2f}%")
