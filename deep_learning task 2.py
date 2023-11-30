import numpy as np

class XavierInitializer:
    def initialize(self, shape):
        return np.random.randn(*shape) / np.sqrt(shape[1])

class SimpleConv1d:
    def __init__(self, in_channels, out_channels, kernel_size, initializer):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weights = initializer.initialize((out_channels, in_channels, kernel_size))
        self.bias = initializer.initialize((out_channels,))
        self.grad_w = np.zeros_like(self.weights)
        self.grad_b = np.zeros_like(self.bias)
        self.input = None

    def forward(self, x):
        self.input = x
        output_size = len(x) - self.kernel_size + 1
        a = np.zeros((self.out_channels, output_size))

        for i in range(output_size):
            for j in range(self.kernel_size):
                a[:, i] += x[i + j] * self.weights[:, :, j]

            a[:, i] += self.bias

        return a

    def backward(self, delta_a, learning_rate):
        delta_x = np.zeros_like(self.input)

        for s in range(self.kernel_size):
            indexes = np.arange(s, len(self.input) - self.kernel_size + 1 + s)
            delta_x[indexes] += np.sum(delta_a * self.weights[:, :, s], axis=0)

        self.grad_b = np.sum(delta_a, axis=1)
        self.grad_w = np.sum(delta_a[:, :, None] * self.input[None, :, :], axis=2)

        self.weights -= learning_rate * self.grad_w
        self.bias -= learning_rate * self.grad_b

        return delta_x

class FullyConnectedLayer:
    def __init__(self, input_size, output_size, initializer):
        self.weights = initializer.initialize((input_size, output_size))
        self.bias = initializer.initialize((output_size,))
        self.grad_w = np.zeros_like(self.weights)
        self.grad_b = np.zeros_like(self.bias)
        self.input = None

    def forward(self, x):
        self.input = x
        return np.dot(x, self.weights) + self.bias

    def backward(self, delta_a, learning_rate):
        delta_x = np.dot(delta_a, self.weights.T)
        self.grad_b = np.sum(delta_a, axis=0)
        self.grad_w = np.dot(self.input.T, delta_a)

        self.weights -= learning_rate * self.grad_w
        self.bias -= learning_rate * self.grad_b

        return delta_x

class Scratch1dCNNClassifier:
    def __init__(self, conv_layer, fc_layer):
        self.conv_layer = conv_layer
        self.fc_layer = fc_layer

    def fit(self, x, t, epochs, learning_rate):
        for epoch in range(epochs):
            # Forward pass
            a_conv = self.conv_layer.forward(x)
            a_fc = self.fc_layer.forward(a_conv.flatten())

            # Loss calculation (Assuming softmax cross-entropy loss for classification)
            loss = self.softmax_cross_entropy_loss(a_fc, t)

            # Backward pass
            delta_a_fc = self.softmax_cross_entropy_loss_backward(a_fc, t)
            delta_a_conv_flat = self.fc_layer.backward(delta_a_fc, learning_rate)
            delta_a_conv = delta_a_conv_flat.reshape(self.conv_layer.out_channels, -1)
            self.conv_layer.backward(delta_a_conv, learning_rate)

            if epoch % 10 == 0:
                print(f"Epoch: {epoch}, Loss: {loss}")

    def predict(self, x):
        a_conv = self.conv_layer.forward(x)
        a_fc = self.fc_layer.forward(a_conv.flatten())
        return np.argmax(a_fc, axis=1)

    def softmax_cross_entropy_loss(self, a, t):
        exp_a = np.exp(a - np.max(a, axis=1, keepdims=True))
        sum_exp_a = np.sum(exp_a, axis=1, keepdims=True)
        y = exp_a / sum_exp_a
        loss = -np.sum(t * np.log(y + 1e-7)) / len(t)
        return loss

    def softmax_cross_entropy_loss_backward(self, a, t):
        batch_size = len(t)
        return (a - t) / batch_size

def calculate_output_size(input_size, kernel_size, padding, stride):
    return (input_size + 2 * padding - kernel_size) // stride + 1

def apply_padding(x, padding_value):
    return np.pad(x, ((0, 0), (padding_value, padding_value)), mode='constant', constant_values=padding_value)

class Conv1dStrided(SimpleConv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride, initializer):
        super().__init__(in_channels, out_channels, kernel_size, initializer)
        self.stride = stride

    def forward(self, x):
        self.input = x
        batch_size, input_size = x.shape
        output_size = calculate_output_size(input_size, self.kernel_size, padding=0, stride=self.stride)
        a = np.zeros((batch_size, self.out_channels, output_size))

        for b in range(batch_size):
            for i in range(0, input_size - self.kernel_size + 1, self.stride):
                for j in range(self.kernel_size):
                    a[b, :, i // self.stride] += x[b, i + j] * self.weights[:, :, j]

                a[b, :, i // self.stride] += self.bias

        return a

    def backward(self, delta_a, learning_rate):
        batch_size, input_size = self.input.shape
        delta_x = np.zeros_like(self.input)
        output_size = calculate_output_size(input_size, self.kernel_size, padding=0, stride=self.stride)

        for b in range(batch_size):
            for s in range(self.kernel_size):
                indexes = np.arange(s, output_size * self.stride, self.stride)
                delta_x[b, indexes] += np.sum(delta_a[b, :, None] * self.weights[:, :, s][None, :, :], axis=2)

        self.grad_b = np.sum(delta_a, axis=(0, 2))
        self.grad_w = np.sum(delta_a[:, :, None, None] * self.input[:, None, :, None], axis=(0, 3))

        self.weights -= learning_rate * self.grad_w
        self.bias -= learning_rate * self.grad_b

        return delta_x

# Example usage
# Assuming you have the MNIST dataset (x_train, t_train, x_test, t_test)

# Reshape the data to be suitable for 1D convolution
x_train = x_train.reshape(-1, 1, 28 * 28)
x_test = x_test.reshape(-1, 1, 28 * 28)

# Initialize the layers
conv_layer = Conv1dStrided(in_channels=1, out_channels=5, kernel_size=3, stride=2, initializer=XavierInitializer())
fc_layer = FullyConnectedLayer(input_size=5 * 13, output_size=10, initializer=XavierInitializer())

# Initialize the CNN classifier
classifier = Scratch1dCNNClassifier(conv_layer, fc_layer)

# Train the model
classifier.fit(x
