import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

# Problem 1: Creating a 2-D convolutional layer
class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Conv2d, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        return nn.functional.conv2d(x, self.weight, self.bias)


# Problem 2: Experiments with 2D convolutional layers on small arrays
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

conv_layer = Conv2d(1, 2, (3, 3))
output = conv_layer(torch.FloatTensor(x))
print("Problem 2 Output:")
print(output.detach().numpy())

# Problem 3: Output size after 2-dimensional convolution
def calculate_output_size(N_in, P, F, S):
    return ((N_in + 2 * P - F) // S) + 1

N_in_h, N_in_w = 28, 28  # MNIST input size
P_h, P_w = 0, 0  # No padding
F_h, F_w = 3, 3  # Filter size
S_h, S_w = 1, 1  # Stride size

N_out_h = calculate_output_size(N_in_h, P_h, F_h, S_h)
N_out_w = calculate_output_size(N_in_w, P_w, F_w, S_w)

print("\nProblem 3 Output:")
print(f"N_out_h: {N_out_h}, N_out_w: {N_out_w}")

# Problem 4: Creation of maximum pooling layer
class MaxPool2D(nn.Module):
    def __init__(self, kernel_size):
        super(MaxPool2D, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        return nn.functional.max_pool2d(x, self.kernel_size)

# Problem 5: (Advance task) Creating average pooling
class AveragePool2D(nn.Module):
    def __init__(self, kernel_size):
        super(AveragePool2D, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        return nn.functional.avg_pool2d(x, self.kernel_size)

# Problem 6: Smoothing
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

    # Problem 7: (Advance assignment) Rewriting to PyTorch
    class PyTorchModelBinary(nn.Module):
        def __init__(self):
            super(PyTorchModelBinary, self).__init__()
            self.fc1 = nn.Linear(4, 10)
            self.fc2 = nn.Linear(10, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.sigmoid(self.fc2(x))
            return x

    # Convert data to PyTorch tensors
    X_binary_train_torch = torch.FloatTensor(X_binary_train)
    y_binary_train_torch = torch.FloatTensor(y_binary_train).view(-1, 1)

    X_binary_test_torch = torch.FloatTensor(X_binary_test)
    y_binary_test_torch = torch.FloatTensor(y_binary_test).view(-1, 1)

    # Initialize model, loss function, and optimizer
    model_binary_pytorch = PyTorchModelBinary()
    criterion_binary = nn.BCELoss()
    optimizer_binary = optim.Adam(model_binary_pytorch.parameters(), lr=0.001)

    # Training loop
    for epoch in range(10):
        optimizer_binary.zero_grad()
        output = model_binary_pytorch(X_binary_train_torch)
        loss = criterion_binary(output, y_binary_train_torch)
        loss.backward()
        optimizer_binary.step()

    # Evaluation
    with torch.no_grad():
        y_binary_pred_pytorch = (model_binary_pytorch(X_binary_test_torch) > 0.5).float()
        accuracy_binary_pytorch = accuracy_score(y_binary_test_torch.numpy(), y_binary_pred_pytorch.numpy())
        print(f"Iris Binary Classification Accuracy (PyTorch): {accuracy_binary_pytorch * 100:.2f}%")

    # Problem 8: (Advance assignment) Comparison of frameworks
    # Summary of differences between frameworks
    print("\nComparison of TensorFlow and PyTorch:")
    print("1. Calculation Speed: Both TensorFlow and PyTorch have similar computational performance.")
    print(
        "2. Number of Lines of Code and Readability: PyTorch often requires fewer lines of code and is considered more readable.")
    print(
        "3. Functions Provided: TensorFlow has a more extensive ecosystem with TensorFlow Extended (TFX), TensorFlow Lite, and TensorFlow.js, whereas PyTorch is preferred for research and experimentation due to its dynamic computation graph.")
