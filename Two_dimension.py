import numpy as np
import matplotlib.pyplot as plt

# Problem 1: Linear function
def function1(array_x):
    return 0.5 * array_x + 1

x_values = np.arange(-50, 50.1, 0.1)
y_values = function1(x_values)

# Problem 2: Array combination
array_xy = np.column_stack((x_values, y_values))

# Problem 3: Find the gradient
gradient = np.diff(y_values) / np.diff(x_values)

# Problem 4: Draw a graph
plt.figure(figsize=(10, 5))

plt.subplot(121)
plt.plot(x_values, y_values, label='Linear Function')
plt.title('Linear Function')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.subplot(122)
plt.plot(x_values[:-1], gradient, label='Gradient')
plt.title('Gradient')
plt.xlabel('x')
plt.ylabel('Gradient')
plt.legend()

plt.show()

# Problem 5: Python functionalization
def compute_gradient(function, x_range=(-50, 50.1, 0.1)):
    array_x = np.arange(x_range[0], x_range[1], x_range[2])
    array_y = function(array_x)
    array_xy = np.column_stack((array_x, array_y))
    gradient = np.diff(array_y) / np.diff(array_x)
    return array_xy, gradient

# Example usage for the three equations
def function2(array_x):
    return array_x**2

def function3(array_x):
    return 2 * array_x**2 + 2 * array_x

def function4(array_x):
    return np.sin(array_x**2)

array_xy_2, gradient_2 = compute_gradient(function2)
array_xy_3, gradient_3 = compute_gradient(function3)
array_xy_4, gradient_4 = compute_gradient(function4, x_range=(0, 50.1, 0.1))

# Draw graphs for the three equations
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.plot(array_xy_2[:, 0], array_xy_2[:, 1], label='Function 2')
plt.title('Function 2')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.subplot(132)
plt.plot(array_xy_3[:, 0], array_xy_3[:, 1], label='Function 3')
plt.title('Function 3')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.subplot(133)
plt.plot(array_xy_4[:, 0], array_xy_4[:, 1], label='Function 4')
plt.title('Function 4')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.show()

# Problem 6: Find the minimum value
min_y_index = np.argmin(array_xy_4[:, 1])
min_y_value = np.min(array_xy_4[:, 1])

print(f"Minimum value of y: {min_y_value}")
print(f"Index of minimum value: {min_y_index}")
print(f"Slope before minimum: {gradient_4[min_y_index - 1]}")
print(f"Slope after minimum: {gradient_4[min_y_index]}")
