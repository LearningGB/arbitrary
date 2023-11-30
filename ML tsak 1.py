import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class ScratchLinearRegression():
    def __init__(self, num_iter, lr, no_bias, verbose):
        self.iter = num_iter
        self.lr = lr
        self.no_bias = no_bias
        self.verbose = verbose
        self.loss = np.zeros(self.iter)
        self.val_loss = np.zeros(self.iter)
        self.coef_ = None

    def _linear_hypothesis(self, X):
        return X @ self.coef_

    def _gradient_descent(self, X, error):
        gradient = X.T @ error / len(X)
        self.coef_ -= self.lr * gradient

    def fit(self, X, y, X_val=None, y_val=None):
        if not self.no_bias:
            X = np.hstack([np.ones((len(X), 1)), X])
            if X_val is not None:
                X_val = np.hstack([np.ones((len(X_val), 1)), X_val])

        self.coef_ = np.zeros(X.shape[1])

        for i in range(self.iter):
            error = self._linear_hypothesis(X) - y
            self._gradient_descent(X, error)
            self.loss[i] = mean_squared_error(y, self._linear_hypothesis(X))

            if X_val is not None:
                val_error = self._linear_hypothesis(X_val) - y_val
                self.val_loss[i] = mean_squared_error(y_val, self._linear_hypothesis(X_val))

            if self.verbose and (i % 100 == 0):
                print(f"Iteration {i + 1}/{self.iter} | Training Loss: {self.loss[i]}")

    def predict(self, X):
        if not self.no_bias:
            X = np.hstack([np.ones((len(X), 1)), X])
        return self._linear_hypothesis(X)

def MSE(y_pred, y):
    return np.mean((y_pred - y) ** 2)

# Problem 1
# _linear_hypothesis method added to the ScratchLinearRegression class

# Problem 2
# _gradient_descent method added to the ScratchLinearRegression class

# Problem 3
# predict method added to the ScratchLinearRegression class

# Problem 4
def MSE(y_pred, y):
    return np.mean((y_pred - y) ** 2)

# Problem 5
# fit method updated to record self.loss and self.val_loss

# Problem 6
# Load the data
# Assuming X_train, X_test, y_train, y_test are loaded from the dataset

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Instantiate and fit the ScratchLinearRegression model
model = ScratchLinearRegression(num_iter=1000, lr=0.001, no_bias=False, verbose=True)
model.fit(X_train, y_train, X_val, y_val)

# Compare with scikit-learn implementation
sklearn_model = LinearRegression()
sklearn_model.fit(X_train, y_train)
sklearn_pred = sklearn_model.predict(X_val)

# Compare the coefficients
print("Comparison of coefficients:")
print("Scratch Coefficients:", model.coef_)
print("Scikit-Learn Coefficients:", np.insert(sklearn_model.coef_, 0, sklearn_model.intercept_))

# Compare the mean squared error
scratch_mse = MSE(model.predict(X_val), y_val)
sklearn_mse = mean_squared_error(sklearn_pred, y_val)

print("\nComparison of Mean Squared Error:")
print("Scratch MSE:", scratch_mse)
print("Scikit-Learn MSE:", sklearn_mse)

# Problem 7
# Plotting the learning curve
plt.plot(model.loss, label='Training Loss')
plt.plot(model.val_loss, label='Validation Loss')
plt.title('Learning Curve')
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()

# Problem 8 (Advance task)
# Bias term removal experiment
model_no_bias = ScratchLinearRegression(num_iter=1000, lr=0.001, no_bias=True, verbose=False)
model_no_bias.fit(X_train, y_train, X_val, y_val)
no_bias_mse = MSE(model_no_bias.predict(X_val), y_val)

print("\nComparison without Bias Term:")
print("Scratch MSE (with bias):", scratch_mse)
print("Scratch MSE (without bias):", no_bias_mse)

# Problem 9 (Advance task)
# Multidimensional feature quantity experiment
X_train_squared = X_train ** 2
X_val_squared = X_val ** 2

model_squared = ScratchLinearRegression(num_iter=1000, lr=0.001, no_bias=False, verbose=False)
model_squared.fit(X_train_squared, y_train, X_val_squared, y_val)
squared_mse = MSE(model_squared.predict(X_val_squared), y_val)

print("\nComparison with Squared Features:")
print("Scratch MSE (original):", scratch_mse)
print("Scratch MSE (squared features):", squared_mse)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class ScratchLinearRegression():
    def __init__(self, num_iter, lr, no_bias, verbose):
        self.iter = num_iter
        self.lr = lr
        self.no_bias = no_bias
        self.verbose = verbose
        self.loss = np.zeros(self.iter)
        self.val_loss = np.zeros(self.iter)
        self.coef_ = None

    def _linear_hypothesis(self, X):
        return X @ self.coef_

    def _gradient_descent(self, X, error):
        gradient = X.T @ error / len(X)
        self.coef_ -= self.lr * gradient

    def fit(self, X, y, X_val=None, y_val=None):
        if not self.no_bias:
            X = np.hstack([np.ones((len(X), 1)), X])
            if X_val is not None:
                X_val = np.hstack([np.ones((len(X_val), 1)), X_val])

        self.coef_ = np.zeros(X.shape[1])

        for i in range(self.iter):
            error = self._linear_hypothesis(X) - y
            self._gradient_descent(X, error)
            self.loss[i] = mean_squared_error(y, self._linear_hypothesis(X))

            if X_val is not None:
                val_error = self._linear_hypothesis(X_val) - y_val
                self.val_loss[i] = mean_squared_error(y_val, self._linear_hypothesis(X_val))

            if self.verbose and (i % 100 == 0):
                print(f"Iteration {i + 1}/{self.iter} | Training Loss: {self.loss[i]}")

    def predict(self, X):
        if not self.no_bias:
            X = np.hstack([np.ones((len(X), 1)), X])
        return self._linear_hypothesis(X)

def MSE(y_pred, y):
    return np.mean((y_pred - y) ** 2)

# Problem 1 to 9 (Code provided in the previous response)

# Problem 10 (Advance task)
# Derivation of update formula
# The update formula is derived from the gradient of the mean squared error (MSE) with respect to the parameters.
# Derivative of J(θ) w.r.t. θ_j is given by:
# ∂/∂θ_j J(θ) = 1/m * ∑(hθ(xi) - yi) * xi_j
# Update formula: θ_j := θ_j - α * ∂/∂θ_j J(θ)
# This leads to the _gradient_descent method in the class.

# Problem 11 (Advance task)
# Problem of local optimum solution
# In linear regression, the MSE loss function is convex, meaning it has only one global minimum.
# The steepest descent method may encounter local minima in non-convex problems, but for linear regression, it's not an issue.
# The mathematical proof involves showing that the MSE loss function is a convex quadratic function.
# Visualization of a convex loss function guarantees that the steepest descent will converge to the global minimum.
# However, in non-convex problems, there might be multiple local minima, making convergence to a global minimum difficult.

# Let's visualize the convexity for linear regression
def visualize_convexity():
    theta_values = np.linspace(-5, 5, 100)
    J_values = np.zeros_like(theta_values)

    for i, theta in enumerate(theta_values):
        J_values[i] = np.mean((X_train @ np.array([theta, 1]) - y_train) ** 2) / 2

    plt.plot(theta_values, J_values)
    plt.title('Convexity of Mean Squared Error (Linear Regression)')
    plt.xlabel('Theta')
    plt.ylabel('Mean Squared Error (J)')
    plt.show()

visualize_convexity()

