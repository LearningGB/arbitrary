import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

class ScratchLogisticRegression():
    def __init__(self, num_iter, lr, bias, verbose, reg_param=0.1):
        self.iter = num_iter
        self.lr = lr
        self.bias = bias
        self.verbose = verbose
        self.reg_param = reg_param
        self.loss = np.zeros(self.iter)
        self.val_loss = np.zeros(self.iter)
        self.coef_ = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _gradient_descent(self, X, error):
        gradient = np.dot(X.T, error) / len(X)
        self.coef_ -= self.lr * gradient

    def _compute_regularization_term(self):
        return (self.reg_param / (2 * len(self.coef_))) * np.sum(self.coef_[1:]**2)

    def _compute_loss(self, h, y):
        regularization_term = self._compute_regularization_term()
        return -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h)) + regularization_term

    def fit(self, X, y, X_val=None, y_val=None):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        if self.verbose:
            print("Training...")

        self.coef_ = np.zeros(X_train.shape[1])

        for i in range(self.iter):
            z = np.dot(X_train, self.coef_)
            h = self._sigmoid(z)
            error = h - y_train

            self._gradient_descent(X_train, error)

            # Compute training loss
            self.loss[i] = self._compute_loss(h, y_train)

            # Compute validation loss
            if X_val is not None and y_val is not None:
                z_val = np.dot(X_val, self.coef_)
                h_val = self._sigmoid(z_val)
                self.val_loss[i] = self._compute_loss(h_val, y_val)

            if self.verbose and (i % 100 == 0 or i == self.iter - 1):
                print(f"Iteration {i}: Training Loss = {self.loss[i]:.4f}, Validation Loss = {self.val_loss[i]:.4f}")

    def predict(self, X):
        return np.round(self._sigmoid(np.dot(X, self.coef_)))

    def predict_proba(self, X):
        return self._sigmoid(np.dot(X, self.coef_))

# Problem 5: Learning and estimation
# Assuming X_train and y_train are your training data and labels
X_train = np.random.rand(100, 2)
y_train = np.random.randint(0, 2, size=100)

# Create an instance of ScratchLogisticRegression
lr_scratch = ScratchLogisticRegression(num_iter=1000, lr=0.01, bias=True, verbose=True, reg_param=0.1)

# Fit the model
lr_scratch.fit(X_train, y_train)

# Compare with scikit-learn implementation
lr_sklearn = LogisticRegression(C=1 / (2 * len(X_train)), max_iter=1000)
lr_sklearn.fit(X_train, y_train)

# Print weights
print("Scratch Coefficients:", lr_scratch.coef_)
print("Scikit-learn Coefficients:", lr_sklearn.coef_)

# Use scikit-learn for evaluation
X_val = np.random.rand(20, 2)
y_val = np.random.randint(0, 2, size=20)

y_pred_scratch = lr_scratch.predict(X_val)
y_pred_sklearn = lr_sklearn.predict(X_val)

print("Scratch Accuracy:", accuracy_score(y_val, y_pred_scratch))
print("Scikit-learn Accuracy:", accuracy_score(y_val, y_pred_sklearn))

# Problem 6: Plot of learning curve
plt.plot(lr_scratch.loss, label="Training Loss")
plt.plot(lr_scratch.val_loss, label="Validation Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Problem 7: Visualization of decision area
# Visualization code depends on the number of features in your dataset. If it's 2D, you can plot the decision boundary.
# For simplicity, let's assume 2D data.
if X_train.shape[1] == 2:
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    # Plot decision boundary
    h = .02
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = lr_scratch.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.title("Decision Boundary")
    plt.show()

# Problem 8: (Advance assignment) Saving weights
# Saving weights using pickle
import pickle
with open("weights.pkl", "wb") as f:
    pickle.dump(lr_scratch.coef_, f)

# Saving weights using numpy
np.savez("weights.npz", coef=lr_scratch.coef_)
