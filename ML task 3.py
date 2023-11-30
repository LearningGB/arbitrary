import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class ScratchSVMClassifier():
    def __init__(self, num_iter, lr, kernel='linear', threshold=1e-5, verbose=False):
        self.iter = num_iter
        self.lr = lr
        self.kernel = kernel
        self.threshold = threshold
        self.verbose = verbose

    def fit(self, X, y, X_val=None, y_val=None):
        self.lam_sv = np.zeros(X.shape[0])
        self.index_support_vectors = np.arange(X.shape[0])
        self.X_sv = X
        self.y_sv = y

        if self.verbose:
            print(f"Number of support vectors: {len(self.index_support_vectors)}")

    def _update_lambda(self, lam, xi, yi, X):
        update = lam + self.lr * (1 - np.sum(lam * yi * self.y_sv * self._kernel(xi, X)))
        return np.maximum(0, update)

    def _kernel(self, xi, X):
        if self.kernel == 'linear':
            return np.dot(X, xi)
        elif self.kernel == 'polynomial':
            gamma = 1
            theta_0 = 0
            d = 2
            return (gamma * np.dot(X, xi) + theta_0) ** d

    def _is_support_vector(self, lam):
        return lam > self.threshold

    def predict(self, X):
        y_pred = np.sum(self.lam_sv[self.index_support_vectors] * self.y_sv[self.index_support_vectors]
                        * self._kernel(X, self.X_sv[self.index_support_vectors]), axis=1)
        y_pred = np.where(y_pred >= 0, 1, -1)
        return y_pred

# Problem 4
# Assume X_train, y_train, X_val, y_val are already defined

# Train Scratch SVM
scratch_svm = ScratchSVMClassifier(num_iter=100, lr=0.01, kernel='polynomial', threshold=1e-5, verbose=True)
scratch_svm.fit(X_train, y_train)

# Predict with Scratch SVM
y_val_pred_scratch = scratch_svm.predict(X_val)

# Train and predict with Scikit-learn SVM
from sklearn.svm import SVC

svm_sklearn = SVC(kernel='poly', degree=2, gamma=1, coef0=0)
svm_sklearn.fit(X_train, y_train)
y_val_pred_sklearn = svm_sklearn.predict(X_val)

# Compare accuracies
accuracy_scratch = accuracy_score(y_val, y_val_pred_scratch)
accuracy_sklearn = accuracy_score(y_val, y_val_pred_sklearn)

print(f"Accuracy - Scratch SVM: {accuracy_scratch:.4f}, Scikit-learn SVM: {accuracy_sklearn:.4f}")

# Problem 5: Visualization of decision area
def plot_decision_boundary(X, y, model, title="Decision Boundary"):
    h = .02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

# Visualize decision boundary for Scratch SVM
plot_decision_boundary(X_val, y_val, scratch_svm, title="Scratch SVM Decision Boundary")

# Visualize decision boundary for Scikit-learn SVM
plot_decision_boundary(X_val, y_val, svm_sklearn, title="Scikit-learn SVM Decision Boundary")
