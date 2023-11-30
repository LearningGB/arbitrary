import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import pickle

class ScratchLogisticRegression():
    def __init__(self, num_iter=1000, lr=0.01, bias=True, lambda_reg=0.1, verbose=False):
        self.iter = num_iter
        self.lr = lr
        self.bias = bias
        self.lambda_reg = lambda_reg
        self.verbose = verbose
        self.loss = np.zeros(self.iter)
        self.val_loss = np.zeros(self.iter)
        
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def _gradient_descent(self, X, error):
        regularization_term = (self.lambda_reg / len(X)) * self.theta[1:]
        self.theta -= self.lr * (np.dot(X.T, error) + regularization_term)
    
    def _calculate_loss(self, y_true, y_pred):
        regularization_term = (self.lambda_reg / (2 * len(y_true))) * np.sum(self.theta[1:]**2)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) + regularization_term
        return loss
    
    def fit(self, X, y, X_val=None, y_val=None):
        if self.bias:
            X = np.column_stack((np.ones(len(X)), X))
            if X_val is not None:
                X_val = np.column_stack((np.ones(len(X_val)), X_val))
        
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.iter):
            z = np.dot(X, self.theta)
            h = self._sigmoid(z)
            error = h - y
            self._gradient_descent(X, error)
            
            self.loss[i] = self._calculate_loss(y, h)
            
            if X_val is not None:
                z_val = np.dot(X_val, self.theta)
                h_val = self._sigmoid(z_val)
                self.val_loss[i] = self._calculate_loss(y_val, h_val)
            
            if self.verbose and i % 100 == 0:
                print(f"Iteration {i}, Training Loss: {self.loss[i]}")
                
    def predict_proba(self, X):
        if self.bias:
            X = np.column_stack((np.ones(len(X)), X))
        return self._sigmoid(np.dot(X, self.theta))
    
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

# Problem 5: Learning and Estimation
iris = load_iris()
X = iris.data[50:, :2]  # Take only two features for binary classification
y = (iris.target[50:] == 2).astype(int)  # 1 if virginica, 0 if versicolor

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Scratch Logistic Regression
scratch_lr = ScratchLogisticRegression(num_iter=1000, lr=0.01, lambda_reg=0.1, verbose=True)
scratch_lr.fit(X_train_scaled, y_train, X_val_scaled, y_val)

# Scikit-learn Logistic Regression
sklearn_lr = LogisticRegression(C=1 / 0.1, random_state=42)
sklearn_lr.fit(X_train_scaled, y_train)

# Compare Accuracy, Precision, and Recall
y_pred_scratch = scratch_lr.predict(X_val_scaled)
y_pred_sklearn = sklearn_lr.predict(X_val_scaled)

print("Scratch Logistic Regression:")
print(f"Accuracy: {accuracy_score(y_val, y_pred_scratch)}")
print(f"Precision: {precision_score(y_val, y_pred_scratch)}")
print(f"Recall: {recall_score(y_val, y_pred_scratch)}")

print("\nScikit-learn Logistic Regression:")
print(f"Accuracy: {accuracy_score(y_val, y_pred_sklearn)}")
print(f"Precision: {precision_score(y_val, y_pred_sklearn)}")
print(f"Recall: {recall_score(y_val, y_pred_sklearn)}")

# Problem 6: Plot of Learning Curve
plt.plot(scratch_lr.loss, label='Training Loss')
plt.plot(scratch_lr.val_loss, label='Validation Loss')
plt.title('Learning Curve')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Problem 7: Visualization of Decision Area
def plot_decision_boundary(X, y, model, title):
    h = .02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Plot decision boundary for Scratch Logistic Regression
plot_decision_boundary(X_val_scaled, y_val, scratch_lr, 'Decision Boundary - Scratch Logistic Regression')

# Plot decision boundary for Scikit-learn Logistic Regression
plot_decision_boundary(X_val_scaled, y_val, sklearn_lr, 'Decision Boundary - Scikit-learn Logistic Regression')

# Problem 8: Saving Weights
def save_weights(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model.theta, file)

def load_weights(model, filename):
    with open(filename, 'rb') as file:
        model.theta = pickle.load(file)

# Save weights of Scratch Logistic Regression
save_weights(scratch_lr, 'scratch_lr_weights.pkl')

# Load weights into a new instance of Scratch Logistic Regression
loaded_scratch_lr = ScratchLogisticRegression()
load_weights(loaded_scratch_lr, 'scratch_lr_weights.pkl')

# Test the loaded model
y_pred_loaded = loaded_scratch_lr.predict(X_val_scaled)

# Compare Accuracy, Precision, and Recall for the loaded model
print("\nLoaded Scratch Logistic Regression:")
print(f"Accuracy: {accuracy_score(y_val, y_pred_loaded)}")
print(f"Precision: {precision_score(y_val, y_pred_loaded)}")
print(f"Recall: {recall_score(y_val, y_pred_loaded)}")
