import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load iris dataset
iris = datasets.load_iris()
X = iris.data[:, [0, 2]]  # Selecting two features for visualization
y = iris.target

# Problem 1: Select features and categories for practice
# Choosing virgicolor and virginica, sepal_length and petal_length
X_binary = X[np.isin(y, [1, 2])]
y_binary = y[np.isin(y, [1, 2])]

# Problem 2: Data analysis
# Scatter plot, boxplot, violinplot
df = pd.DataFrame(data=np.column_stack((X_binary, y_binary)), columns=['sepal_length', 'petal_length', 'species'])
df['species'] = df['species'].astype(int)

# Problem 3: Division of preprocessing/training data and verification data
X_train, X_val, y_train, y_val = train_test_split(X_binary, y_binary, test_size=0.25, random_state=42)

# Problem 4: Pretreatment/Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Problem 5: Learning and estimation with k-nn
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
y_val_pred_knn = knn_model.predict(X_val_scaled)

# Problem 6: Evaluation
accuracy_knn = accuracy_score(y_val, y_val_pred_knn)
precision_knn = precision_score(y_val, y_val_pred_knn)
recall_knn = recall_score(y_val, y_val_pred_knn)
f1_knn = f1_score(y_val, y_val_pred_knn)
confusion_mat_knn = confusion_matrix(y_val, y_val_pred_knn)

print(f'k-nn - Accuracy: {accuracy_knn:.4f}, Precision: {precision_knn:.4f}, Recall: {recall_knn:.4f}, F1: {f1_knn:.4f}')
print('Confusion Matrix:')
print(confusion_mat_knn)

# Problem 7: Visualization
def decision_region(X, y, model, step=0.01, title='Decision Region', xlabel='xlabel', ylabel='ylabel',
                    target_names=['versicolor', 'virginica']):
    # Function implementation (same as provided in the text)

# Plot decision region for k-nn
decision_region(X_train_scaled, y_train, knn_model, title='k-nn Decision Region')

# Problem 8: Learning by other methods
models = {
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_val_pred = model.predict(X_val_scaled)

    # Problem 6: Evaluation
    accuracy = accuracy_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred)
    recall = recall_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)
    confusion_mat = confusion_matrix(y_val, y_val_pred)

    print(f'{model_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
    print('Confusion Matrix:')
    print(confusion_mat)

    # Problem 7: Visualization
    decision_region(X_train_scaled, y_train, model, title=f'{model_name} Decision Region')

# Problem 9: (Advanced task) Comparison with and without standardization
# Problem 9: (Advanced task) Comparison with and without standardization
results_no_standardization = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)

    # Problem 6: Evaluation
    accuracy = accuracy_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred)
    recall = recall_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)
    confusion_mat = confusion_matrix(y_val, y_val_pred)

    results_no_standardization[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Confusion Matrix': confusion_mat
    }

    print(f'{model_name} (No Standardization) - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, '
          f'Recall: {recall:.4f}, F1: {f1:.4f}')
    print('Confusion Matrix:')
    print(confusion_mat)

# Problem 10: (Advance assignment) Highly accurate method using all objective variables
X_all = iris.data
y_all = iris.target

X_train_all, X_val_all, y_train_all, y_val_all = train_test_split(X_all, y_all, test_size=0.25, random_state=42)

# Choose a model (e.g., Random Forest) for multi-value classification
rf_model_all = RandomForestClassifier()
rf_model_all.fit(X_train_all, y_train_all)
y_val_pred_all = rf_model_all.predict(X_val_all)

# Problem 6: Evaluation
accuracy_all = accuracy_score(y_val_all, y_val_pred_all)
precision_all = precision_score(y_val_all, y_val_pred_all, average='weighted')
recall_all = recall_score(y_val_all, y_val_pred_all, average='weighted')
f1_all = f1_score(y_val_all, y_val_pred_all, average='weighted')
confusion_mat_all = confusion_matrix(y_val_all, y_val_pred_all)

print(f'Random Forest (All Variables) - Accuracy: {accuracy_all:.4f}, Precision: {precision_all:.4f}, '
      f'Recall: {recall_all:.4f}, F1: {f1_all:.4f}')
print('Confusion Matrix:')
print(confusion_mat_all)

# Visualization for Random Forest (All Variables)
decision_region(X_train_all, y_train_all, rf_model_all, title='Random Forest Decision Region (All Variables)')

# Display results without standardization for comparison
print('\nResults without Standardization:')
print(results_no_standardization)
