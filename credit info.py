# Import libraries
import pandas as pd
import numpy as np

# Load the training and testing data
train_data = pd.read_csv("home-credit-default-risk/train.csv")
test_data = pd.read_csv("home-credit-default-risk/test.csv")

# Check the data for missing values
print("Number of missing values in training data:", train_data.isnull().sum())
print("Number of missing values in testing data:", test_data.isnull().sum())

# Preprocess the data
# Replace missing values in numerical features with the mean of the corresponding column
train_data.fillna(train_data.mean(), inplace=True)
test_data.fillna(test_data.mean(), inplace=True)

# One-hot encode categorical features
train_data = pd.get_dummies(train_data)
test_data = pd.get_dummies(test_data)

# Separate target variable and features
X_train = train_data.drop("TARGET", axis=1)
y_train = train_data["TARGET"]
X_test = test_data

# Create a baseline model using logistic regression
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model on the training data
from sklearn.metrics import roc_auc_score

y_pred = model.predict(X_train)
print("AUC score on training data:", roc_auc_score(y_train, y_pred))

# Make predictions on the test data
y_pred = model.predict(X_test)

# Create a submission file
submission_data = pd.DataFrame({"ID": test_data["ID"], "TARGET": y_pred})
submission_data.to_csv("submission.csv", index=False)
