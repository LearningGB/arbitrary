import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample

class Blending:
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)

    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models])
        return np.mean(predictions, axis=1)

class Bagging:
    def __init__(self, model, n_estimators):
        self.model = model
        self.n_estimators = n_estimators

    def fit(self, X, y):
        self.models = []
        for _ in range(self.n_estimators):
            X_boot, y_boot = resample(X, y)
            model = self.model()
            model.fit(X_boot, y_boot)
            self.models.append(model)

    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models])
        return np.mean(predictions, axis=1)

class Stacking:
    def __init__(self, base_models, final_model):
        self.base_models = base_models
        self.final_model = final_model

    def fit(self, X, y):
        blend_data = np.column_stack([model.predict(X) for model in self.base_models])
        self.final_model.fit(blend_data, y)

    def predict(self, X):
        blend_test = np.column_stack([model.predict(X) for model in self.base_models])
        return self.final_model.predict(blend_test)

# Load dataset (House Prices: Advanced Regression Techniques)
# Assuming X contains GrLivArea and YearBuilt, and y contains SalePrice
# Replace this part with your dataset loading logic
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Example usage
# You need to replace X_train, y_train, X_val, y_val with your actual dataset
# For simplicity, I am using a placeholder linear regression model
base_model = LinearRegression
blending_models = [base_model(), base_model(), base_model()]
bagging_model = Bagging(model=base_model, n_estimators=5)
stacking_base_models = [base_model(), base_model(), base_model()]
stacking_final_model = base_model()

# Blending
blending = Blending(models=blending_models)
blending.fit(X_train, y_train)
blending_predictions = blending.predict(X_val)
blending_mse = mean_squared_error(y_val, blending_predictions)
print(f'Blending MSE: {blending_mse}')

# Bagging
bagging = Bagging(model=base_model, n_estimators=5)
bagging.fit(X_train, y_train)
bagging_predictions = bagging.predict(X_val)
bagging_mse = mean_squared_error(y_val, bagging_predictions)
print(f'Bagging MSE: {bagging_mse}')

# Stacking
stacking = Stacking(base_models=stacking_base_models, final_model=stacking_final_model)
stacking.fit(X_train, y_train)
stacking_predictions = stacking.predict(X_val)
stacking_mse = mean_squared_error(y_val, stacking_predictions)
print(f'Stacking MSE: {stacking_mse}')
