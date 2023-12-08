import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# Load the dataset
train_df = pd.read_csv('data/house_price/train.csv')

# Prepare the dataset
input_cols = ['GrLivArea', 'YearBuilt']
target = 'SalePrice'
train_df[target] = np.log(train_df[target])

# Split the dataset
train_set, test_set = train_test_split(train_df, test_size=0.2, shuffle=True, random_state=42)

# Define models
models = [
    LinearRegression(),
    DecisionTreeRegressor(),
    SVR()
]

model_names = ['linear_reg', 'dt', 'svr']

# Initialize variables
metrics = {name: [] for name in model_names}
preds = {name: [] for name in model_names}
oofs = []

# Train models and calculate metrics
for name, model in zip(model_names, models):
    reg = model.fit(train_set[input_cols], train_set[target])
    oofs.append(reg.predict(train_set[input_cols]))
    pred = reg.predict(test_set[input_cols])
    score = mean_squared_error(test_set[target], pred)
    metrics[name].append(score)
    preds[name] = pred

# Display metrics for each individual model
for name in metrics.keys():
    print(f'Model {name}:', np.round(np.mean(metrics[name]), 3))

# Blending
weights = [0.4, 0.2, 0.4]
final_pred = np.sum([weights[i] * preds[name] for i, name in enumerate(model_names)], axis=0)
blending_score = mean_squared_error(test_set[target], final_pred)
print(f'Blending:', np.round(blending_score, 3))

# Stacking
stacking_model = LinearRegression()
X_pred = np.asarray(oofs).T
X_test_pred = np.asarray([preds[name] for name in model_names]).T

reg = stacking_model.fit(X_pred, train_set[target])
pred_stack = reg.predict(X_test_pred)

stacking_score = mean_squared_error(test_set[target], pred_stack)
print(f'Stacking:', np.round(stacking_score, 3))

# Bagging
bagging_runs = 5
bagging_metrics = {name: [] for name in model_names}
bagging_preds = {name: np.zeros(len(test_set)) for name in model_names}

for _ in range(bagging_runs):
    frac = np.random.uniform(0.8, 0.9)
    train_subset = train_set.sample(frac=frac)
    
    for name, model in zip(model_names, models):
        reg = model.fit(train_subset[input_cols], train_subset[target])
        pred = reg.predict(test_set[input_cols])
        bagging_preds[name] += pred / bagging_runs
        score = mean_squared_error(test_set[target], pred)
        bagging_metrics[name].append(score)

# Display metrics for each individual model in bagging
for name in bagging_metrics.keys():
    print(f'Bagging Model {name}:', np.round(np.mean(bagging_metrics[name]), 3))

# Blending on bagging predictions
bagging_weights = [0.4, 0.2, 0.4]
final_bagging_pred = np.sum([bagging_weights[i] * bagging_preds[name] for i, name in enumerate(model_names)], axis=0)
bagging_blending_score = mean_squared_error(test_set[target], final_bagging_pred)
print(f'Bagging and Blending:', np.round(bagging_blending_score, 3))
