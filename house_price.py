import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Problem 1
# Obtaining a dataset
df = pd.read_csv('train.csv')

# Problem 3
# Checking the data
print(df.info())
print(df.columns)
print(df.describe())

# Problem 4
# Dealing with missing values
missing_values = df.isnull().sum()
print(missing_values)

# Delete features with 5 or more missing values
df = df.dropna(thresh=len(df) - 5, axis=1)

# Drop samples with missing values
df = df.dropna()

# Problem 6
# Confirming distribution
sns.displot(df['objective_variable'])
sns.histplot(df['objective_variable'])
kurtosis = df['objective_variable'].kurtosis()
skewness = df['objective_variable'].skew()

# Perform a logarithmic transformation on the objective variable
df['log_objective_variable'] = np.log(df['objective_variable'])

# Display the distribution of the logarithmically transformed objective variable
sns.displot(df['log_objective_variable'])
sns.histplot(df['log_objective_variable'])
log_kurtosis = df['log_objective_variable'].kurtosis()
log_skewness = df['log_objective_variable'].skew()

# Problem 7
# Confirming the correlation coefficient
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')

# Select 10 features with high correlation with the target variable
high_corr_features = correlation_matrix.nlargest(10, 'objective_variable')['objective_variable'].index
high_corr_matrix = df[high_corr_features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(high_corr_matrix, annot=True, cmap='coolwarm')

# Summarize whether the 10 selected features represent something by referring to the description in Kaggle's DataDescription
# Find 3 combinations of the 10 selected features that have high correlation coefficients with each other
high_corr_features_combinations = high_corr_matrix.unstack().sort_values(ascending=False).drop_duplicates()
