# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Problem 1: Understanding the Competition
print("Competition Name:", "Home Credit Default Risk")

# Load the required datasets
application_data = pd.read_csv('application_train.csv')
description_data = pd.read_csv('HomeCredit_columns_description.csv')

# Problem 2: Understanding the Overview of Data
print("\nHead of Application Data:")
print(application_data.head())

print("\nInformation about Application Data:")
print(application_data.info())

print("\nDescription of Application Data:")
print(application_data.describe())

# Check for missing values
missing_values = application_data.isnull().sum()
print("\nMissing Values:")
print(missing_values)

# Draw a graph showing the percentage of classes
application_data['target'].value_counts().plot(kind='bar', title='Percentage of Classes')
plt.show()

# Problem 3: Defining Issues
# Set multiple issues/questions based on data overview
# Example: What is the distribution of the target variable?
# Example: Are there any correlations between features?

# Problem 4: Data Exploration
# Create at least 5 tables and graphs for exploration

# Plot a correlation matrix
correlation_matrix = application_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Explore the distribution of age
plt.figure(figsize=(10, 6))
sns.histplot(application_data['DAYS_BIRTH'] // -365, bins=30, kde=False)
plt.title('Distribution of Age')
plt.xlabel('Age (Years)')
plt.ylabel('Count')
plt.show()

