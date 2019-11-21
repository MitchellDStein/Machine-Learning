# Linear Regression

# Viewing the dataset
# ===================
# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Reading Data
dataset = pd.read_csv(r'Regression\Simple Linear Regression\Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into test and trainging sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3)

# Visualizing data using pyplot
plt.plot(X, y, 'g-', label = "Dataset")
plt.plot(X_train, y_train, 'r+', label="Training Set")
plt.plot(X_test, y_test, 'bx', label="Test Set")
plt.legend(loc = "upper left")
plt.show

# Feature Scaling
