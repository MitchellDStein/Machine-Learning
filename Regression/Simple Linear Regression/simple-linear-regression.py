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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Visualizing data using pyplot
plt.plot(X, y, 'g-', label = "Dataset")
plt.plot(X_train, y_train, 'r+', label="Training Set")
plt.plot(X_test, y_test, 'bx', label="Test Set")
plt.title("Salary vs Experience (Training Set)")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.legend(loc = "upper left")
plt.show

# Fitting Simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_prediction = regressor.predict(X_test)

# Visualizing Trainging set results
plt.plot(X_train, y_train, 'ro', label="Training Set")
plt.plot(X_train, regressor.predict(X_train), 'b-', label = 'Predictions')
plt.title("Salary vs Experience (Training Set)")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.legend(loc = "upper left")
plt.show

# Visualizing Test set results
plt.plot(X_test, y_test, 'ro', label="Testing Set")
plt.plot(X_train, regressor.predict(X_train), 'b-', label = 'Predictions')
plt.title("Salary vs Experience (Test Set)")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.legend(loc = "upper left")
plt.show