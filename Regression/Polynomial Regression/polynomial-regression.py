# Polynomial Regression

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import the dataset
dataset = pd.read_csv(r"Regression/Polynomial Regression/Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values # we want to keeo X as a matrix/array
y = dataset.iloc[:, 2].values

# We do not need to split into a training and test set because we have such a small dataset

plt.plot(X, y, 'r.', label = "Salary")
plt.title("Salary vs Position Level")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.legend(loc = "upper left")
plt.show

# Fitting Linear Regression to the data
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the data
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X)