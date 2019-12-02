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

# Fitting Linear Regression to the data
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the data
from sklearn.preprocessing import PolynomialFeatures
# transform the original matrix X into new X_poly with original independent variable positon levels and associated polynomial terms
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
# create new linear regression model to fit to x_poly
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

# Visualize original salary data
plt.plot(X, y, 'r.', label = "Salary")
# Visualizing the Linear Regression results
plt.plot(X, lin_reg.predict(X), 'b-', label = "Lin. Reg.")
# Visualizing the Polynomial Regression results
X_grid = np.arange(min(X), max(X), step = 0.1) # creates a vector from min X to max X at 0.1 intervals
X_grid = X_grid.reshape((len(X_grid), 1)) # reshape the vector to an array
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), 'g-', label = "Poly. Reg.") # we do not use X_poly as it was already defined
plt.title("Salary vs Position Level")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.legend(loc = "upper left")
plt.show

# Predict a new result with Linear Regression
lin_reg.predict(np.reshape(6.5, (1,1))) # predict position level 6.5. Reshape makes prediction into expected array format
lin_reg.predict(np.array([[6.5]])) # similar method as above

# Predict a new result with Polynomial Regression
lin_reg2.predict(poly_reg.fit_transform(np.reshape(6.5, (1,1))))
lin_reg2.predict(poly_reg.fit_transform(np.array([[6.5]])))